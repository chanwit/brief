// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

package engine

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/chanwit/brief/ivf"
)

// IndexSchemaVersion bumps whenever the on-disk layout changes in a way that
// older binaries can't load. Queries refuse to run against a newer schema.
const IndexSchemaVersion = 2

// Chunk is a section of a knowledge file. Vector is omitted from JSON when
// the index uses IVF — vectors live in the sibling .ivf/ directory in that
// case.
type Chunk struct {
	File     string         `json:"file"`
	Title    string         `json:"title"`
	Body     string         `json:"body"`
	Vector   []float32      `json:"vector,omitempty"`
	TermFreq map[string]int `json:"tf"`
	DocLen   int            `json:"doc_len"`
}

// Index is the serialized vector + BM25 store. ModelInfo and Config are
// embedded so queries can verify and rebuild the exact embedding pipeline.
type Index struct {
	Schema    int            `json:"schema_version"`
	ModelInfo ModelInfo      `json:"model_info"`
	Config    IndexConfig    `json:"config"`
	Chunks    []Chunk        `json:"chunks"`
	DocFreq   map[string]int `json:"df"`
	AvgDocLen float64        `json:"avg_dl"`

	// Transient — not serialized. Populated by LoadIndex when the sibling
	// <path>.ivf/ directory exists and the index was built with UseIVF.
	IVFIndex *ivf.IVFFlat `json:"-"`
}

func LoadIndex(path string) (*Index, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read index: %w", err)
	}
	var idx Index
	if err := json.Unmarshal(data, &idx); err != nil {
		return nil, fmt.Errorf("parse index: %w", err)
	}
	if idx.Schema == 0 {
		return nil, fmt.Errorf("index %s has no schema_version — rebuild with this version of brief", path)
	}
	if idx.Schema > IndexSchemaVersion {
		return nil, fmt.Errorf("index %s uses schema v%d, this binary understands up to v%d",
			path, idx.Schema, IndexSchemaVersion)
	}
	if idx.ModelInfo.Key == "" {
		return nil, fmt.Errorf("index %s is missing model_info", path)
	}

	// Attach the mmap'd IVF companion index when one is present.
	if idx.Config.UseIVF {
		ivfDir := IVFSiblingPath(path)
		if !FileExists(ivfDir) {
			return nil, fmt.Errorf("index %s was built with use_ivf=true but %s is missing",
				path, ivfDir)
		}
		IVFIndex, err := ivf.Open(ivfDir)
		if err != nil {
			return nil, fmt.Errorf("open ivf: %w", err)
		}
		idx.IVFIndex = IVFIndex
	}
	return &idx, nil
}

// Close releases any resources held by the index, including mmap regions
// for the IVF companion. Safe to call multiple times; safe on an index
// without an IVF attached.
func (idx *Index) Close() error {
	if idx == nil || idx.IVFIndex == nil {
		return nil
	}
	err := idx.IVFIndex.Close()
	idx.IVFIndex = nil
	return err
}

// IVFSiblingPath returns the conventional location of the IVF directory
// for a given index JSON path: <path>.ivf/ .
func IVFSiblingPath(indexPath string) string { return indexPath + ".ivf" }

// autoIVFCentroids picks a reasonable K when the user doesn't specify one.
// The classical rule of thumb for IVF is K ≈ 4·√N.
func autoIVFCentroids(n int) int {
	if n <= 0 {
		return 16
	}
	k := int(4 * math.Sqrt(float64(n)))
	if k < 16 {
		k = 16
	}
	if k > n {
		k = n
	}
	return k
}

func SaveIndex(idx *Index, path string) error {
	data, err := json.MarshalIndent(idx, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

// BuildIndex computes BM25 statistics and embeddings for every chunk and
// returns a ready-to-serialize Index. Shared by cmdIndex, the tuner, and
// tests so all three go through one code path.
func BuildIndex(chunks []Chunk, emb *Embedder, cfg IndexConfig) *Index {
	docFreq := make(map[string]int)
	totalLen := 0
	for i := range chunks {
		terms := bm25Tokenize(chunks[i].Title + "\n" + chunks[i].Body)
		chunks[i].DocLen = len(terms)
		totalLen += len(terms)

		tf := make(map[string]int)
		seen := make(map[string]bool)
		for _, t := range terms {
			tf[t]++
			if !seen[t] {
				docFreq[t]++
				seen[t] = true
			}
		}
		chunks[i].TermFreq = tf
	}

	avgDocLen := 0.0
	if len(chunks) > 0 {
		avgDocLen = float64(totalLen) / float64(len(chunks))
	}

	cap := cfg.EmbedMaxChars
	for i := range chunks {
		text := chunks[i].Title + "\n" + chunks[i].Body
		if cap > 0 && len(text) > cap {
			text = text[:cap]
		}
		chunks[i].Vector = emb.Embed(text)
	}

	return &Index{
		Schema:    IndexSchemaVersion,
		ModelInfo: emb.Info,
		Config:    cfg,
		Chunks:    chunks,
		DocFreq:   docFreq,
		AvgDocLen: avgDocLen,
	}
}

// BuildIVFFromIndex trains an IVF-Flat over every chunk vector in idx and
// writes it to `dir`. Must be called BEFORE the caller strips vectors off
// the chunks for serialization.
func BuildIVFFromIndex(idx *Index, dir string, cfg IndexConfig) (*ivf.IVFFlat, error) {
	n := len(idx.Chunks)
	if n == 0 {
		return nil, fmt.Errorf("BuildIVFFromIndex: no chunks")
	}
	dim := idx.ModelInfo.Dim

	k := cfg.IVFCentroids
	if k == 0 {
		k = autoIVFCentroids(n)
	}
	if k > n {
		k = n
	}
	iters := cfg.IVFKmeansIt
	if iters == 0 {
		iters = 20
	}
	nprobe := cfg.IVFNprobe
	if nprobe == 0 {
		// Probe ~√K by default; cheap and hits ~90% recall on typical data.
		nprobe = int(math.Sqrt(float64(k)))
		if nprobe < 4 {
			nprobe = 4
		}
		if nprobe > k {
			nprobe = k
		}
	}

	// Flatten vectors for training + add.
	flat := make([]float32, 0, n*dim)
	ids := make([]uint64, n)
	for i := range idx.Chunks {
		vec := idx.Chunks[i].Vector
		if len(vec) != dim {
			return nil, fmt.Errorf("chunk %d vector dim %d != index dim %d",
				i, len(vec), dim)
		}
		flat = append(flat, vec...)
		ids[i] = uint64(i)
	}

	ix := ivf.New(dim, k)
	ix.Nprobe = nprobe
	if err := ix.Train(flat, iters, 42); err != nil {
		return nil, fmt.Errorf("ivf.Train: %w", err)
	}
	if err := ix.AddBatch(ids, flat); err != nil {
		return nil, fmt.Errorf("ivf.AddBatch: %w", err)
	}
	if err := ix.Save(dir); err != nil {
		return nil, fmt.Errorf("ivf.Save: %w", err)
	}
	// Caller will close the in-memory instance; attach it so tests and
	// the same-process cmdIndex can use it immediately.
	return ix, nil
}

// StripChunkVectors drops every chunk's Vector slice so the JSON stays
// small. Used right before saving when UseIVF=true.
func StripChunkVectors(idx *Index) {
	for i := range idx.Chunks {
		idx.Chunks[i].Vector = nil
	}
}

// ParseKnowledge walks dir, applies include/exclude globs, and chunks every
// matching file using the configured chunking strategy.
func ParseKnowledge(dir string, cfg IndexConfig) []Chunk {
	var chunks []Chunk
	includes := cfg.Include
	if len(includes) == 0 {
		includes = []string{"*"}
	}
	filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		rel, _ := filepath.Rel(dir, path)
		base := filepath.Base(rel)
		if !matchAnyGlob(base, rel, includes) {
			return nil
		}
		if matchAnyGlob(base, rel, cfg.Exclude) {
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return nil
		}
		var sections []Chunk
		switch cfg.ChunkStrategy {
		case "size":
			sections = splitBySize(string(data), rel, cfg.ChunkSize, cfg.ChunkOverlap)
		default:
			sections = splitSections(string(data), rel)
		}
		for _, s := range sections {
			if cfg.MinChunkChars > 0 && len(s.Body) < cfg.MinChunkChars {
				continue
			}
			if cfg.MaxChunkChars > 0 && len(s.Body) > cfg.MaxChunkChars {
				s.Body = s.Body[:cfg.MaxChunkChars]
			}
			chunks = append(chunks, s)
		}
		return nil
	})
	return chunks
}

func matchAnyGlob(base, rel string, patterns []string) bool {
	for _, p := range patterns {
		if ok, _ := filepath.Match(p, base); ok {
			return true
		}
		if ok, _ := filepath.Match(p, rel); ok {
			return true
		}
	}
	return false
}

// splitSections breaks a markdown file on "## " headings, respecting fenced code blocks.
func splitSections(text, file string) []Chunk {
	lines := strings.Split(text, "\n")
	var chunks []Chunk
	currentTitle := strings.TrimSuffix(filepath.Base(file), filepath.Ext(file))
	var body []string
	inFence := false

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "```") || strings.HasPrefix(trimmed, "~~~") {
			inFence = !inFence
			body = append(body, line)
			continue
		}
		if inFence {
			body = append(body, line)
			continue
		}
		if strings.HasPrefix(line, "## ") {
			if len(body) > 0 || currentTitle != "" {
				joined := strings.TrimSpace(strings.Join(body, "\n"))
				if joined != "" {
					chunks = append(chunks, Chunk{File: file, Title: currentTitle, Body: joined})
				}
			}
			currentTitle = strings.TrimPrefix(line, "## ")
			body = nil
		} else {
			body = append(body, line)
		}
	}
	if len(body) > 0 {
		joined := strings.TrimSpace(strings.Join(body, "\n"))
		if joined != "" {
			chunks = append(chunks, Chunk{File: file, Title: currentTitle, Body: joined})
		}
	}
	return chunks
}

// splitBySize produces overlapping chunks of approximately size chars each.
// Used for non-markdown corpora or when you want consistent chunk granularity.
func splitBySize(text, file string, size, overlap int) []Chunk {
	if size <= 0 {
		size = 500
	}
	if overlap < 0 {
		overlap = 0
	}
	if overlap >= size {
		overlap = size / 2
	}
	step := size - overlap
	n := len(text)
	var chunks []Chunk
	idx := 1
	for start := 0; start < n; start += step {
		end := start + size
		if end > n {
			end = n
		}
		body := strings.TrimSpace(text[start:end])
		if body != "" {
			chunks = append(chunks, Chunk{
				File:  file,
				Title: fmt.Sprintf("chunk-%d", idx),
				Body:  body,
			})
			idx++
		}
		if end >= n {
			break
		}
	}
	return chunks
}

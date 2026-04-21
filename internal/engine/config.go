// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

package engine

import (
	"encoding/json"
	"fmt"
	"os"
)

// IndexConfig captures every knob that controls how a corpus is turned into
// an index. Stored inside the index header so queries and the tuner can see
// exactly how it was built.
type IndexConfig struct {
	ModelKey      string   `json:"model"`
	ChunkStrategy string   `json:"chunk_strategy"` // "heading" or "size"
	ChunkSize     int      `json:"chunk_size"`     // target chars per chunk (size strategy)
	ChunkOverlap  int      `json:"chunk_overlap"`  // char overlap between adjacent chunks
	MinChunkChars int      `json:"min_chunk_chars"`
	MaxChunkChars int      `json:"max_chunk_chars"` // 0 = unbounded
	EmbedMaxChars int      `json:"embed_max_chars"` // truncate body before tokenizing
	Include       []string `json:"include"`         // file globs (basename match)
	Exclude       []string `json:"exclude"`
	Pooling       string   `json:"pooling,omitempty"`   // override model default
	Normalize     *bool    `json:"normalize,omitempty"` // override model default

	// IVF-Flat semantic index. When UseIVF is true, chunk vectors are
	// stored in a sibling `<index>.ivf/` directory (mmap'd at query time)
	// and omitted from the main JSON — which matters once the corpus
	// exceeds ~100k chunks.
	UseIVF       bool `json:"use_ivf,omitempty"`
	IVFCentroids int  `json:"ivf_centroids,omitempty"` // 0 = auto: max(16, 4√N)
	IVFKmeansIt  int  `json:"ivf_kmeans_iters,omitempty"` // 0 = 20
	IVFNprobe    int  `json:"ivf_nprobe,omitempty"` // default probe count baked into the IVF manifest; 0 = auto

	// Stem runs every BM25 token through a Porter2 English stemmer at
	// both index and query time so "refresh", "refreshing", and
	// "refreshed" collapse to one key. Off by default — flipping it
	// can subtly change retrieval on non-English or heavily
	// identifier-based corpora. Must match between learn and recall;
	// the value is read back from the loaded index.
	Stem bool `json:"stem,omitempty"`

	// ParseFrontmatter tells the chunker to read a leading
	//   ---\n<yaml>\n---\n
	// block from each markdown file, using `title` as the root
	// section title, folding `aliases` into the chunk's title terms
	// (so they get TitleBoost at recall), and assigning `tags` to
	// every chunk from that file. Default true; set to false to
	// treat frontmatter-ish text as regular body content.
	ParseFrontmatter bool `json:"parse_frontmatter,omitempty"`
}

func DefaultIndexConfig() IndexConfig {
	return IndexConfig{
		ModelKey:      DefaultModelKey,
		ChunkStrategy: "heading",
		ChunkSize:     500,
		ChunkOverlap:  100,
		MinChunkChars: 0,
		MaxChunkChars: 0,
		EmbedMaxChars: 1500,
		Include:       []string{"*.md", "*.markdown", "*.txt"},
		Exclude:       nil,
		// Stemming defaults off: safe for all corpora, opt-in lift
		// for English prose. Set to true in IndexConfig JSON or via
		// --stem at learn time.
		Stem: false,
		// Frontmatter parsing on by default. Files without a leading
		// --- YAML --- block are parsed normally, so this is a no-op
		// for non-Obsidian corpora.
		ParseFrontmatter: true,
	}
}

// QueryConfig captures every query-time knob. The tuner searches over this
// whole struct; the `query` command can load one via --config.
type QueryConfig struct {
	Mode               string  `json:"mode"` // "hybrid", "bm25", "semantic"
	K                  int     `json:"k"`
	WeightSemantic     float64 `json:"weight_semantic"`
	WeightBM25         float64 `json:"weight_bm25"`
	BM25K1             float64 `json:"bm25_k1"`
	BM25B              float64 `json:"bm25_b"`
	SemanticHardFloor  float64 `json:"semantic_hard_floor"`
	SemanticSoftFloor  float64 `json:"semantic_soft_floor"`
	BM25MinForSoftZone float64 `json:"bm25_min_for_soft_zone"`
	MinQueryTerms      int     `json:"min_query_terms_in_corpus"` // 0 = auto

	// IVF-only query knobs (ignored when the index has no IVF).
	// Nprobe    = how many centroids to probe (higher = better recall, slower).
	// NSemantic = size of the IVF shortlist; BM25 still scans all chunks,
	//             but semantic scores only contribute to this subset. 0 = auto.
	Nprobe    int `json:"nprobe,omitempty"`
	NSemantic int `json:"n_semantic,omitempty"`

	// Wikilink expansion: after selecting the top-K primary hits,
	// follow each hit's [[wikilinks]] one hop and surface the linked
	// chunks as additional results (marked with LinkedFrom). Gives
	// extra context for human-curated knowledge graphs (Obsidian,
	// Logseq) without running any embedder.
	// MaxLinked caps the total number of linked chunks added across
	// all primary hits. 0 disables the expansion entirely.
	MaxLinked int `json:"max_linked,omitempty"`

	// TitleBoost multiplies the effective term frequency for terms
	// that appear in a chunk's title (or in a frontmatter alias).
	// Effective tf = TermFreq[t] + (TitleBoost-1) * TitleTermFreq[t],
	// so TitleBoost=1 reduces to vanilla BM25 and higher values
	// progressively privilege title/alias hits. Default ~2.5 on new
	// indexes; has no effect on indexes that don't carry
	// TitleTermFreq (pre-v0.3).
	TitleBoost float64 `json:"title_boost,omitempty"`

	// RequireTags filters the search space to chunks whose Tags field
	// contains at least one of these values. Empty disables filtering.
	// Useful when an agent knows the domain of the user's question
	// and wants to narrow retrieval to one section of the vault.
	RequireTags []string `json:"require_tags,omitempty"`
}

// DefaultQueryConfig returns the baked-in best-known hybrid hyperparameters.
//
// Provenance: these values come from TestScenariosTuneToFullHitRate —
// random search over the sampleQueryConfig space against an 18-query
// eval set on the cub-claude scenarios corpus (8 files, 187 chunks).
// The search optimized hit_rate@5 and the winning trial scored
// hit@5=1.0 / MRR=1.0 on both the flat and IVF backends.
//
// Caveats:
//   - Tuned on one English technical-docs corpus; other domains may
//     benefit from re-tuning via `brief tune recall`.
//   - The canonical BM25 textbook values are k1=1.2, b=0.75; our k1/b
//     drifted upward because the scenarios corpus has short, term-dense
//     sections. If you see poor recall on long prose, run tune-query or
//     override --bm25-k1 / --bm25-b from the CLI.
//   - The semantic floors (hard/soft) are deliberately tighter than the
//     old 0.20/0.30 defaults — this rejects low-cosine noise earlier
//     and improves precision without hurting hit-rate on our eval set.
func DefaultQueryConfig() QueryConfig {
	return QueryConfig{
		Mode: "hybrid",
		// K=3 is chosen for the primary use case — coding-agent hooks
		// that inject retrieved chunks back into the prompt. Three
		// chunks cover most queries without blowing the context budget.
		// CLI power users can still pass -k N.
		K:                  3,
		WeightSemantic:     0.48,
		WeightBM25:         0.52,
		BM25K1:             2.33,
		BM25B:              0.97,
		SemanticHardFloor:  0.35,
		SemanticSoftFloor:  0.43,
		BM25MinForSoftZone: 0.44,
		MinQueryTerms:      0,
		// Wikilink expansion adds at most 3 linked chunks total.
		// Small enough to keep hook-mode output bounded, big enough to
		// catch the most-referenced notes in a small Obsidian vault.
		MaxLinked: 3,
		// Title hits count 2.5× body hits. Gentle enough to never
		// hurt, strong enough to visibly help on corpora with
		// informative section titles.
		TitleBoost: 2.5,
	}
}

func LoadQueryConfig(path string) (QueryConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return QueryConfig{}, err
	}
	cfg := DefaultQueryConfig()
	if err := json.Unmarshal(data, &cfg); err != nil {
		return QueryConfig{}, fmt.Errorf("parse %s: %w", path, err)
	}
	return cfg, nil
}

func SaveQueryConfig(cfg QueryConfig, path string) error {
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func LoadIndexConfig(path string) (IndexConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return IndexConfig{}, err
	}
	cfg := DefaultIndexConfig()
	if err := json.Unmarshal(data, &cfg); err != nil {
		return IndexConfig{}, fmt.Errorf("parse %s: %w", path, err)
	}
	return cfg, nil
}

func SaveIndexConfig(cfg IndexConfig, path string) error {
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"
	ort "github.com/yalue/onnxruntime_go"

	"github.com/chanwit/brief/internal/engine"
)

var learnFlags struct {
	from       string
	output     string
	configPath string
	model      string
	embedder   string

	chunkStrategy string
	chunkSize     int
	chunkOverlap  int
	minChunk      int
	maxChunk      int
	embedMax      int
	include       string
	exclude       string
	pooling       string

	useIVF           bool
	noIVF            bool
	autoIVFThreshold int
	ivfCentroids     int
	ivfIters         int
	ivfNprobe        int

	stem            bool
	noFrontmatter   bool
}

var learnCmd = &cobra.Command{
	Use:   "learn",
	Short: "Learn from a knowledge directory (builds the index)",
	Long: `Scans a directory of documents, chunks them, embeds each chunk with
the configured ONNX model, and writes a JSON index. By default IVF-Flat
ANN kicks in once the corpus crosses --auto-ivf-threshold chunks.

With no flags, learn looks for knowledge at $BRIEF_KNOWLEDGE, then
./.claude/knowledge, then ./knowledge, then ./docs. It writes to
./.brief/index.json.`,
	RunE: runLearn,
}

func runLearn(cmd *cobra.Command, args []string) error {
	if learnFlags.from == "" {
		learnFlags.from = locateKnowledge()
	}
	if learnFlags.from == "" {
		errNotFound("from", "BRIEF_KNOWLEDGE", knowledgeLookupOrder) // exits
	}
	if learnFlags.output == "" {
		learnFlags.output = DefaultIndexOutput
	}
	if dir := filepath.Dir(learnFlags.output); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("create output dir %s: %w", dir, err)
		}
	}

	cfg := engine.DefaultIndexConfig()
	if learnFlags.configPath != "" {
		c, err := engine.LoadIndexConfig(learnFlags.configPath)
		if err != nil {
			return err
		}
		cfg = c
	}
	if learnFlags.model != "" {
		cfg.ModelKey = learnFlags.model
	}
	if learnFlags.chunkStrategy != "" {
		cfg.ChunkStrategy = learnFlags.chunkStrategy
	}
	if learnFlags.chunkSize > 0 {
		cfg.ChunkSize = learnFlags.chunkSize
	}
	if learnFlags.chunkOverlap >= 0 {
		cfg.ChunkOverlap = learnFlags.chunkOverlap
	}
	if learnFlags.minChunk >= 0 {
		cfg.MinChunkChars = learnFlags.minChunk
	}
	if learnFlags.maxChunk >= 0 {
		cfg.MaxChunkChars = learnFlags.maxChunk
	}
	if learnFlags.embedMax > 0 {
		cfg.EmbedMaxChars = learnFlags.embedMax
	}
	if learnFlags.include != "" {
		cfg.Include = splitCSV(learnFlags.include)
	}
	if learnFlags.exclude != "" {
		cfg.Exclude = splitCSV(learnFlags.exclude)
	}
	if learnFlags.pooling != "" {
		cfg.Pooling = learnFlags.pooling
	}
	if cmd.Flags().Changed("stem") {
		cfg.Stem = learnFlags.stem
	}
	if learnFlags.noFrontmatter {
		cfg.ParseFrontmatter = false
	}

	// Mode-conditional default: --embedder none enables stemming
	// unless the user explicitly said otherwise. Rationale — nop-
	// mode's dominant audience is prose-heavy Obsidian vaults where
	// Porter2 English stemming consistently improves recall. The
	// ONNX path keeps its stem=off default since the embedder
	// already handles morphology. Fully overridable via --stem=false.
	if learnFlags.embedder == "none" && !cmd.Flags().Changed("stem") {
		cfg.Stem = true
	}

	// --embedder overrides --model / --config for the embedder choice.
	// "none" is the BM25-only mode: no ONNX download, no embedding.
	if learnFlags.embedder == "none" {
		cfg.ModelKey = engine.NopModelKey
	}

	info, err := engine.ResolveModel(cfg.ModelKey)
	if err != nil {
		return err
	}
	if cfg.Pooling != "" {
		info.Pooling = cfg.Pooling
	}
	if cfg.Normalize != nil {
		info.Normalize = *cfg.Normalize
	}

	if err := engine.EnsureSetup(info.Key); err != nil {
		return fmt.Errorf("setup: %w", err)
	}
	// Skip ONNX runtime init entirely for the nop embedder — no
	// library load, no process cost beyond argument parsing.
	if info.Key != engine.NopModelKey {
		engine.InitORT()
		defer ort.DestroyEnvironment()
	}
	emb, err := engine.LoadEmbedder(info)
	if err != nil {
		return err
	}
	defer emb.Close()

	chunks := engine.ParseKnowledge(learnFlags.from, cfg)
	fmt.Printf("Parsed %d chunks from %s (strategy=%s include=%v)\n",
		len(chunks), learnFlags.from, cfg.ChunkStrategy, cfg.Include)
	if len(chunks) == 0 {
		return fmt.Errorf("no chunks produced — check --from path and --include globs")
	}

	// IVF decision: --no-ivf > --use-ivf > auto-threshold.
	// IVF requires vectors, so force off when the embedder is nop.
	switch {
	case info.Key == engine.NopModelKey:
		cfg.UseIVF = false
	case learnFlags.noIVF:
		cfg.UseIVF = false
	case learnFlags.useIVF:
		cfg.UseIVF = true
	case len(chunks) >= learnFlags.autoIVFThreshold:
		cfg.UseIVF = true
		fmt.Printf("IVF: auto-enabled (%d chunks ≥ threshold %d); pass --no-ivf to disable\n",
			len(chunks), learnFlags.autoIVFThreshold)
	}
	if learnFlags.ivfCentroids > 0 {
		cfg.IVFCentroids = learnFlags.ivfCentroids
	}
	if learnFlags.ivfIters > 0 {
		cfg.IVFKmeansIt = learnFlags.ivfIters
	}
	if learnFlags.ivfNprobe > 0 {
		cfg.IVFNprobe = learnFlags.ivfNprobe
	}

	idx := engine.BuildIndex(chunks, emb, cfg)
	fmt.Printf("BM25: %d unique terms, avg doc len %.0f\n", len(idx.DocFreq), idx.AvgDocLen)

	if cfg.UseIVF {
		ivfDir := engine.IVFSiblingPath(learnFlags.output)
		fmt.Printf("Building IVF index at %s\n", ivfDir)
		ivfix, err := engine.BuildIVFFromIndex(idx, ivfDir, cfg)
		if err != nil {
			return fmt.Errorf("ivf build: %w", err)
		}
		fmt.Printf("  IVF: K=%d nprobe=%d ntotal=%d\n",
			ivfix.K, ivfix.Nprobe, ivfix.Ntotal())
		// Vectors live in the IVF now; strip them from the JSON.
		engine.StripChunkVectors(idx)
	}

	if err := engine.SaveIndex(idx, learnFlags.output); err != nil {
		return fmt.Errorf("write index: %w", err)
	}
	fi, _ := os.Stat(learnFlags.output)
	fmt.Printf("Index written to %s (%d chunks, %d bytes, model=%s dim=%d)\n",
		learnFlags.output, len(idx.Chunks), fi.Size(), info.Key, info.Dim)
	return nil
}

func init() {
	f := learnCmd.Flags()
	f.StringVar(&learnFlags.from, "from", "", "directory of source documents (auto-located if empty)")
	f.StringVar(&learnFlags.from, "knowledge", "", "alias for --from")
	_ = f.MarkHidden("knowledge")
	f.StringVarP(&learnFlags.output, "output", "o", "", "output index.json path (default: ./.brief/index.json)")
	f.StringVar(&learnFlags.configPath, "config", "", "IndexConfig JSON; CLI flags override its values")
	f.StringVar(&learnFlags.model, "model", "", "embedding model key (see `brief models`)")
	f.StringVar(&learnFlags.embedder, "embedder", "onnx", "embedder backend: onnx | none (none = BM25-only, no ONNX download)")

	f.StringVar(&learnFlags.chunkStrategy, "chunk-strategy", "", "heading | size")
	f.IntVar(&learnFlags.chunkSize, "chunk-size", 0, "target chars per chunk (size strategy)")
	f.IntVar(&learnFlags.chunkOverlap, "chunk-overlap", -1, "char overlap between adjacent chunks")
	f.IntVar(&learnFlags.minChunk, "min-chunk-chars", -1, "drop chunks smaller than this")
	f.IntVar(&learnFlags.maxChunk, "max-chunk-chars", -1, "truncate chunk body at this length (0=off)")
	f.IntVar(&learnFlags.embedMax, "embed-max-chars", 0, "truncate chunk text before tokenization")
	f.StringVar(&learnFlags.include, "include", "", "comma-separated file globs (e.g. '*.md,*.txt')")
	f.StringVar(&learnFlags.exclude, "exclude", "", "comma-separated file globs to exclude")
	f.StringVar(&learnFlags.pooling, "pooling", "", "override pooling: mean | cls")
	f.BoolVar(&learnFlags.stem, "stem", false, "stem BM25 tokens with Porter2 English (boosts recall on inflected queries)")
	f.BoolVar(&learnFlags.noFrontmatter, "no-frontmatter", false, "skip YAML frontmatter parsing (title/aliases/tags)")

	f.BoolVar(&learnFlags.useIVF, "use-ivf", false, "force IVF-Flat ANN companion")
	f.BoolVar(&learnFlags.noIVF, "no-ivf", false, "disable IVF even when auto-threshold would enable it")
	f.IntVar(&learnFlags.autoIVFThreshold, "auto-ivf-threshold", 5000, "auto-enable IVF when chunk count ≥ this")
	f.IntVar(&learnFlags.ivfCentroids, "ivf-centroids", 0, "IVF K (0 = auto 4·√N)")
	f.IntVar(&learnFlags.ivfIters, "ivf-kmeans-iters", 0, "IVF k-means iterations (0 = 20)")
	f.IntVar(&learnFlags.ivfNprobe, "ivf-nprobe", 0, "default nprobe stored in the IVF manifest (0 = auto √K)")

	rootCmd.AddCommand(learnCmd)
}

func splitCSV(s string) []string {
	parts := strings.Split(s, ",")
	out := parts[:0]
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

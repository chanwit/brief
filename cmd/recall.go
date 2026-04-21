// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

package cmd

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/spf13/cobra"
	ort "github.com/yalue/onnxruntime_go"

	"github.com/chanwit/brief/internal/engine"
)

var recallFlags struct {
	indexPath  string
	configPath string
	mode       string
	k          int

	ws   float64
	wb   float64
	k1   float64
	b    float64
	shf  float64
	ssf  float64
	bmsz float64
	mqt  int

	nprobe int
	nsem   int

	maxLinked int
	noLinks   bool

	titleBoost float64
	tags       []string

	jsonOut bool
}

var recallCmd = &cobra.Command{
	Use:   "recall \"question text\"",
	Short: "Recall relevant passages for a question",
	Long: `Runs a query against an index and prints the top-K ranked chunks.

With no flags, recall looks for an index at $BRIEF_INDEX, then
./.brief/index.json, then ./.claude/knowledge.index.json, then
./brief.index.json.

Output is TTY-aware: a terminal gets a human-readable banner and
ranked list; a pipe gets a <knowledge>...</knowledge> Markdown block
designed for a UserPromptSubmit hook. --json forces machine-readable
JSON regardless.`,
	Args: cobra.MinimumNArgs(1),
	RunE: runRecall,
}

func runRecall(cmd *cobra.Command, args []string) error {
	query := strings.Join(args, " ")

	if recallFlags.indexPath == "" {
		recallFlags.indexPath = locateIndex()
	}
	if recallFlags.indexPath == "" {
		errNotFound("index", "BRIEF_INDEX", indexLookupOrder) // exits
	}

	idx, err := engine.LoadIndex(recallFlags.indexPath)
	if err != nil {
		return err
	}
	defer idx.Close()

	cfg := engine.DefaultQueryConfig()
	if recallFlags.configPath != "" {
		c, err := engine.LoadQueryConfig(recallFlags.configPath)
		if err != nil {
			return err
		}
		cfg = c
	}
	if cmd.Flags().Changed("mode") {
		cfg.Mode = recallFlags.mode
	}
	if cmd.Flags().Changed("k") {
		cfg.K = recallFlags.k
	}
	setFloatIfChanged(cmd, "weight-semantic", &cfg.WeightSemantic, recallFlags.ws)
	setFloatIfChanged(cmd, "weight-bm25", &cfg.WeightBM25, recallFlags.wb)
	setFloatIfChanged(cmd, "bm25-k1", &cfg.BM25K1, recallFlags.k1)
	setFloatIfChanged(cmd, "bm25-b", &cfg.BM25B, recallFlags.b)
	setFloatIfChanged(cmd, "semantic-hard-floor", &cfg.SemanticHardFloor, recallFlags.shf)
	setFloatIfChanged(cmd, "semantic-soft-floor", &cfg.SemanticSoftFloor, recallFlags.ssf)
	setFloatIfChanged(cmd, "bm25-min-for-soft-zone", &cfg.BM25MinForSoftZone, recallFlags.bmsz)
	if cmd.Flags().Changed("min-query-terms") {
		cfg.MinQueryTerms = recallFlags.mqt
	}
	if recallFlags.nprobe > 0 {
		cfg.Nprobe = recallFlags.nprobe
	}
	if recallFlags.nsem > 0 {
		cfg.NSemantic = recallFlags.nsem
	}
	if cmd.Flags().Changed("max-linked") {
		cfg.MaxLinked = recallFlags.maxLinked
	}
	if recallFlags.noLinks {
		cfg.MaxLinked = 0
	}
	if cmd.Flags().Changed("title-boost") {
		cfg.TitleBoost = recallFlags.titleBoost
	}
	if len(recallFlags.tags) > 0 {
		cfg.RequireTags = recallFlags.tags
	}

	// An index built with --embedder none has no vectors. Downgrade
	// semantic/hybrid to BM25 silently (and skip all ONNX init) so
	// recall still returns useful results. If the user explicitly
	// asked for semantic, surface an error instead.
	indexHasVectors := idx.ModelInfo.Key != engine.NopModelKey && idx.ModelInfo.Dim > 0
	if !indexHasVectors {
		if cfg.Mode == "semantic" {
			return fmt.Errorf("index has no embeddings (built with --embedder none); semantic mode is unavailable. Use --mode bm25, or rebuild with an ONNX embedder.")
		}
		cfg.Mode = "bm25"
	}

	// All modes go through DispatchSearch so wikilink expansion runs
	// uniformly. BM25 mode doesn't need an embedder (no qVec required);
	// semantic and hybrid do, so we only load ONNX in those cases.
	var qVec []float32
	if cfg.Mode != "bm25" {
		if err := engine.EnsureSetup(idx.ModelInfo.Key); err != nil {
			return fmt.Errorf("setup: %w", err)
		}
		engine.InitORT()
		defer ort.DestroyEnvironment()
		emb, err := engine.LoadEmbedder(idx.ModelInfo)
		if err != nil {
			return err
		}
		defer emb.Close()
		qVec = emb.Embed(query)
	}
	results := engine.DispatchSearch(idx, qVec, query, cfg)

	switch {
	case recallFlags.jsonOut:
		data, _ := json.MarshalIndent(results, "", "  ")
		fmt.Println(string(data))
	case isStdoutTTY():
		printHumanOutput(idx, query, cfg, results)
	default:
		printHookOutput(results)
	}
	return nil
}

func setFloatIfChanged(cmd *cobra.Command, name string, dst *float64, v float64) {
	if cmd.Flags().Changed(name) && !math.IsNaN(v) {
		*dst = v
	}
}

func init() {
	f := recallCmd.Flags()
	f.StringVar(&recallFlags.indexPath, "index", "", "path to index.json (auto-located if empty)")
	f.StringVar(&recallFlags.configPath, "config", "", "QueryConfig JSON; CLI flags override its values")
	f.StringVar(&recallFlags.mode, "mode", "hybrid", "hybrid | bm25 | semantic")
	f.IntVarP(&recallFlags.k, "k", "k", 3, "top-k results")

	f.Float64Var(&recallFlags.ws, "weight-semantic", math.NaN(), "hybrid semantic weight")
	f.Float64Var(&recallFlags.wb, "weight-bm25", math.NaN(), "hybrid BM25 weight")
	f.Float64Var(&recallFlags.k1, "bm25-k1", math.NaN(), "BM25 k1")
	f.Float64Var(&recallFlags.b, "bm25-b", math.NaN(), "BM25 b")
	f.Float64Var(&recallFlags.shf, "semantic-hard-floor", math.NaN(), "reject hits below this cosine")
	f.Float64Var(&recallFlags.ssf, "semantic-soft-floor", math.NaN(), "keep if BM25 also agrees above this")
	f.Float64Var(&recallFlags.bmsz, "bm25-min-for-soft-zone", math.NaN(), "BM25 norm floor inside soft zone")
	f.IntVar(&recallFlags.mqt, "min-query-terms", 0, "min query terms that must hit corpus (0=auto)")

	f.IntVar(&recallFlags.nprobe, "nprobe", 0, "IVF probe count at query time (0 = index default)")
	f.IntVar(&recallFlags.nsem, "n-semantic", 0, "IVF shortlist size fed into hybrid re-ranking (0 = auto)")

	f.IntVar(&recallFlags.maxLinked, "max-linked", 3, "max wikilink-expanded chunks to add to results (0 = disable)")
	f.BoolVar(&recallFlags.noLinks, "no-links", false, "disable wikilink expansion (shorthand for --max-linked=0)")

	f.Float64Var(&recallFlags.titleBoost, "title-boost", 2.5, "BM25F title-field multiplier (1 = vanilla BM25)")
	f.StringSliceVar(&recallFlags.tags, "tag", nil, "filter to chunks with one of these tags (repeatable; ORed)")

	f.BoolVar(&recallFlags.jsonOut, "json", false, "emit JSON results to stdout")

	rootCmd.AddCommand(recallCmd)
}

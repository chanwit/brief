// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

// rag-engine is a dataset-agnostic Go-native RAG index/query tool backed by a
// local ONNX sentence-transformer model and BM25.
//
// Usage:
//
//	rag-engine setup        [--model KEY]
//	rag-engine models
//	rag-engine index        --knowledge DIR --output PATH [--model KEY] [--config JSON] [chunking/globs/pooling flags]
//	rag-engine query        --index PATH [--config JSON] [--mode MODE] [-k N] [knob flags] "query text"
//	rag-engine tune-query   --index PATH --eval JSON [--mode MODE] [-k N] [--trials N] [--output PATH]
//	rag-engine tune-index   --knowledge DIR --eval JSON [--model KEY] [--trials N] [--query-config JSON] [--output-config PATH] [--output-index PATH]
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"strings"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/chanwit/rag-engine/ivf"
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}
	cmd, args := os.Args[1], os.Args[2:]
	switch cmd {
	case "setup":
		cmdSetup(args)
	case "models":
		cmdModels(args)
	case "index":
		cmdIndex(args)
	case "query":
		cmdQuery(args)
	case "tune-query":
		cmdTuneQuery(args)
	case "tune-index":
		cmdTuneIndex(args)
	case "help", "-h", "--help":
		printUsage()
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n\n", cmd)
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Fprint(os.Stderr, `rag-engine — Go-native RAG index & query

Commands:
  setup         download ONNX runtime + requested embedding model
  models        list built-in embedding models
  index         build an index from a knowledge directory
  query         run a query against an index
  tune-query    random-search query-time knobs against an eval set (MRR@k)
  tune-index    grid-search index-time knobs (chunking) against an eval set

Environment:
  RAG_HOME         root dir (default: ~/.rag-engine)
  ORT_LIB_PATH     full path to libonnxruntime shared library
  RAG_MODELS_DIR   where per-model subdirs live

Run any command with -h for its flags.
`)
}

// ---------- setup ----------

func cmdSetup(args []string) {
	fs := flag.NewFlagSet("setup", flag.ExitOnError)
	model := fs.String("model", defaultModelKey, "model key to download")
	fs.Parse(args)

	if err := ensureSetup(*model); err != nil {
		fmt.Fprintf(os.Stderr, "setup failed: %v\n", err)
		os.Exit(1)
	}
	info, _ := resolveModel(*model)
	fmt.Printf("ONNX runtime: %s\n", ortLibPath)
	fmt.Printf("Model:        %s\n", modelDirFor(info.Key))
	fmt.Println("Setup complete.")
}

// ---------- models ----------

func cmdModels(args []string) {
	fs := flag.NewFlagSet("models", flag.ExitOnError)
	fs.Parse(args)

	fmt.Println("Built-in ONNX embedding models:")
	for _, k := range modelKeys() {
		m := knownModels[k]
		marker := " "
		if k == defaultModelKey {
			marker = "*"
		}
		fmt.Printf("  %s %-30s dim=%-4d pool=%-4s max_len=%-4d repo=%s\n",
			marker, k, m.Dim, m.Pooling, m.MaxLength, m.HFRepo)
	}
	fmt.Printf("\n(%s marks the default)\n", "*")
}

// ---------- index ----------

func cmdIndex(args []string) {
	fs := flag.NewFlagSet("index", flag.ExitOnError)
	knowledge := fs.String("knowledge", "", "directory of source documents")
	output := fs.String("output", "", "output index.json path")
	configPath := fs.String("config", "", "IndexConfig JSON; CLI flags override its values")
	model := fs.String("model", "", "embedding model key (see `rag-engine models`)")
	chunkStrategy := fs.String("chunk-strategy", "", "heading | size")
	chunkSize := fs.Int("chunk-size", 0, "target chars per chunk (size strategy)")
	chunkOverlap := fs.Int("chunk-overlap", -1, "char overlap between adjacent chunks")
	minChunk := fs.Int("min-chunk-chars", -1, "drop chunks smaller than this")
	maxChunk := fs.Int("max-chunk-chars", -1, "truncate chunk body at this length (0=off)")
	embedMax := fs.Int("embed-max-chars", 0, "truncate chunk text before tokenization")
	include := fs.String("include", "", "comma-separated file globs (e.g. '*.md,*.txt')")
	exclude := fs.String("exclude", "", "comma-separated file globs to exclude")
	pooling := fs.String("pooling", "", "override pooling: mean | cls")
	useIVF := fs.Bool("use-ivf", false, "store vectors in a sibling IVF-Flat index for fast ANN search")
	ivfCentroids := fs.Int("ivf-centroids", 0, "IVF K (number of centroids); 0 = auto 4·√N")
	ivfIters := fs.Int("ivf-kmeans-iters", 0, "IVF k-means iterations (0 = 20)")
	ivfNprobe := fs.Int("ivf-nprobe", 0, "default nprobe stored in the IVF manifest (0 = auto √K)")
	fs.Parse(args)

	if *knowledge == "" || *output == "" {
		fs.Usage()
		fmt.Fprintln(os.Stderr, "error: --knowledge and --output required")
		os.Exit(1)
	}

	cfg := defaultIndexConfig()
	if *configPath != "" {
		c, err := loadIndexConfig(*configPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			os.Exit(1)
		}
		cfg = c
	}
	if *model != "" {
		cfg.ModelKey = *model
	}
	if *chunkStrategy != "" {
		cfg.ChunkStrategy = *chunkStrategy
	}
	if *chunkSize > 0 {
		cfg.ChunkSize = *chunkSize
	}
	if *chunkOverlap >= 0 {
		cfg.ChunkOverlap = *chunkOverlap
	}
	if *minChunk >= 0 {
		cfg.MinChunkChars = *minChunk
	}
	if *maxChunk >= 0 {
		cfg.MaxChunkChars = *maxChunk
	}
	if *embedMax > 0 {
		cfg.EmbedMaxChars = *embedMax
	}
	if *include != "" {
		cfg.Include = splitCSV(*include)
	}
	if *exclude != "" {
		cfg.Exclude = splitCSV(*exclude)
	}
	if *pooling != "" {
		cfg.Pooling = *pooling
	}
	if *useIVF {
		cfg.UseIVF = true
	}
	if *ivfCentroids > 0 {
		cfg.IVFCentroids = *ivfCentroids
	}
	if *ivfIters > 0 {
		cfg.IVFKmeansIt = *ivfIters
	}
	if *ivfNprobe > 0 {
		cfg.IVFNprobe = *ivfNprobe
	}

	info, err := resolveModel(cfg.ModelKey)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
	if cfg.Pooling != "" {
		info.Pooling = cfg.Pooling
	}
	if cfg.Normalize != nil {
		info.Normalize = *cfg.Normalize
	}

	if err := ensureSetup(info.Key); err != nil {
		fmt.Fprintf(os.Stderr, "setup: %v\n", err)
		os.Exit(1)
	}
	initORT()
	defer ort.DestroyEnvironment()
	emb, err := loadEmbedder(info)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
	defer emb.Close()

	chunks := parseKnowledge(*knowledge, cfg)
	fmt.Printf("Parsed %d chunks from %s (strategy=%s include=%v)\n",
		len(chunks), *knowledge, cfg.ChunkStrategy, cfg.Include)
	if len(chunks) == 0 {
		fmt.Fprintln(os.Stderr, "error: no chunks produced — check --knowledge path and --include globs")
		os.Exit(1)
	}

	idx := buildIndex(chunks, emb, cfg)
	fmt.Printf("BM25: %d unique terms, avg doc len %.0f\n", len(idx.DocFreq), idx.AvgDocLen)

	if cfg.UseIVF {
		ivfDir := ivfSiblingPath(*output)
		fmt.Printf("Building IVF index at %s\n", ivfDir)
		ivfix, err := buildIVFFromIndex(idx, ivfDir, cfg)
		if err != nil {
			fmt.Fprintf(os.Stderr, "ivf build: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("  IVF: K=%d nprobe=%d ntotal=%d\n",
			ivfix.K, ivfix.Nprobe, ivfix.Ntotal())
		// Vectors are now in the IVF — strip them from the JSON so the
		// on-disk size drops from (dim*4*N + text) to just text + BM25.
		stripChunkVectors(idx)
	}

	if err := saveIndex(idx, *output); err != nil {
		fmt.Fprintf(os.Stderr, "error writing index: %v\n", err)
		os.Exit(1)
	}
	fi, _ := os.Stat(*output)
	fmt.Printf("Index written to %s (%d chunks, %d bytes, model=%s dim=%d)\n",
		*output, len(idx.Chunks), fi.Size(), info.Key, info.Dim)
}

// ---------- query ----------

func cmdQuery(args []string) {
	fs := flag.NewFlagSet("query", flag.ExitOnError)
	indexPath := fs.String("index", "", "path to index.json")
	configPath := fs.String("config", "", "QueryConfig JSON; CLI flags override its values")
	mode := fs.String("mode", "", "hybrid | bm25 | semantic")
	k := fs.Int("k", 0, "top-k results")
	ws := fs.Float64("weight-semantic", math.NaN(), "hybrid semantic weight")
	wb := fs.Float64("weight-bm25", math.NaN(), "hybrid BM25 weight")
	k1 := fs.Float64("bm25-k1", math.NaN(), "BM25 k1 (default 1.2)")
	bb := fs.Float64("bm25-b", math.NaN(), "BM25 b  (default 0.75)")
	shf := fs.Float64("semantic-hard-floor", math.NaN(), "reject hits below this cosine")
	ssf := fs.Float64("semantic-soft-floor", math.NaN(), "keep if BM25 also agrees above this")
	bmsz := fs.Float64("bm25-min-for-soft-zone", math.NaN(), "BM25 norm floor inside soft zone")
	mqt := fs.Int("min-query-terms", -1, "min query terms that must hit corpus (0=auto)")
	nprobe := fs.Int("nprobe", 0, "IVF probe count at query time (0 = index default)")
	nsem := fs.Int("n-semantic", 0, "IVF shortlist size fed into hybrid re-ranking (0 = auto)")
	jsonOut := fs.Bool("json", false, "emit JSON results to stdout")
	fs.Parse(args)

	query := strings.Join(fs.Args(), " ")
	if *indexPath == "" || query == "" {
		fs.Usage()
		fmt.Fprintln(os.Stderr, "error: --index and query string required")
		os.Exit(1)
	}

	idx, err := loadIndex(*indexPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}

	cfg := defaultQueryConfig()
	if *configPath != "" {
		c, err := loadQueryConfig(*configPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			os.Exit(1)
		}
		cfg = c
	}
	if *mode != "" {
		cfg.Mode = *mode
	}
	if *k > 0 {
		cfg.K = *k
	}
	setFloat(&cfg.WeightSemantic, *ws)
	setFloat(&cfg.WeightBM25, *wb)
	setFloat(&cfg.BM25K1, *k1)
	setFloat(&cfg.BM25B, *bb)
	setFloat(&cfg.SemanticHardFloor, *shf)
	setFloat(&cfg.SemanticSoftFloor, *ssf)
	setFloat(&cfg.BM25MinForSoftZone, *bmsz)
	if *mqt >= 0 {
		cfg.MinQueryTerms = *mqt
	}
	if *nprobe > 0 {
		cfg.Nprobe = *nprobe
	}
	if *nsem > 0 {
		cfg.NSemantic = *nsem
	}
	defer idx.Close()

	var results []SearchResult
	if cfg.Mode == "bm25" {
		results = searchBM25(idx, query, cfg)
	} else {
		if err := ensureSetup(idx.ModelInfo.Key); err != nil {
			fmt.Fprintf(os.Stderr, "setup: %v\n", err)
			os.Exit(1)
		}
		initORT()
		defer ort.DestroyEnvironment()
		emb, err := loadEmbedder(idx.ModelInfo)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			os.Exit(1)
		}
		defer emb.Close()
		qVec := emb.Embed(query)
		results = dispatchSearch(idx, qVec, query, cfg)
	}

	if *jsonOut {
		data, _ := json.MarshalIndent(results, "", "  ")
		fmt.Println(string(data))
		return
	}
	backend := "flat"
	if idx.ivfix != nil {
		backend = fmt.Sprintf("ivf(K=%d,nprobe=%d)", idx.ivfix.K, effectiveNprobe(idx.ivfix, cfg))
	}
	fmt.Printf("%s results for: %s (model=%s backend=%s)\n\n",
		cfg.Mode, query, idx.ModelInfo.Key, backend)
	for i, r := range results {
		switch cfg.Mode {
		case "hybrid":
			fmt.Printf("%d. [%.4f sem=%.3f bm25=%.3f] %s — %s\n",
				i+1, r.Score, r.Semantic, r.BM25, r.File, r.Title)
		default:
			fmt.Printf("%d. [%.4f] %s — %s\n", i+1, r.Score, r.File, r.Title)
		}
	}
}

// ---------- tune-query ----------

func cmdTuneQuery(args []string) {
	fs := flag.NewFlagSet("tune-query", flag.ExitOnError)
	indexPath := fs.String("index", "", "path to index.json")
	evalPath := fs.String("eval", "", "path to eval JSON (see README for schema)")
	mode := fs.String("mode", "hybrid", "mode to tune: hybrid | bm25 | semantic")
	trials := fs.Int("trials", 200, "number of random-search trials")
	k := fs.Int("k", 5, "top-k used to compute the objective")
	objective := fs.String("objective", "mrr", "objective: mrr | hit_rate (hit_rate targets 100% in-top-K)")
	output := fs.String("output", "", "save best QueryConfig JSON here")
	fs.Parse(args)

	if *indexPath == "" || *evalPath == "" {
		fs.Usage()
		fmt.Fprintln(os.Stderr, "error: --index and --eval required")
		os.Exit(1)
	}

	idx, err := loadIndex(*indexPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
	set, err := loadEvalSet(*evalPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}

	if err := ensureSetup(idx.ModelInfo.Key); err != nil {
		fmt.Fprintf(os.Stderr, "setup: %v\n", err)
		os.Exit(1)
	}
	initORT()
	defer ort.DestroyEnvironment()
	emb, err := loadEmbedder(idx.ModelInfo)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
	defer emb.Close()

	best, metrics := tuneQueryConfig(idx, emb, set, *trials, *mode, *k, *objective)
	fmt.Printf("\nBest hit@%d=%.4f MRR@%d=%.4f recall@%d=%.4f precision@%d=%.4f\n",
		*k, metrics.HitRate, *k, metrics.MRR, *k, metrics.Recall, *k, metrics.Precision)
	data, _ := json.MarshalIndent(best, "", "  ")
	fmt.Println(string(data))
	if *output != "" {
		if err := saveQueryConfig(best, *output); err != nil {
			fmt.Fprintf(os.Stderr, "save: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Saved best QueryConfig to %s\n", *output)
	}
}

// ---------- tune-index ----------

func cmdTuneIndex(args []string) {
	fs := flag.NewFlagSet("tune-index", flag.ExitOnError)
	knowledgeDir := fs.String("knowledge", "", "directory of source documents")
	evalPath := fs.String("eval", "", "path to eval JSON")
	model := fs.String("model", defaultModelKey, "embedding model key")
	trials := fs.Int("trials", 8, "number of chunking configs to try")
	k := fs.Int("k", 5, "top-k used to compute the objective")
	objective := fs.String("objective", "mrr", "objective: mrr | hit_rate")
	queryConfigPath := fs.String("query-config", "", "QueryConfig JSON to use when scoring candidates")
	outConfig := fs.String("output-config", "", "save best IndexConfig here")
	outIndex := fs.String("output-index", "", "save best index here")
	fs.Parse(args)

	if *knowledgeDir == "" || *evalPath == "" {
		fs.Usage()
		fmt.Fprintln(os.Stderr, "error: --knowledge and --eval required")
		os.Exit(1)
	}

	set, err := loadEvalSet(*evalPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}

	info, err := resolveModel(*model)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
	if err := ensureSetup(info.Key); err != nil {
		fmt.Fprintf(os.Stderr, "setup: %v\n", err)
		os.Exit(1)
	}
	initORT()
	defer ort.DestroyEnvironment()
	emb, err := loadEmbedder(info)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
	defer emb.Close()

	qCfg := defaultQueryConfig()
	if *queryConfigPath != "" {
		c, err := loadQueryConfig(*queryConfigPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			os.Exit(1)
		}
		qCfg = c
	}
	qCfg.K = *k

	bestCfg, bestIdx, metrics := tuneIndexConfig(*knowledgeDir, emb, set, *trials, qCfg, *objective)
	fmt.Printf("\nBest hit@%d=%.4f MRR@%d=%.4f recall@%d=%.4f\n",
		*k, metrics.HitRate, *k, metrics.MRR, *k, metrics.Recall)
	data, _ := json.MarshalIndent(bestCfg, "", "  ")
	fmt.Println(string(data))
	if *outConfig != "" {
		if err := saveIndexConfig(bestCfg, *outConfig); err != nil {
			fmt.Fprintf(os.Stderr, "save: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Saved best IndexConfig to %s\n", *outConfig)
	}
	if *outIndex != "" {
		if err := saveIndex(bestIdx, *outIndex); err != nil {
			fmt.Fprintf(os.Stderr, "save: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Saved best index to %s\n", *outIndex)
	}
}

// ---------- helpers ----------

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

// setFloat overwrites dst only if v is not NaN — lets us use NaN as the
// flag.Float64 "unset" sentinel.
func setFloat(dst *float64, v float64) {
	if !math.IsNaN(v) {
		*dst = v
	}
}

// effectiveNprobe is just for diagnostics in the query banner.
func effectiveNprobe(ix *ivf.IVFFlat, cfg QueryConfig) int {
	if cfg.Nprobe > 0 {
		return cfg.Nprobe
	}
	return ix.Nprobe
}

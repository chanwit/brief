// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

package engine

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	ort "github.com/yalue/onnxruntime_go"
)

// TestMain runs auto-setup for the default model once and keeps one ONNX
// environment alive for the whole test binary.
func TestMain(m *testing.M) {
	if err := EnsureSetup(DefaultModelKey); err != nil {
		fmt.Fprintf(os.Stderr, "auto-setup failed: %v\n", err)
		os.Exit(1)
	}
	if err := InitORTSafe(); err != nil {
		fmt.Fprintf(os.Stderr, "init ORT: %v\n", err)
		os.Exit(1)
	}
	code := m.Run()
	ort.DestroyEnvironment()
	os.Exit(code)
}

func TestAutoSetupArtifacts(t *testing.T) {
	info, err := ResolveModel(DefaultModelKey)
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range []string{
		OrtLibPath,
		filepath.Join(ModelDirFor(info.Key), "model.onnx"),
		filepath.Join(ModelDirFor(info.Key), "tokenizer.json"),
	} {
		st, err := os.Stat(p)
		if err != nil {
			t.Fatalf("expected %s after setup: %v", p, err)
		}
		if st.Size() == 0 {
			t.Fatalf("%s is empty", p)
		}
	}
}

func TestModelRegistryKnowsBuiltins(t *testing.T) {
	// Every built-in model must be fully specified.
	for _, k := range ModelKeys() {
		info := KnownModels[k]
		if info.HFRepo == "" || info.Dim == 0 || info.MaxLength == 0 ||
			info.Pooling == "" || len(info.Inputs) == 0 || len(info.Outputs) == 0 {
			t.Errorf("model %q is under-specified: %+v", k, info)
		}
	}
	if _, err := ResolveModel("does-not-exist"); err == nil {
		t.Error("ResolveModel should reject unknown keys")
	}
}

func TestUnknownModelKeyRejected(t *testing.T) {
	if _, err := ResolveModel("llama3-gguf"); err == nil {
		t.Fatal("expected unknown model to be rejected")
	}
}

func TestBuildIndexAndSearch(t *testing.T) {
	tmp := t.TempDir()
	kdir := filepath.Join(tmp, "knowledge")
	if err := os.Mkdir(kdir, 0o755); err != nil {
		t.Fatal(err)
	}

	writeFile(t, filepath.Join(kdir, "apples.md"), applesMD)
	writeFile(t, filepath.Join(kdir, "cars.md"), carsMD)

	info, _ := ResolveModel(DefaultModelKey)
	emb, err := LoadEmbedder(info)
	if err != nil {
		t.Fatal(err)
	}
	defer emb.Close()

	cfg := DefaultIndexConfig()
	chunks := ParseKnowledge(kdir, cfg)
	if len(chunks) < 4 {
		t.Fatalf("expected ≥4 chunks from 2 files, got %d", len(chunks))
	}

	idx := BuildIndex(chunks, emb, cfg)
	if idx.Schema != IndexSchemaVersion {
		t.Fatalf("schema = %d, want %d", idx.Schema, IndexSchemaVersion)
	}
	if idx.ModelInfo.Key != info.Key {
		t.Fatalf("index ModelInfo = %q, want %q", idx.ModelInfo.Key, info.Key)
	}
	if got := len(idx.Chunks[0].Vector); got != info.Dim {
		t.Fatalf("vector dim = %d, want %d", got, info.Dim)
	}

	// Round-trip through disk.
	indexPath := filepath.Join(tmp, "index.json")
	if err := SaveIndex(idx, indexPath); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadIndex(indexPath)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.ModelInfo.Key != info.Key {
		t.Fatalf("round-trip model mismatch: %q", loaded.ModelInfo.Key)
	}

	qcfg := DefaultQueryConfig()

	// BM25: distinctive term.
	if bm := SearchBM25(loaded, "oil", qcfg); len(bm) == 0 || !strings.Contains(bm[0].File, "cars") {
		t.Fatalf(`BM25("oil") top=%v, want cars.md`, bm)
	}

	// Semantic: paraphrase with no keyword overlap.
	qv := emb.Embed("how do I raise fruit trees in an orchard")
	if sem := SearchSemantic(loaded, qv, 3); len(sem) == 0 || !strings.Contains(sem[0].File, "apples") {
		t.Fatalf("semantic paraphrase top=%v, want apples.md", sem)
	}

	// Hybrid: combines both.
	qv2 := emb.Embed("how often to change the oil")
	hyb := SearchHybrid(loaded, qv2, "how often to change the oil", qcfg)
	if len(hyb) == 0 || !strings.Contains(hyb[0].File, "cars") {
		t.Fatalf("hybrid top=%v, want cars.md", hyb)
	}

	// Relevance gate.
	offQ := "xylophone manufacturing quarterly report"
	offV := emb.Embed(offQ)
	if off := SearchHybrid(loaded, offV, offQ, qcfg); len(off) != 0 {
		t.Fatalf("off-topic query returned %d hits, want 0", len(off))
	}
}

func TestSizeBasedChunking(t *testing.T) {
	tmp := t.TempDir()
	kdir := filepath.Join(tmp, "k")
	os.Mkdir(kdir, 0o755)
	// A flat prose document with no headings — heading strategy would produce one chunk.
	writeFile(t, filepath.Join(kdir, "prose.txt"),
		strings.Repeat("The quick brown fox jumps over the lazy dog. ", 80))

	cfg := DefaultIndexConfig()
	cfg.ChunkStrategy = "size"
	cfg.ChunkSize = 300
	cfg.ChunkOverlap = 50
	cfg.Include = []string{"*.txt"}

	chunks := ParseKnowledge(kdir, cfg)
	if len(chunks) < 3 {
		t.Fatalf("size chunking produced %d chunks, expected ≥3", len(chunks))
	}
	for _, c := range chunks {
		if len(c.Body) > cfg.ChunkSize+20 { // slack for trimming
			t.Errorf("chunk body %d chars exceeds chunk-size %d", len(c.Body), cfg.ChunkSize)
		}
	}
}

func TestModelMismatchDetection(t *testing.T) {
	// An index built with one model must be distinguishable from another, and
	// the query path must be able to pick the right embedder from the index.
	info, _ := ResolveModel(DefaultModelKey)
	emb, err := LoadEmbedder(info)
	if err != nil {
		t.Fatal(err)
	}
	defer emb.Close()

	tmp := t.TempDir()
	kdir := filepath.Join(tmp, "k")
	os.Mkdir(kdir, 0o755)
	writeFile(t, filepath.Join(kdir, "a.md"), applesMD)

	cfg := DefaultIndexConfig()
	idx := BuildIndex(ParseKnowledge(kdir, cfg), emb, cfg)
	path := filepath.Join(tmp, "idx.json")
	if err := SaveIndex(idx, path); err != nil {
		t.Fatal(err)
	}

	loaded, err := LoadIndex(path)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.ModelInfo.Key != info.Key ||
		loaded.ModelInfo.Dim != info.Dim ||
		loaded.ModelInfo.Pooling != info.Pooling {
		t.Fatalf("loaded.ModelInfo mismatch: %+v vs %+v", loaded.ModelInfo, info)
	}

	// Tamper with the schema version to simulate a future index.
	loaded.Schema = IndexSchemaVersion + 1
	future := filepath.Join(tmp, "future.json")
	data, _ := json.Marshal(loaded)
	os.WriteFile(future, data, 0o644)
	if _, err := LoadIndex(future); err == nil {
		t.Fatal("expected LoadIndex to reject a newer schema")
	}
}

func TestAlternateModelSmoke(t *testing.T) {
	if testing.Short() {
		t.Skip("skip alternate-model download in short mode")
	}
	altKey := "all-MiniLM-L12-v2"
	if err := EnsureSetup(altKey); err != nil {
		t.Fatalf("EnsureSetup(%s): %v", altKey, err)
	}
	info, _ := ResolveModel(altKey)
	emb, err := LoadEmbedder(info)
	if err != nil {
		t.Fatal(err)
	}
	defer emb.Close()

	v := emb.Embed("hello world")
	if len(v) != info.Dim {
		t.Fatalf("alt model %s produced dim=%d, want %d", altKey, len(v), info.Dim)
	}
}

func TestTuneQueryImprovesOrHoldsBaseline(t *testing.T) {
	info, _ := ResolveModel(DefaultModelKey)
	emb, err := LoadEmbedder(info)
	if err != nil {
		t.Fatal(err)
	}
	defer emb.Close()

	tmp := t.TempDir()
	kdir := filepath.Join(tmp, "k")
	os.Mkdir(kdir, 0o755)
	writeFile(t, filepath.Join(kdir, "apples.md"), applesMD)
	writeFile(t, filepath.Join(kdir, "cars.md"), carsMD)

	cfg := DefaultIndexConfig()
	idx := BuildIndex(ParseKnowledge(kdir, cfg), emb, cfg)

	set := &EvalSet{Queries: []EvalQuery{
		{Query: "how often should I change engine oil", RelevantFiles: []string{"cars.md"}},
		{Query: "pruning orchard fruit trees", RelevantFiles: []string{"apples.md"}},
		{Query: "storing fruit in a cool cellar", RelevantFiles: []string{"apples.md"}},
		{Query: "driving motor vehicles on the road", RelevantFiles: []string{"cars.md"}},
	}}

	baseline := DefaultQueryConfig()
	baseline.Mode = "hybrid"
	baseline.K = 3
	preQ := PreEmbedQueries(emb, set)
	baseMetrics := EvaluateConfig(idx, preQ, baseline)

	best, bestMetrics := TuneQueryConfig(idx, emb, set, 40, "hybrid", 3, "mrr")
	if bestMetrics.MRR < baseMetrics.MRR-1e-9 {
		t.Fatalf("tuner regressed: best=%.4f baseline=%.4f", bestMetrics.MRR, baseMetrics.MRR)
	}
	// Sanity: the tuned config should still produce non-empty results.
	got := SearchHybrid(idx, emb.Embed(set.Queries[0].Query), set.Queries[0].Query, best)
	if len(got) == 0 {
		t.Fatalf("tuned config returned 0 results on a known-good query")
	}
}

// TestHybridGateAppliesBeforeTopK guards against a regression where the
// relevance gate was applied AFTER truncating to top-K. In that buggy order,
// a cluster of high-BM25 but low-semantic hits at the top of the ranking
// could evict all the eligible hits just below.
func TestHybridGateAppliesBeforeTopK(t *testing.T) {
	// Hand-build an index with unit-normalized vectors so we control scores
	// without invoking the ONNX model.
	queryVec := []float32{1, 0, 0, 0}

	// 3 "decoys": high BM25 (distinctive token "bananaword") but orthogonal
	// semantic vectors (cosine = 0, way below hard floor).
	decoyBody := "bananaword bananaword bananaword bananaword bananaword"
	// 2 "keepers": no special BM25 signal, strong semantic match.
	keeperBody := "ordinary ordinary words"

	mkChunk := func(file, body string, vec []float32) Chunk {
		terms := bm25Tokenize(body, false)
		tf := make(map[string]int)
		for _, tok := range terms {
			tf[tok]++
		}
		return Chunk{File: file, Title: file, Body: body, Vector: vec, TermFreq: tf, DocLen: len(terms)}
	}

	chunks := []Chunk{
		mkChunk("decoy1.md", decoyBody, []float32{0, 1, 0, 0}),
		mkChunk("decoy2.md", decoyBody, []float32{0, 0, 1, 0}),
		mkChunk("decoy3.md", decoyBody, []float32{0, 0, 0, 1}),
		mkChunk("keeper1.md", keeperBody, []float32{1, 0, 0, 0}),
		mkChunk("keeper2.md", keeperBody, []float32{0.95, 0.312, 0, 0}), // cos ≈ 0.95
	}
	// Recompute df from chunks.
	df := make(map[string]int)
	total := 0
	for _, c := range chunks {
		total += c.DocLen
		seen := make(map[string]bool)
		for tok := range c.TermFreq {
			if !seen[tok] {
				df[tok]++
				seen[tok] = true
			}
		}
	}
	idx := &Index{
		Schema:    IndexSchemaVersion,
		ModelInfo: KnownModels[DefaultModelKey],
		Chunks:    chunks, DocFreq: df, AvgDocLen: float64(total) / float64(len(chunks)),
	}

	cfg := DefaultQueryConfig()
	cfg.Mode = "hybrid"
	cfg.K = 3
	// Default floors reject anything with semantic < 0.2.

	results := SearchHybrid(idx, queryVec, "bananaword", cfg)
	if len(results) == 0 {
		t.Fatal("expected keepers to survive the gate; got 0 results")
	}
	for _, r := range results {
		if !strings.HasPrefix(r.File, "keeper") {
			t.Errorf("decoy %q leaked through the gate", r.File)
		}
	}
}

// TestIVFEndToEnd covers the whole IVF path: build a brute-force and an
// IVF-backed index over the same corpus, query both, and confirm
// (a) the IVF top-1 agrees with brute-force on straightforward queries,
// (b) the JSON is smaller because chunk vectors are stripped,
// (c) reloading the IVF index from disk through LoadIndex() produces a
//     working mmap'd search.
func TestIVFEndToEnd(t *testing.T) {
	info, _ := ResolveModel(DefaultModelKey)
	emb, err := LoadEmbedder(info)
	if err != nil {
		t.Fatal(err)
	}
	defer emb.Close()

	tmp := t.TempDir()
	kdir := filepath.Join(tmp, "k")
	os.Mkdir(kdir, 0o755)
	writeFile(t, filepath.Join(kdir, "apples.md"), applesMD)
	writeFile(t, filepath.Join(kdir, "cars.md"), carsMD)
	// Pad with extra docs so k-means has more than two points to cluster.
	for i := 0; i < 12; i++ {
		writeFile(t, filepath.Join(kdir, fmt.Sprintf("pad%d.md", i)),
			fmt.Sprintf("# Topic %d\n\n## Section %d\n\nFiller about subject %d unrelated to fruit or vehicles.\n",
				i, i, i))
	}

	// ---- Flat (brute-force) baseline ----
	flatCfg := DefaultIndexConfig()
	flatIdx := BuildIndex(ParseKnowledge(kdir, flatCfg), emb, flatCfg)
	flatPath := filepath.Join(tmp, "flat.json")
	if err := SaveIndex(flatIdx, flatPath); err != nil {
		t.Fatal(err)
	}
	flatSize := fileSize(t, flatPath)

	// ---- IVF-backed index over the same corpus ----
	ivfCfg := DefaultIndexConfig()
	ivfCfg.UseIVF = true
	ivfCfg.IVFCentroids = 4 // deliberately small for a 14-doc corpus
	ivfCfg.IVFNprobe = 4    // = K, so recall is effectively exact
	ivfChunks := ParseKnowledge(kdir, ivfCfg)
	ivfIdx := BuildIndex(ivfChunks, emb, ivfCfg)
	ivfPath := filepath.Join(tmp, "ivf.json")
	ivfDir := IVFSiblingPath(ivfPath)
	if _, err := BuildIVFFromIndex(ivfIdx, ivfDir, ivfCfg); err != nil {
		t.Fatal(err)
	}
	StripChunkVectors(ivfIdx)
	if err := SaveIndex(ivfIdx, ivfPath); err != nil {
		t.Fatal(err)
	}
	ivfSize := fileSize(t, ivfPath)

	// JSON with vectors stripped should be meaningfully smaller.
	if ivfSize >= flatSize {
		t.Errorf("expected IVF JSON (%d B) to be smaller than flat JSON (%d B)",
			ivfSize, flatSize)
	}

	// ---- Load and query through the same code path cmdQuery uses ----
	reloaded, err := LoadIndex(ivfPath)
	if err != nil {
		t.Fatalf("LoadIndex: %v", err)
	}
	defer reloaded.Close()
	if reloaded.IVFIndex == nil {
		t.Fatal("LoadIndex didn't attach the IVF companion")
	}

	cfg := DefaultQueryConfig()
	cfg.Mode = "hybrid"
	cfg.K = 3

	for _, tc := range []struct {
		query    string
		wantFile string
	}{
		{"how often should I change engine oil", "cars.md"},
		{"pruning orchard fruit trees", "apples.md"},
	} {
		qv := emb.Embed(tc.query)
		res := SearchHybridIVF(reloaded, reloaded.IVFIndex, qv, tc.query, cfg)
		if len(res) == 0 {
			t.Fatalf("IVF hybrid returned 0 results for %q", tc.query)
		}
		if !strings.Contains(res[0].File, tc.wantFile) {
			t.Errorf("IVF hybrid top=%q for %q, want %s",
				res[0].File, tc.query, tc.wantFile)
		}
	}

	// Semantic-only path through IVF should also work.
	qv := emb.Embed("how to store fruit")
	sem := SearchSemanticIVF(reloaded, reloaded.IVFIndex, qv, cfg)
	if len(sem) == 0 {
		t.Fatal("IVF semantic returned 0 results")
	}
}

// TestIVFLoadIndexRequiresCompanion: if Config.UseIVF is true but the
// sibling directory is missing, LoadIndex must fail loudly instead of
// returning an index with no semantic backend.
func TestIVFLoadIndexRequiresCompanion(t *testing.T) {
	tmp := t.TempDir()
	// Fake an IVF-marked index JSON without a .ivf directory.
	data, _ := json.Marshal(Index{
		Schema:    IndexSchemaVersion,
		ModelInfo: KnownModels[DefaultModelKey],
		Config:    IndexConfig{UseIVF: true, ModelKey: DefaultModelKey},
		Chunks:    nil, DocFreq: map[string]int{},
	})
	p := filepath.Join(tmp, "bad.json")
	writeFile(t, p, string(data))

	if _, err := LoadIndex(p); err == nil {
		t.Fatal("expected LoadIndex to reject a UseIVF=true index with no .ivf sibling")
	}
}

func fileSize(t *testing.T, path string) int64 {
	t.Helper()
	st, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	return st.Size()
}

func TestEvalSetValidation(t *testing.T) {
	tmp := t.TempDir()
	p := filepath.Join(tmp, "bad.json")
	writeFile(t, p, `{"queries":[{"query":"no labels"}]}`)
	if _, err := LoadEvalSet(p); err == nil {
		t.Fatal("expected validation error for query with no relevance labels")
	}
}

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

const applesMD = `# Apples

## Growing apples

Apples grow on trees in temperate climates. Most orchards prune
branches every winter to encourage fruit yield.

## Storing apples

Keep apples in a cool, dry cellar. Proper storage extends shelf life.
`

const carsMD = `# Cars

## Driving cars

Cars are motor vehicles used for transportation on roads.

## Engine maintenance

Regular oil changes keep your engine running smoothly. A typical
recommendation is every 5000 miles.
`

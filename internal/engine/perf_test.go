// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

package engine

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// perfQueries pairs realistic user queries with the filename(s) of the
// scenarios doc where the answer actually lives. Picked by scanning the
// section titles in each file so the labels are unambiguous — e.g. only
// scenarios-argocd.md contains anything about "App is OutOfSync".
var perfQueries = []struct {
	query string
	want  []string // substrings of relative file path that count as relevant
}{
	{"argocd application is out of sync after I applied a unit", []string{"scenarios-argocd.md"}},
	{"argocd app stuck progressing for hours", []string{"scenarios-argocd.md"}},
	{"flux kustomization stuck reconciling", []string{"scenarios-flux.md"}},
	{"helm release stuck in pending upgrade", []string{"scenarios-flux.md"}},
	{"refresh a unit to detect drift from live state", []string{"drift-detection.md", "scenarios-drift.md"}},
	{"schedule a nightly drift sweep across all spaces", []string{"scenarios-drift.md"}},
	{"rollback a unit to a previous revision", []string{"revisions.md", "scenarios-lifecycle.md"}},
	{"compare two revisions side by side", []string{"revisions.md", "scenarios-lifecycle.md"}},
	{"bootstrap a new confighub space from scratch", []string{"scenarios-onboarding.md"}},
	{"import an existing flux gitops setup into confighub", []string{"scenarios-onboarding.md"}},
	{"audit every unit for missing resource limits", []string{"scenarios-bulk-audit.md"}},
	{"pin container images to digests across the fleet", []string{"scenarios-bulk-audit.md"}},
	{"kubernetes deployment crashlooping after applying a unit", []string{"scenarios-kubernetes.md"}},
	{"pod stuck in crashloopbackoff trace back to unit", []string{"scenarios-incidents.md"}},
	{"install a confighub worker using the 4-phase pattern", []string{"scenarios-onboarding.md"}},
	{"trace a live resource back to its managing unit", []string{"tracing.md", "scenarios-argocd.md"}},
	{"what is a bridge and when should I use one", []string{"bridges.md"}},
	{"functions that transform unit data at apply time", []string{"functions.md"}},
}

// perfCorpus caches an end-to-end setup (flat index + ivf index + embedder
// + pre-embedded queries) so all benchmarks in this file share the same
// ~5-second build cost.
type perfCorpus struct {
	flatIdx    *Index
	ivfIdx     *Index   // loaded via LoadIndex so the IVF is mmap'd
	emb        *Embedder
	preEmbeds  [][]float32
}

var (
	perfOnce sync.Once
	perfObj  *perfCorpus
	perfErr  error
)

// locatePerfCorpus finds the scenarios knowledge directory. Priority:
//  1. BRIEF_PERF_CORPUS env var (explicit override)
//  2. ./testdata/scenarios-knowledge  (when a fixture is vendored)
//  3. ../cub-claude/cmd/assets/.claude/knowledge  (when the sibling repo
//     is checked out next to this one, e.g. in CI)
//
// If none of those resolve, the perf tests skip — they're optional
// benchmarks, not part of the correctness bar.
func locatePerfCorpus() (string, error) {
	if p := os.Getenv("BRIEF_PERF_CORPUS"); p != "" {
		if st, err := os.Stat(p); err == nil && st.IsDir() {
			return p, nil
		}
		return "", fmt.Errorf("BRIEF_PERF_CORPUS=%q is not a directory", p)
	}
	candidates := []string{
		"testdata/scenarios-knowledge",
		"../cub-claude/cmd/assets/.claude/knowledge",
	}
	for _, c := range candidates {
		if st, err := os.Stat(c); err == nil && st.IsDir() {
			return c, nil
		}
	}
	return "", fmt.Errorf("corpus not found; set BRIEF_PERF_CORPUS to point at a knowledge directory")
}

// setupPerfCorpus does the heavy one-time work: parse → build flat →
// build IVF → save → reload IVF (so the mmap path is the one being
// measured) → pre-embed every perf query.
func setupPerfCorpus(tb testing.TB) *perfCorpus {
	tb.Helper()
	perfOnce.Do(func() {
		kdir, err := locatePerfCorpus()
		if err != nil {
			perfErr = err
			return
		}

		info, _ := ResolveModel(DefaultModelKey)
		emb, err := LoadEmbedder(info)
		if err != nil {
			perfErr = fmt.Errorf("LoadEmbedder: %w", err)
			return
		}

		// Flat baseline.
		flatCfg := DefaultIndexConfig()
		flatCfg.Include = []string{"*.md"}
		flatChunks := ParseKnowledge(kdir, flatCfg)
		flatIdx := BuildIndex(flatChunks, emb, flatCfg)

		// IVF companion. K tuned for the ~180-chunk corpus; with
		// nprobe=K we effectively get exact recall, which lets us
		// measure overhead cleanly. Realistic configs would use
		// nprobe=√K≈10 at ~100k chunks.
		ivfCfg := DefaultIndexConfig()
		ivfCfg.Include = []string{"*.md"}
		ivfCfg.UseIVF = true
		ivfCfg.IVFCentroids = 32
		ivfCfg.IVFNprobe = 8

		ivfChunks := ParseKnowledge(kdir, ivfCfg)
		built := BuildIndex(ivfChunks, emb, ivfCfg)

		tmp, err := os.MkdirTemp("", "rag-perf-*")
		if err != nil {
			perfErr = err
			return
		}
		ivfJSON := filepath.Join(tmp, "ivf.json")
		ivfDir := IVFSiblingPath(ivfJSON)
		if _, err := BuildIVFFromIndex(built, ivfDir, ivfCfg); err != nil {
			perfErr = fmt.Errorf("BuildIVFFromIndex: %w", err)
			return
		}
		StripChunkVectors(built)
		if err := SaveIndex(built, ivfJSON); err != nil {
			perfErr = fmt.Errorf("SaveIndex: %w", err)
			return
		}
		// Round-trip through LoadIndex so the *mmap* path is what
		// benchmarks measure — not a fresh in-memory instance.
		ivfIdx, err := LoadIndex(ivfJSON)
		if err != nil {
			perfErr = fmt.Errorf("LoadIndex: %w", err)
			return
		}

		embeds := make([][]float32, len(perfQueries))
		for i, q := range perfQueries {
			embeds[i] = emb.Embed(q.query)
		}

		perfObj = &perfCorpus{
			flatIdx:   flatIdx,
			ivfIdx:    ivfIdx,
			emb:       emb,
			preEmbeds: embeds,
		}
		tb.Logf("perf corpus: %d flat chunks, %d ivf chunks, IVF K=%d nprobe=%d",
			len(flatIdx.Chunks), len(ivfIdx.Chunks),
			ivfIdx.IVFIndex.K, ivfIdx.IVFIndex.Nprobe)
	})
	if perfErr != nil {
		tb.Skipf("perf corpus unavailable: %v", perfErr)
	}
	return perfObj
}

// ---------- Recall test (cheap correctness guard for the IVF path) ----------

// TestScenariosIVFRecallVsFlat is the quality bar: on a realistic corpus
// with realistic queries, the IVF-backed hybrid search must rank the
// expected doc in the top-K at least as often as a brute-force cosine +
// BM25 hybrid does. Guards against regressions where IVF integration
// quietly loses recall.
func TestScenariosIVFRecallVsFlat(t *testing.T) {
	p := setupPerfCorpus(t)

	cfg := DefaultQueryConfig()
	cfg.Mode = "hybrid"
	cfg.K = 5

	var flatHits, ivfHits int
	for i, q := range perfQueries {
		qv := p.preEmbeds[i]
		flat := SearchHybrid(p.flatIdx, qv, q.query, cfg)
		ivfx := SearchHybridIVF(p.ivfIdx, p.ivfIdx.IVFIndex, qv, q.query, cfg)
		fOK, iOK := topHitsAny(flat, q.want), topHitsAny(ivfx, q.want)
		if fOK {
			flatHits++
		} else {
			t.Logf("FLAT MISS: q=%q want=%v got top-3 %s", q.query, q.want, briefTop(flat, 3))
		}
		if iOK {
			ivfHits++
		} else {
			t.Logf("IVF  MISS: q=%q want=%v got top-3 %s", q.query, q.want, briefTop(ivfx, 3))
		}
	}

	flatRate := float64(flatHits) / float64(len(perfQueries))
	ivfRate := float64(ivfHits) / float64(len(perfQueries))
	t.Logf("top-%d correct-file rate: flat=%.2f (%d/%d), ivf=%.2f (%d/%d)",
		cfg.K, flatRate, flatHits, len(perfQueries),
		ivfRate, ivfHits, len(perfQueries))

	// Floor: flat must solve ≥70% of realistic queries for the test to
	// mean anything; IVF must lose no more than 1 query of headroom.
	if flatRate < 0.70 {
		t.Fatalf("flat top-%d correct-file rate %.2f below 0.70 — corpus may have regressed",
			cfg.K, flatRate)
	}
	maxDrop := 1.0 / float64(len(perfQueries))
	if flatRate-ivfRate > maxDrop+1e-9 {
		t.Fatalf("IVF lost more than 1 query's worth of recall vs flat: flat=%.2f ivf=%.2f",
			flatRate, ivfRate)
	}
}

// scenariosEvalSet turns the perfQueries table into an EvalSet the tuner
// can consume. relevant_files carries every file substring labeled as
// relevant, so a result is considered a hit if its path has any of them
// as a suffix.
func scenariosEvalSet() *EvalSet {
	qs := make([]EvalQuery, 0, len(perfQueries))
	for _, q := range perfQueries {
		qs = append(qs, EvalQuery{Query: q.query, RelevantFiles: q.want})
	}
	return &EvalSet{Queries: qs}
}

// TestScenariosTuneToFullHitRate is the explicit "can hyperparameter
// tuning reach 100% hit@K?" experiment. It:
//
//   1. Measures the default hybrid hit@K on the flat index.
//   2. Runs tune-query with objective=hit_rate — up to 400 trials of
//      random search over weights, BM25 params, and the relevance gate.
//   3. Re-measures with the tuned config and asserts the tuner did not
//      regress and reached ≥ the human target (default 1.0).
//
// Configurable targets:
//   BRIEF_PERF_TUNE_K       top-K (default 5)
//   BRIEF_PERF_TUNE_TARGET  required final hit rate (default 1.0)
//   BRIEF_PERF_TUNE_TRIALS  random-search budget (default 400)
func TestScenariosTuneToFullHitRate(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping tuning test in -short mode")
	}
	p := setupPerfCorpus(t)

	k := envInt("BRIEF_PERF_TUNE_K", 5)
	target := envFloat("BRIEF_PERF_TUNE_TARGET", 1.0)
	trials := envInt("BRIEF_PERF_TUNE_TRIALS", 400)

	set := scenariosEvalSet()
	preQ := PreEmbedQueries(p.emb, set)

	// ---- Baseline: defaults ----
	base := DefaultQueryConfig()
	base.Mode = "hybrid"
	base.K = k
	baseMetrics := EvaluateConfig(p.flatIdx, preQ, base)
	t.Logf("baseline hybrid: hit@%d=%.4f MRR=%.4f (%d/%d queries)",
		k, baseMetrics.HitRate, baseMetrics.MRR,
		int(baseMetrics.HitRate*float64(len(set.Queries))+0.5), len(set.Queries))

	// ---- Tune ----
	best, bestMetrics := TuneQueryConfig(p.flatIdx, p.emb, set, trials, "hybrid", k, "hit_rate")

	t.Logf("tuned hybrid:    hit@%d=%.4f MRR=%.4f", k, bestMetrics.HitRate, bestMetrics.MRR)
	t.Logf("best QueryConfig: ws=%.3f wb=%.3f k1=%.3f b=%.3f hf=%.3f sf=%.3f bm=%.3f",
		best.WeightSemantic, best.WeightBM25, best.BM25K1, best.BM25B,
		best.SemanticHardFloor, best.SemanticSoftFloor, best.BM25MinForSoftZone)

	// Correctness guards:
	if bestMetrics.HitRate+1e-9 < baseMetrics.HitRate {
		t.Fatalf("tuner regressed: tuned hit=%.4f baseline hit=%.4f",
			bestMetrics.HitRate, baseMetrics.HitRate)
	}
	if bestMetrics.HitRate+1e-9 < target {
		// List the still-failing queries so the user can judge whether
		// they're ambiguously labeled or the model needs more help
		// (e.g. try a bigger model or different chunking).
		misses := findMisses(p.flatIdx, preQ, best)
		for _, m := range misses {
			t.Logf("still missing: %q  want=%v  top-3=%s", m.query, m.want, m.top3)
		}
		t.Fatalf("tuned hit@%d = %.4f, target %.4f", k, bestMetrics.HitRate, target)
	}

	// Persist the tuned config next to the repo so a human can inspect
	// it — and so it can be fed straight back as `brief recall --config`.
	out := filepath.Join(os.TempDir(), "scenarios-tuned-query.json")
	if err := SaveQueryConfig(best, out); err != nil {
		t.Logf("save tuned config: %v", err)
	} else {
		t.Logf("tuned QueryConfig saved to %s", out)
	}

	// Verify the tuned config also works on IVF — no reason it shouldn't,
	// but this catches any subtle behavioral drift between flat and IVF.
	ivfMetrics := EvaluateConfig(p.ivfIdx, preQ, best)
	t.Logf("tuned on IVF:    hit@%d=%.4f MRR=%.4f", k, ivfMetrics.HitRate, ivfMetrics.MRR)
	if ivfMetrics.HitRate+1e-9 < target {
		t.Fatalf("tuned config fails on IVF backend: hit=%.4f target=%.4f",
			ivfMetrics.HitRate, target)
	}
}

type evalMiss struct {
	query string
	want  []string
	top3  string
}

func findMisses(idx *Index, queries []PreEmbedded, cfg QueryConfig) []evalMiss {
	var out []evalMiss
	for _, q := range queries {
		res := DispatchSearch(idx, q.qVec, q.Query, cfg)
		if topHitsAny(res, q.RelevantFiles) {
			continue
		}
		out = append(out, evalMiss{
			query: q.Query, want: q.RelevantFiles, top3: briefTop(res, 3),
		})
	}
	return out
}

func envInt(k string, def int) int {
	if v := os.Getenv(k); v != "" {
		var n int
		if _, err := fmt.Sscanf(v, "%d", &n); err == nil && n > 0 {
			return n
		}
	}
	return def
}

func envFloat(k string, def float64) float64 {
	if v := os.Getenv(k); v != "" {
		var f float64
		if _, err := fmt.Sscanf(v, "%f", &f); err == nil && f > 0 {
			return f
		}
	}
	return def
}

func briefTop(rs []SearchResult, n int) string {
	if n > len(rs) {
		n = len(rs)
	}
	parts := make([]string, 0, n)
	for _, r := range rs[:n] {
		parts = append(parts, fmt.Sprintf("%s/%s", r.File, r.Title))
	}
	return "[" + strings.Join(parts, " | ") + "]"
}

func topHitsAny(results []SearchResult, want []string) bool {
	for _, r := range results {
		for _, w := range want {
			if strings.Contains(r.File, w) {
				return true
			}
		}
	}
	return false
}

// ---------- Benchmarks ----------

// All benchmarks iterate perfQueries round-robin so the cache-locality
// characteristics match a realistic mixed workload. Query vectors are
// pre-embedded to isolate search-kernel cost from tokenization + ONNX.

func BenchmarkScenariosBM25Flat(b *testing.B) {
	p := setupPerfCorpus(b)
	cfg := DefaultQueryConfig()
	cfg.Mode = "bm25"
	cfg.K = 5
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q := perfQueries[i%len(perfQueries)]
		_ = SearchBM25(p.flatIdx, q.query, cfg)
	}
}

func BenchmarkScenariosSemanticFlat(b *testing.B) {
	p := setupPerfCorpus(b)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = SearchSemantic(p.flatIdx, p.preEmbeds[i%len(p.preEmbeds)], 5)
	}
}

func BenchmarkScenariosSemanticIVF(b *testing.B) {
	p := setupPerfCorpus(b)
	cfg := DefaultQueryConfig()
	cfg.K = 5
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = SearchSemanticIVF(p.ivfIdx, p.ivfIdx.IVFIndex,
			p.preEmbeds[i%len(p.preEmbeds)], cfg)
	}
}

func BenchmarkScenariosHybridFlat(b *testing.B) {
	p := setupPerfCorpus(b)
	cfg := DefaultQueryConfig()
	cfg.Mode = "hybrid"
	cfg.K = 5
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q := perfQueries[i%len(perfQueries)]
		_ = SearchHybrid(p.flatIdx, p.preEmbeds[i%len(p.preEmbeds)], q.query, cfg)
	}
}

func BenchmarkScenariosHybridIVF(b *testing.B) {
	p := setupPerfCorpus(b)
	cfg := DefaultQueryConfig()
	cfg.Mode = "hybrid"
	cfg.K = 5
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q := perfQueries[i%len(perfQueries)]
		_ = SearchHybridIVF(p.ivfIdx, p.ivfIdx.IVFIndex,
			p.preEmbeds[i%len(p.preEmbeds)], q.query, cfg)
	}
}

// BenchmarkScenariosQueryEmbed isolates the query-time embedding cost,
// which dominates end-to-end latency at this corpus size. Useful for
// deciding whether to cache embeddings or switch to a smaller model.
func BenchmarkScenariosQueryEmbed(b *testing.B) {
	p := setupPerfCorpus(b)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = p.emb.Embed(perfQueries[i%len(perfQueries)].query)
	}
}

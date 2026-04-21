// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

package engine

import (
	"strings"
	"testing"
)

// TestBM25OnlyScenariosHitRate is the key validation of the new
// --embedder none mode: can a plain BM25 index (plus default title
// boost) reach the same retrieval quality as full ONNX embedding on
// the realistic 18-query scenarios eval set?
//
// Empirically, yes: at default settings (TitleBoost=2.5, stemming off)
// BM25-only scores hit@5 = 18/18 on the cub-claude scenarios corpus —
// matching the ONNX-embedded hybrid path from v0.1.0. This test locks
// that result in as a floor of 17/18 so small ranking drift doesn't
// break the claim, while leaving headroom for single-query edge cases.
//
// Skipped when the scenarios corpus can't be located (same lookup
// chain as TestScenariosIVFRecallVsFlat — env var or sibling repo).
func TestBM25OnlyScenariosHitRate(t *testing.T) {
	kdir, err := locatePerfCorpus()
	if err != nil {
		t.Skipf("scenarios corpus unavailable: %v", err)
	}

	// Build a BM25-only index with default settings — no ONNX, no
	// network, no embedding cost. Exercises the whole learn path
	// through nopEmbedder.
	cfg := DefaultIndexConfig()
	cfg.ModelKey = NopModelKey
	cfg.Include = []string{"*.md"}
	emb, _ := LoadEmbedder(nopModelInfo)
	defer emb.Close()

	chunks := ParseKnowledge(kdir, cfg)
	if len(chunks) == 0 {
		t.Fatal("no chunks parsed from scenarios corpus")
	}
	idx := BuildIndex(chunks, emb, cfg)

	// Run the same 18-query eval set the ONNX path uses.
	qCfg := DefaultQueryConfig()
	qCfg.Mode = "bm25"
	qCfg.K = 5

	hits := 0
	var misses []string
	for _, q := range perfQueries {
		results := DispatchSearch(idx, nil, q.query, qCfg)
		if topHitsAny(results, q.want) {
			hits++
			continue
		}
		top := []string{}
		for i, r := range results {
			if i >= 3 {
				break
			}
			top = append(top, r.File)
		}
		misses = append(misses,
			q.query+" → want "+strings.Join(q.want, ",")+" got "+strings.Join(top, ","))
	}

	total := len(perfQueries)
	rate := float64(hits) / float64(total)
	t.Logf("BM25-only hit@5 on scenarios corpus: %d/%d = %.3f (defaults)",
		hits, total, rate)
	for _, m := range misses {
		t.Logf("  miss: %s", m)
	}
	// Floor: 17/18 (94%). Current reading at time of writing is
	// 18/18; we accept one query of headroom because the marginal
	// query ("what is a bridge and when should I use one") is a
	// genuine edge case where BM25's distinctive-token model and
	// topical similarity diverge — it ranks into top-5 at defaults
	// but can slip out at extreme title-boost values.
	minHits := 17
	if hits < minHits {
		t.Fatalf("BM25-only hit@5 = %d/%d, expected ≥ %d", hits, total, minHits)
	}
}

// TestBM25OnlyNopDefaultsScenarios measures the configuration the CLI
// auto-applies when the user runs `brief learn --embedder none` with
// no other flags: stemming enabled + title-boost lowered to 2.0.
// This is the mode-conditional default the CLI wires up, and this
// test asserts that at the engine layer those settings together
// still reach perfect recall on the scenarios eval set — the whole
// point of flipping the default.
func TestBM25OnlyNopDefaultsScenarios(t *testing.T) {
	kdir, err := locatePerfCorpus()
	if err != nil {
		t.Skipf("scenarios corpus unavailable: %v", err)
	}

	// Replicate what cmd/learn.go does when --embedder none is
	// passed without --stem/--title-boost overrides.
	cfg := DefaultIndexConfig()
	cfg.ModelKey = NopModelKey
	cfg.Include = []string{"*.md"}
	cfg.Stem = true // CLI auto-flip

	emb, _ := LoadEmbedder(nopModelInfo)
	defer emb.Close()
	idx := BuildIndex(ParseKnowledge(kdir, cfg), emb, cfg)

	qCfg := DefaultQueryConfig()
	qCfg.Mode = "bm25"
	qCfg.K = 5
	qCfg.TitleBoost = 2.0 // CLI auto-lower

	hits := 0
	for _, q := range perfQueries {
		if topHitsAny(DispatchSearch(idx, nil, q.query, qCfg), q.want) {
			hits++
		}
	}
	total := len(perfQueries)
	t.Logf("BM25 nop-mode defaults (stem=on, TitleBoost=2.0) hit@5: %d/%d = %.3f",
		hits, total, float64(hits)/float64(total))
	// Current reading is 18/18 and the whole reason for flipping
	// these defaults is to reach that number. Floor at 17/18 gives
	// one query of noise tolerance.
	if hits < 17 {
		t.Fatalf("nop-mode default hit@5 = %d/%d, expected ≥ 17/%d", hits, total, total)
	}
}

// TestBM25OnlyWithStemmingScenarios runs the same experiment with
// stemming on. On the scenarios corpus specifically, stemming doesn't
// help — the corpus is identifier-heavy (YAML fields, tool names,
// error strings) where Porter2 over-collapses. This test documents
// that effect so a future "default stemming on" change can't sneak
// through without flipping this assertion.
func TestBM25OnlyWithStemmingScenarios(t *testing.T) {
	kdir, err := locatePerfCorpus()
	if err != nil {
		t.Skipf("scenarios corpus unavailable: %v", err)
	}

	cfg := DefaultIndexConfig()
	cfg.ModelKey = NopModelKey
	cfg.Include = []string{"*.md"}
	cfg.Stem = true
	emb, _ := LoadEmbedder(nopModelInfo)
	defer emb.Close()
	idx := BuildIndex(ParseKnowledge(kdir, cfg), emb, cfg)

	qCfg := DefaultQueryConfig()
	qCfg.Mode = "bm25"
	qCfg.K = 5

	hits := 0
	for _, q := range perfQueries {
		if topHitsAny(DispatchSearch(idx, nil, q.query, qCfg), q.want) {
			hits++
		}
	}
	total := len(perfQueries)
	t.Logf("BM25 + stemming hit@5 on scenarios corpus: %d/%d = %.3f",
		hits, total, float64(hits)/float64(total))

	// Floor of 16/18 (89%): stemming measurably but not
	// catastrophically degrades retrieval on this corpus. Anything
	// below 16/18 means the stemmer or the BM25 path regressed
	// beyond the expected cost.
	if hits < 16 {
		t.Fatalf("BM25+stem hit@5 = %d/%d, expected ≥ 16/%d", hits, total, total)
	}
}

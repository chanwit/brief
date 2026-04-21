// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

package engine

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"strings"
)

// EvalQuery labels one query with the files / titles that should rank high.
// A result is considered relevant if its file matches any entry in
// RelevantFiles, or its title matches any entry in RelevantTitles.
type EvalQuery struct {
	Query          string   `json:"query"`
	RelevantFiles  []string `json:"relevant_files,omitempty"`
	RelevantTitles []string `json:"relevant_titles,omitempty"`
}

// EvalSet is the wire format of the --eval JSON file passed to the tuners.
type EvalSet struct {
	Queries []EvalQuery `json:"queries"`
}

func LoadEvalSet(path string) (*EvalSet, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read eval: %w", err)
	}
	var set EvalSet
	if err := json.Unmarshal(data, &set); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	if len(set.Queries) == 0 {
		return nil, fmt.Errorf("eval set %s has no queries", path)
	}
	for i, q := range set.Queries {
		if q.Query == "" {
			return nil, fmt.Errorf("eval query %d has empty text", i)
		}
		if len(q.RelevantFiles) == 0 && len(q.RelevantTitles) == 0 {
			return nil, fmt.Errorf("eval query %d (%q) has no relevant_files or relevant_titles",
				i, q.Query)
		}
	}
	return &set, nil
}

func isRelevant(r SearchResult, eq EvalQuery) bool {
	for _, f := range eq.RelevantFiles {
		if strings.EqualFold(r.File, f) || strings.HasSuffix(r.File, f) {
			return true
		}
	}
	for _, t := range eq.RelevantTitles {
		if strings.EqualFold(r.Title, t) {
			return true
		}
	}
	return false
}

// EvalMetrics summarizes how well a query config retrieves an eval set.
//
// HitRate is the fraction of queries whose top-K results contain at least
// one relevant item — i.e. "100% recall" in the user-facing sense. MRR
// rewards ranking the relevant item at position 1; HitRate treats any
// position within K equally. For a "must show the right file" correctness
// goal, tune against HitRate.
type EvalMetrics struct {
	HitRate   float64 `json:"hit_rate_at_k"`
	MRR       float64 `json:"mrr"`
	Recall    float64 `json:"recall_at_k"`
	Precision float64 `json:"precision_at_k"`
}

func meanMetrics(all []EvalMetrics) EvalMetrics {
	if len(all) == 0 {
		return EvalMetrics{}
	}
	var m EvalMetrics
	for _, x := range all {
		m.HitRate += x.HitRate
		m.MRR += x.MRR
		m.Recall += x.Recall
		m.Precision += x.Precision
	}
	n := float64(len(all))
	m.HitRate /= n
	m.MRR /= n
	m.Recall /= n
	m.Precision /= n
	return m
}

func evalOneQuery(results []SearchResult, eq EvalQuery) EvalMetrics {
	totalRelevant := len(eq.RelevantFiles) + len(eq.RelevantTitles)
	if totalRelevant == 0 {
		return EvalMetrics{}
	}

	// Recall needs set semantics: two results matching the same relevant_file
	// count as covering one relevant item, not two.
	hitLabels := make(map[string]bool)
	relevantInTopK := 0
	firstRank := 0
	for i, r := range results {
		matched := false
		for _, f := range eq.RelevantFiles {
			if strings.EqualFold(r.File, f) || strings.HasSuffix(r.File, f) {
				hitLabels["f:"+f] = true
				matched = true
			}
		}
		for _, t := range eq.RelevantTitles {
			if strings.EqualFold(r.Title, t) {
				hitLabels["t:"+t] = true
				matched = true
			}
		}
		if matched {
			relevantInTopK++
			if firstRank == 0 {
				firstRank = i + 1
			}
		}
	}

	var m EvalMetrics
	if firstRank > 0 {
		m.MRR = 1.0 / float64(firstRank)
		m.HitRate = 1.0
	}
	m.Recall = float64(len(hitLabels)) / float64(totalRelevant)
	if len(results) > 0 {
		m.Precision = float64(relevantInTopK) / float64(len(results))
	}
	return m
}

// PreEmbedded caches the query vector so random search can evaluate hundreds
// of configs without re-running the embedding model each time.
type PreEmbedded struct {
	EvalQuery
	qVec []float32
}

func PreEmbedQueries(emb Embedder, set *EvalSet) []PreEmbedded {
	out := make([]PreEmbedded, len(set.Queries))
	for i, q := range set.Queries {
		out[i] = PreEmbedded{EvalQuery: q, qVec: emb.Embed(q.Query)}
	}
	return out
}

func EvaluateConfig(idx *Index, queries []PreEmbedded, cfg QueryConfig) EvalMetrics {
	var all []EvalMetrics
	for _, q := range queries {
		results := DispatchSearch(idx, q.qVec, q.Query, cfg)
		all = append(all, evalOneQuery(results, q.EvalQuery))
	}
	return meanMetrics(all)
}

// scoreObjective picks the scalar the tuner maximizes. "hit_rate"
// optimizes "is the relevant doc anywhere in top-K"; "mrr" optimizes
// "how high did it rank". Ties are broken by MRR so two configs with
// equal hit-rate prefer the one that ranks more relevant docs higher.
func scoreObjective(m EvalMetrics, objective string) float64 {
	switch objective {
	case "hit_rate":
		return m.HitRate + m.MRR*1e-3
	default:
		return m.MRR
	}
}

// TuneQueryConfig runs a random search over query-time knobs and returns
// the config that maximizes the chosen objective on the eval set.
// objective: "mrr" (default) or "hit_rate" (for a strict "in top-K at all"
// correctness bar).
func TuneQueryConfig(idx *Index, emb Embedder, set *EvalSet, trials int, mode string, k int, objective string) (QueryConfig, EvalMetrics) {
	if trials <= 0 {
		trials = 200
	}
	r := rand.New(rand.NewSource(42))
	queries := PreEmbedQueries(emb, set)

	best := DefaultQueryConfig()
	best.Mode = mode
	best.K = k
	bestMetrics := EvaluateConfig(idx, queries, best)
	bestScore := scoreObjective(bestMetrics, objective)
	fmt.Fprintf(os.Stderr, "tune-query: baseline hit=%.4f MRR=%.4f recall=%.4f (mode=%s k=%d, n=%d, objective=%s)\n",
		bestMetrics.HitRate, bestMetrics.MRR, bestMetrics.Recall, mode, k, len(queries), objective)

	for t := 0; t < trials; t++ {
		cfg := sampleQueryConfig(r, mode, k)
		m := EvaluateConfig(idx, queries, cfg)
		s := scoreObjective(m, objective)
		if s > bestScore {
			bestMetrics = m
			bestScore = s
			best = cfg
			fmt.Fprintf(os.Stderr, "  trial %d: hit=%.4f MRR=%.4f (new best) ws=%.2f k1=%.2f b=%.2f hf=%.2f sf=%.2f bm=%.2f\n",
				t+1, m.HitRate, m.MRR,
				cfg.WeightSemantic, cfg.BM25K1, cfg.BM25B,
				cfg.SemanticHardFloor, cfg.SemanticSoftFloor, cfg.BM25MinForSoftZone)
		}
	}
	return best, bestMetrics
}

func sampleQueryConfig(r *rand.Rand, mode string, k int) QueryConfig {
	ws := 0.1 + r.Float64()*0.8
	hf := r.Float64() * 0.4
	return QueryConfig{
		Mode:               mode,
		K:                  k,
		WeightSemantic:     ws,
		WeightBM25:         1.0 - ws,
		BM25K1:             0.5 + r.Float64()*2.0,
		BM25B:              0.25 + r.Float64()*0.75,
		SemanticHardFloor:  hf,
		SemanticSoftFloor:  hf + r.Float64()*0.3,
		BM25MinForSoftZone: r.Float64() * 0.6,
	}
}

// TuneIndexConfig grid-walks a small set of chunking strategies, building
// a fresh index for each config. Because embedding the corpus dominates
// the wall clock, keep trials small (default 8).
func TuneIndexConfig(knowledgeDir string, emb Embedder, set *EvalSet, trials int, queryCfg QueryConfig, objective string) (IndexConfig, *Index, EvalMetrics) {
	if trials <= 0 {
		trials = 8
	}
	queries := PreEmbedQueries(emb, set)

	bestCfg := DefaultIndexConfig()
	bestCfg.ModelKey = emb.Info().Key
	bestChunks := ParseKnowledge(knowledgeDir, bestCfg)
	bestIdx := BuildIndex(bestChunks, emb, bestCfg)
	bestMetrics := EvaluateConfig(bestIdx, queries, queryCfg)
	bestScore := scoreObjective(bestMetrics, objective)
	fmt.Fprintf(os.Stderr, "tune-index: baseline (strategy=heading) hit=%.4f MRR=%.4f (objective=%s)\n",
		bestMetrics.HitRate, bestMetrics.MRR, objective)

	strategies := []string{"heading", "size"}
	sizes := []int{250, 500, 750, 1000, 1500}
	overlaps := []int{0, 50, 100, 200}
	embedCaps := []int{1000, 1500, 2500}

	r := rand.New(rand.NewSource(42))
	for t := 0; t < trials; t++ {
		cfg := DefaultIndexConfig()
		cfg.ModelKey = emb.Info().Key
		cfg.ChunkStrategy = strategies[r.Intn(len(strategies))]
		cfg.ChunkSize = sizes[r.Intn(len(sizes))]
		cfg.ChunkOverlap = overlaps[r.Intn(len(overlaps))]
		if cfg.ChunkOverlap >= cfg.ChunkSize {
			cfg.ChunkOverlap = cfg.ChunkSize / 4
		}
		cfg.EmbedMaxChars = embedCaps[r.Intn(len(embedCaps))]

		chunks := ParseKnowledge(knowledgeDir, cfg)
		if len(chunks) == 0 {
			continue
		}
		idx := BuildIndex(chunks, emb, cfg)
		m := EvaluateConfig(idx, queries, queryCfg)
		s := scoreObjective(m, objective)
		fmt.Fprintf(os.Stderr, "  trial %d: strat=%s size=%d overlap=%d embed_cap=%d → hit=%.4f MRR=%.4f\n",
			t+1, cfg.ChunkStrategy, cfg.ChunkSize, cfg.ChunkOverlap, cfg.EmbedMaxChars,
			m.HitRate, m.MRR)
		if s > bestScore {
			bestMetrics = m
			bestScore = s
			bestCfg = cfg
			bestIdx = idx
		}
	}
	return bestCfg, bestIdx, bestMetrics
}

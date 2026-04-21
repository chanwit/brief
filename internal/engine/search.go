// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

package engine

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/kljensen/snowball/english"

	"github.com/chanwit/brief/ivf"
)

// SearchResult is a query hit.
type SearchResult struct {
	Score    float64 `json:"score"`
	File     string  `json:"file"`
	Title    string  `json:"title"`
	Body     string  `json:"body"`
	Semantic float64 `json:"semantic,omitempty"`
	BM25     float64 `json:"bm25,omitempty"`

	// LinkedFrom is set for results surfaced via wikilink expansion
	// (chunks reachable by 1 hop from a primary hit). Empty for
	// primary hits. Present so hook-mode output can format them as
	// "Related: …" headings and JSON consumers can tell them apart.
	LinkedFrom string `json:"linked_from,omitempty"`
}

// ---------- semantic ----------

// cosineSim assumes both inputs are L2-normalized unit vectors (runEmbed
// guarantees this for every chunk and every query vector), so cosine
// similarity collapses to the plain dot product. Avoiding the two sqrts
// is ~3× faster per call on the hot search path.
func cosineSim(a, b []float32) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var dot float64
	for i := 0; i < n; i++ {
		dot += float64(a[i]) * float64(b[i])
	}
	return dot
}

func SearchSemantic(idx *Index, queryVec []float32, topK int) []SearchResult {
	type scored struct {
		score float64
		idx   int
	}
	results := make([]scored, 0, len(idx.Chunks))
	for i, chunk := range idx.Chunks {
		results = append(results, scored{cosineSim(queryVec, chunk.Vector), i})
	}
	sort.Slice(results, func(i, j int) bool { return results[i].score > results[j].score })
	if topK > 0 && len(results) > topK {
		results = results[:topK]
	}
	var out []SearchResult
	for _, r := range results {
		c := idx.Chunks[r.idx]
		out = append(out, SearchResult{
			Score: r.score, Semantic: r.score,
			File: c.File, Title: c.Title, Body: c.Body,
		})
	}
	return out
}

// ---------- BM25 ----------

// bm25Tokenize splits text into lowercase alphanumeric tokens (keeping
// '-' and '_'). When stem is true, each token is passed through an
// English Porter2 stemmer so "refresh", "refreshes", "refreshing", and
// "refreshed" all collapse to a single index key. Both the index build
// and recall must agree on the stem flag; the value is persisted on
// the index via IndexConfig.Stem and read back at query time.
func bm25Tokenize(text string, stem bool) []string {
	text = strings.ToLower(text)
	var tokens []string
	var current []rune
	for _, r := range text {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '-' || r == '_' {
			current = append(current, r)
		} else {
			if len(current) > 1 {
				tokens = append(tokens, maybeStem(string(current), stem))
			}
			current = nil
		}
	}
	if len(current) > 1 {
		tokens = append(tokens, maybeStem(string(current), stem))
	}
	return tokens
}

// chunkMatchesTagFilter reports whether c should be considered for
// scoring given the caller's required-tag set. An empty required set
// lets every chunk through. A non-empty required set passes only
// chunks whose Tags contain at least one of the required values —
// i.e. set intersection, not subset. Chunks without any tags are
// excluded when a filter is active.
func chunkMatchesTagFilter(c *Chunk, required []string) bool {
	if len(required) == 0 {
		return true
	}
	for _, want := range required {
		for _, have := range c.Tags {
			if have == want {
				return true
			}
		}
	}
	return false
}

// maybeStem runs the Snowball English stemmer on a token when stem is
// true. Tokens with digits or dashes are passed through unchanged —
// those are identifiers (version numbers, CLI flags, CamelCase-ish
// terms) where stemming would do more harm than good.
func maybeStem(tok string, stem bool) string {
	if !stem {
		return tok
	}
	for _, r := range tok {
		if r == '-' || r == '_' || (r >= '0' && r <= '9') {
			return tok
		}
	}
	return english.Stem(tok, false)
}

// bm25ChunkScore runs the BM25 scoring formula against one chunk with
// an optional BM25F-style title boost. cfg.TitleBoost=1 reduces to
// vanilla BM25; higher values add extra weight to terms that also
// appear in the chunk's title/aliases (via TitleTermFreq). Chunks
// from indexes built before TitleTermFreq existed get no boost —
// the map lookup returns zero, which collapses the second term.
func bm25ChunkScore(idx *Index, queryTerms []string, chunk *Chunk, cfg QueryConfig) float64 {
	n := float64(len(idx.Chunks))
	boost := cfg.TitleBoost
	if boost < 1 {
		boost = 1
	}
	score := 0.0
	for _, qt := range queryTerms {
		df := float64(idx.DocFreq[qt])
		if df == 0 {
			continue
		}
		tf := float64(chunk.TermFreq[qt])
		if extra := float64(chunk.TitleTermFreq[qt]); extra > 0 {
			tf += (boost - 1) * extra
		}
		idf := math.Log((n-df+0.5)/(df+0.5) + 1.0)
		dl := float64(chunk.DocLen)
		tfNorm := (tf * (cfg.BM25K1 + 1)) / (tf + cfg.BM25K1*(1-cfg.BM25B+cfg.BM25B*dl/idx.AvgDocLen))
		score += idf * tfNorm
	}
	return score
}

func SearchBM25(idx *Index, query string, cfg QueryConfig) []SearchResult {
	queryTerms := bm25Tokenize(query, idx.Config.Stem)
	topK := cfg.K

	type scored struct {
		score float64
		idx   int
	}
	var results []scored
	for i := range idx.Chunks {
		if !chunkMatchesTagFilter(&idx.Chunks[i], cfg.RequireTags) {
			continue
		}
		score := bm25ChunkScore(idx, queryTerms, &idx.Chunks[i], cfg)
		if score > 0 {
			results = append(results, scored{score, i})
		}
	}
	sort.Slice(results, func(i, j int) bool { return results[i].score > results[j].score })
	if topK > 0 && len(results) > topK {
		results = results[:topK]
	}
	var out []SearchResult
	for _, r := range results {
		c := idx.Chunks[r.idx]
		out = append(out, SearchResult{
			Score: r.score, BM25: r.score,
			File: c.File, Title: c.Title, Body: c.Body,
		})
	}
	return out
}

// ---------- hybrid ----------

func SearchHybrid(idx *Index, queryVec []float32, query string, cfg QueryConfig) []SearchResult {
	queryTerms := bm25Tokenize(query, idx.Config.Stem)

	// Domain relevance gate.
	corpusHits := 0
	for _, qt := range queryTerms {
		if idx.DocFreq[qt] > 0 {
			corpusHits++
		}
	}
	minHits := cfg.MinQueryTerms
	if minHits <= 0 {
		minHits = 1
		if len(queryTerms) >= 3 {
			minHits = 2
		}
	}
	if corpusHits < minHits {
		return nil
	}

	type scored struct {
		semantic float64
		bm25Raw  float64
		bm25Norm float64
		combined float64
		idx      int
	}
	scores := make([]scored, len(idx.Chunks))

	// Single pass over chunks computes both semantic and BM25 scores,
	// tracking max BM25 for normalization.
	maxBM25 := 0.0
	for i := range idx.Chunks {
		c := &idx.Chunks[i]
		scores[i].idx = i
		if !chunkMatchesTagFilter(c, cfg.RequireTags) {
			// Mark as excluded by giving it unreachable scores.
			// Using -1 for semantic ensures it fails the relevance
			// gate without a special-case branch.
			scores[i].semantic = -1
			continue
		}
		scores[i].semantic = cosineSim(queryVec, c.Vector)
		scores[i].bm25Raw = bm25ChunkScore(idx, queryTerms, c, cfg)
		if scores[i].bm25Raw > maxBM25 {
			maxBM25 = scores[i].bm25Raw
		}
	}

	// Normalize BM25 and materialize the combined score once — avoids
	// recomputing inside the sort comparator.
	for i := range scores {
		if maxBM25 > 0 {
			scores[i].bm25Norm = scores[i].bm25Raw / maxBM25
		}
		scores[i].combined = cfg.WeightSemantic*scores[i].semantic + cfg.WeightBM25*scores[i].bm25Norm
	}

	// Apply the relevance gate BEFORE truncating to top-K. Otherwise a
	// cluster of high-BM25-but-off-topic hits at the top of the ranking
	// can evict eligible candidates just below the cutoff.
	filtered := scores[:0]
	for _, s := range scores {
		if passesRelevanceGate(s.semantic, s.bm25Norm, cfg) {
			filtered = append(filtered, s)
		}
	}
	scores = filtered

	sort.Slice(scores, func(i, j int) bool {
		return scores[i].combined > scores[j].combined
	})

	topK := cfg.K
	if topK > 0 && len(scores) > topK {
		scores = scores[:topK]
	}

	out := make([]SearchResult, 0, len(scores))
	for _, s := range scores {
		c := idx.Chunks[s.idx]
		out = append(out, SearchResult{
			Score:    s.combined,
			Semantic: s.semantic,
			BM25:     s.bm25Norm,
			File:     c.File, Title: c.Title, Body: c.Body,
		})
	}
	return out
}

// ---------- IVF-backed variants ----------

// autoNSemantic picks an IVF shortlist size when the user hasn't set one.
// 20× topK or at least 100, whichever is larger — gives enough headroom for
// BM25 to re-rank without blowing up latency.
func autoNSemantic(topK int) int {
	n := topK * 20
	if n < 100 {
		n = 100
	}
	return n
}

// SearchSemanticIVF is a drop-in replacement for SearchSemantic when the
// index carries an IVF companion. Skips the O(N·dim) brute-force cosine
// loop in favor of probing only the nearest centroids.
func SearchSemanticIVF(idx *Index, IVFIndex *ivf.IVFFlat, queryVec []float32, cfg QueryConfig) []SearchResult {
	nprobe := cfg.Nprobe
	hits := IVFIndex.Search(queryVec, cfg.K, nprobe)
	out := make([]SearchResult, 0, len(hits))
	for _, h := range hits {
		if int(h.ID) >= len(idx.Chunks) {
			continue
		}
		c := idx.Chunks[h.ID]
		score := float64(h.Score)
		out = append(out, SearchResult{
			Score: score, Semantic: score,
			File: c.File, Title: c.Title, Body: c.Body,
		})
	}
	return out
}

// SearchHybridIVF combines the IVF semantic shortlist with a full-corpus
// BM25 scan. BM25 over N chunks is cheap — it's hash-map lookups per query
// term — so scanning all of them keeps recall high even when the IVF
// shortlist would miss a strong lexical match.
//
// Chunks not in the IVF shortlist get semantic=0, so their combined score
// is BM25-only; the relevance gate then decides whether they survive.
func SearchHybridIVF(idx *Index, IVFIndex *ivf.IVFFlat, queryVec []float32, query string, cfg QueryConfig) []SearchResult {
	queryTerms := bm25Tokenize(query, idx.Config.Stem)

	// Relevance gate on query terms (same as brute-force path).
	corpusHits := 0
	for _, qt := range queryTerms {
		if idx.DocFreq[qt] > 0 {
			corpusHits++
		}
	}
	minHits := cfg.MinQueryTerms
	if minHits <= 0 {
		minHits = 1
		if len(queryTerms) >= 3 {
			minHits = 2
		}
	}
	if corpusHits < minHits {
		return nil
	}

	// Semantic shortlist via IVF.
	nSem := cfg.NSemantic
	if nSem <= 0 {
		nSem = autoNSemantic(cfg.K)
	}
	shortlist := IVFIndex.Search(queryVec, nSem, cfg.Nprobe)
	semanticByChunk := make(map[int]float64, len(shortlist))
	for _, h := range shortlist {
		if int(h.ID) < len(idx.Chunks) {
			semanticByChunk[int(h.ID)] = float64(h.Score)
		}
	}

	// BM25 + combine. We only keep candidates with at least one non-zero
	// signal to avoid scoring the entire corpus when most of it is silent.
	type scored struct {
		semantic float64
		bm25Raw  float64
		bm25Norm float64
		combined float64
		idx      int
	}
	scores := make([]scored, 0, len(semanticByChunk)+64)
	maxBM25 := 0.0
	for i := range idx.Chunks {
		if !chunkMatchesTagFilter(&idx.Chunks[i], cfg.RequireTags) {
			continue
		}
		bm := bm25ChunkScore(idx, queryTerms, &idx.Chunks[i], cfg)
		if bm > maxBM25 {
			maxBM25 = bm
		}
		sem, hasSem := semanticByChunk[i]
		if !hasSem && bm == 0 {
			continue
		}
		scores = append(scores, scored{semantic: sem, bm25Raw: bm, idx: i})
	}
	for i := range scores {
		if maxBM25 > 0 {
			scores[i].bm25Norm = scores[i].bm25Raw / maxBM25
		}
		scores[i].combined = cfg.WeightSemantic*scores[i].semantic +
			cfg.WeightBM25*scores[i].bm25Norm
	}

	// Filter first, then sort + truncate — matches the brute-force path so
	// both produce the same ranking up to IVF recall loss.
	filtered := scores[:0]
	for _, s := range scores {
		if passesRelevanceGate(s.semantic, s.bm25Norm, cfg) {
			filtered = append(filtered, s)
		}
	}
	scores = filtered
	sort.Slice(scores, func(i, j int) bool { return scores[i].combined > scores[j].combined })
	if cfg.K > 0 && len(scores) > cfg.K {
		scores = scores[:cfg.K]
	}

	out := make([]SearchResult, 0, len(scores))
	for _, s := range scores {
		c := idx.Chunks[s.idx]
		out = append(out, SearchResult{
			Score:    s.combined,
			Semantic: s.semantic,
			BM25:     s.bm25Norm,
			File:     c.File, Title: c.Title, Body: c.Body,
		})
	}
	return out
}

// BackendDescription is a short human-readable tag describing which
// search backend is bound to the index — used by cmd/ for diagnostic
// banners. Avoids exposing IVF internals to the CLI layer.
func BackendDescription(idx *Index, cfg QueryConfig) string {
	if idx == nil || idx.IVFIndex == nil {
		return "flat"
	}
	nprobe := cfg.Nprobe
	if nprobe <= 0 {
		nprobe = idx.IVFIndex.Nprobe
	}
	return fmt.Sprintf("ivf(K=%d,nprobe=%d)", idx.IVFIndex.K, nprobe)
}

// DispatchSearch picks the right backend automatically: if the index
// carries an IVF companion, semantic/hybrid queries go through the IVF
// path; otherwise they use brute-force. BM25 never changes. All
// non-CLI callers (the tuner, the perf harness) should use this so a
// single code path handles both backends.
//
// After primary results are selected, wikilink expansion adds up to
// cfg.MaxLinked related chunks (1-hop from the primary hits).
func DispatchSearch(idx *Index, qVec []float32, query string, cfg QueryConfig) []SearchResult {
	var primary []SearchResult
	switch cfg.Mode {
	case "bm25":
		primary = SearchBM25(idx, query, cfg)
	case "semantic":
		if idx.IVFIndex != nil {
			primary = SearchSemanticIVF(idx, idx.IVFIndex, qVec, cfg)
		} else {
			primary = SearchSemantic(idx, qVec, cfg.K)
		}
	default:
		if idx.IVFIndex != nil {
			primary = SearchHybridIVF(idx, idx.IVFIndex, qVec, query, cfg)
		} else {
			primary = SearchHybrid(idx, qVec, query, cfg)
		}
	}
	if cfg.MaxLinked > 0 {
		primary = expandLinks(idx, primary, cfg.MaxLinked)
	}
	return primary
}

// expandLinks appends up to maxAdd linked chunks to primary results,
// following each primary hit's [[wikilinks]] one hop. Duplicates (a
// link that already matches a primary hit) are skipped. Each added
// result has LinkedFrom set to the file+title that sourced the link
// so hook-mode output can group them and humans can trace provenance.
func expandLinks(idx *Index, primary []SearchResult, maxAdd int) []SearchResult {
	if maxAdd <= 0 || len(primary) == 0 {
		return primary
	}

	// primaryChunk maps (File, Title) → chunk index so we can recover
	// the exact chunk a primary hit came from (including its Links).
	// A single file can contribute multiple chunks via heading split,
	// and each section's Links are what we want to expand from.
	primaryChunk := make(map[string]int, len(idx.Chunks))
	for i, c := range idx.Chunks {
		primaryChunk[c.File+"\x00"+c.Title] = i
	}

	// fileToChunk resolves a [[target]] string (a lowercase basename
	// from extractLinks) to a representative chunk of that file. The
	// first chunk of the file wins — typically the title or intro
	// section, which is the most useful neighborhood context.
	fileToChunk := make(map[string]int, len(idx.Chunks))
	for i, c := range idx.Chunks {
		key := normalizeLinkTarget(c.File)
		if _, exists := fileToChunk[key]; !exists {
			fileToChunk[key] = i
		}
	}

	// Suppress duplicates: a linked file that's already a primary hit
	// shouldn't be re-surfaced under Related.
	alreadyPresent := make(map[string]struct{}, len(primary))
	for _, r := range primary {
		alreadyPresent[normalizeLinkTarget(r.File)] = struct{}{}
	}

	added := 0
	out := primary
	for _, r := range primary {
		if added >= maxAdd {
			break
		}
		srcIdx, ok := primaryChunk[r.File+"\x00"+r.Title]
		if !ok {
			continue
		}
		for _, target := range idx.Chunks[srcIdx].Links {
			if added >= maxAdd {
				break
			}
			if _, seen := alreadyPresent[target]; seen {
				continue
			}
			chunkIdx, ok := fileToChunk[target]
			if !ok {
				continue
			}
			alreadyPresent[target] = struct{}{}
			c := idx.Chunks[chunkIdx]
			out = append(out, SearchResult{
				File:       c.File,
				Title:      c.Title,
				Body:       c.Body,
				LinkedFrom: r.File,
			})
			added++
		}
	}
	return out
}

// normalizeLinkTarget strips directory and extension from a file path
// and lowercases it. Matches the basename normalization extractLinks
// uses on [[target]] values, so "docs/Apples.md", "Apples.md", and a
// link [[apples]] all hash to the same key.
func normalizeLinkTarget(s string) string {
	base := s
	if i := strings.LastIndexAny(s, "/\\"); i >= 0 {
		base = s[i+1:]
	}
	if i := strings.LastIndex(base, "."); i > 0 {
		base = base[:i]
	}
	return strings.ToLower(base)
}

// passesRelevanceGate applies the two-tier semantic floor from cfg.
func passesRelevanceGate(semantic, bm25Norm float64, cfg QueryConfig) bool {
	if semantic < cfg.SemanticHardFloor {
		return false
	}
	if semantic < cfg.SemanticSoftFloor {
		return bm25Norm >= cfg.BM25MinForSoftZone
	}
	return true
}

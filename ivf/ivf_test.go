// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

package ivf

import (
	"math"
	"math/rand"
	"path/filepath"
	"sort"
	"testing"
)

// TestIVFRecallVsFlat is the key correctness guarantee: IVF-Flat must
// return a top-K set that overlaps heavily with the brute-force top-K.
// We accept <100% recall (that's the whole point of ANN) but require a
// reasonable fraction for modest nprobe.
func TestIVFRecallVsFlat(t *testing.T) {
	const (
		dim    = 64
		n      = 5000
		k      = 50  // ≈ 4*sqrt(n)
		topK   = 10
		nprobe = 8   // ≈ 16% of partitions
	)
	r := rand.New(rand.NewSource(42))
	corpus := makeClusteredCorpus(r, n, dim, 20)

	ix := New(dim, k)
	if err := ix.Train(corpus, 20, 1); err != nil {
		t.Fatal(err)
	}
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i)
	}
	if err := ix.AddBatch(ids, corpus); err != nil {
		t.Fatal(err)
	}

	// Check recall@topK averaged over 50 queries.
	const nqueries = 50
	var total float64
	for q := 0; q < nqueries; q++ {
		query := randUnitVec(r, dim)
		exact := flatSearch(corpus, ids, dim, query, topK)
		approx := ix.Search(query, topK, nprobe)
		total += recallAt(approx, exact)
	}
	avg := total / float64(nqueries)
	if avg < 0.80 {
		t.Fatalf("recall@%d with nprobe=%d is %.2f, expected ≥ 0.80", topK, nprobe, avg)
	}
	t.Logf("avg recall@%d over %d queries at nprobe=%d: %.2f", topK, nqueries, nprobe, avg)
}

// TestIVFNprobeCtrlsRecall: raising nprobe must monotonically (not strictly,
// but close) improve recall. This is a sanity check that the probe
// mechanism itself is working.
func TestIVFNprobeCtrlsRecall(t *testing.T) {
	const (
		dim  = 48
		n    = 2000
		k    = 40
		topK = 5
	)
	r := rand.New(rand.NewSource(3))
	corpus := makeClusteredCorpus(r, n, dim, 15)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i)
	}
	ix := New(dim, k)
	if err := ix.Train(corpus, 20, 11); err != nil {
		t.Fatal(err)
	}
	if err := ix.AddBatch(ids, corpus); err != nil {
		t.Fatal(err)
	}

	// Same query across probe settings.
	probeSet := []int{1, 4, 16, k}
	recalls := make([]float64, len(probeSet))
	for qi := 0; qi < 30; qi++ {
		query := randUnitVec(r, dim)
		exact := flatSearch(corpus, ids, dim, query, topK)
		for i, nprobe := range probeSet {
			approx := ix.Search(query, topK, nprobe)
			recalls[i] += recallAt(approx, exact)
		}
	}
	for i := range recalls {
		recalls[i] /= 30
	}
	t.Logf("recall@%d by nprobe %v = %v", topK, probeSet, recalls)
	if recalls[len(recalls)-1] != 1.0 {
		t.Errorf("nprobe=K should give perfect recall, got %.3f", recalls[len(recalls)-1])
	}
	if recalls[0] >= recalls[len(recalls)-1] {
		t.Errorf("nprobe=1 (%.3f) should be worse than nprobe=K (%.3f)",
			recalls[0], recalls[len(recalls)-1])
	}
}

// TestIVFRoundTrip: Save then Load must produce bit-for-bit identical
// search results.
func TestIVFRoundTrip(t *testing.T) {
	const (
		dim  = 32
		n    = 400
		k    = 16
		topK = 5
	)
	r := rand.New(rand.NewSource(9))
	corpus := makeClusteredCorpus(r, n, dim, 8)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i * 3)
	}

	orig := New(dim, k)
	if err := orig.Train(corpus, 15, 1); err != nil {
		t.Fatal(err)
	}
	if err := orig.AddBatch(ids, corpus); err != nil {
		t.Fatal(err)
	}

	dir := filepath.Join(t.TempDir(), "idx")
	if err := orig.Save(dir); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}

	query := randUnitVec(r, dim)
	a := orig.Search(query, topK, 4)
	b := loaded.Search(query, topK, 4)
	if len(a) != len(b) {
		t.Fatalf("result length diff: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i].ID != b[i].ID {
			t.Errorf("rank %d: orig id=%d loaded id=%d", i, a[i].ID, b[i].ID)
		}
		if !closeEnough(a[i].Score, b[i].Score) {
			t.Errorf("rank %d: orig score=%g loaded score=%g", i, a[i].Score, b[i].Score)
		}
	}
	if loaded.Ntotal() != orig.Ntotal() {
		t.Errorf("ntotal diff: %d vs %d", orig.Ntotal(), loaded.Ntotal())
	}
}

// ---------- helpers ----------

// makeClusteredCorpus creates n L2-unit vectors distributed around
// `clusters` random centers, so k-means has real structure to find. Without
// clustering, recall tests become meaningless because every partition
// looks equivalent.
func makeClusteredCorpus(r *rand.Rand, n, dim, clusters int) []float32 {
	centers := make([][]float32, clusters)
	for i := range centers {
		centers[i] = randUnitVec(r, dim)
	}
	out := make([]float32, 0, n*dim)
	for i := 0; i < n; i++ {
		c := centers[r.Intn(clusters)]
		v := make([]float32, dim)
		for j := range v {
			// center + small gaussian noise
			v[j] = c[j] + float32(r.NormFloat64())*0.15
		}
		// re-normalize so Dot stays a valid cosine.
		normalize(v)
		out = append(out, v...)
	}
	return out
}

func randUnitVec(r *rand.Rand, dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = float32(r.NormFloat64())
	}
	normalize(v)
	return v
}

// flatSearch is the oracle: brute-force exact top-K via pure-Go Dot.
func flatSearch(corpus []float32, ids []uint64, dim int, query []float32, topK int) []Hit {
	n := len(corpus) / dim
	hits := make([]Hit, n)
	for i := 0; i < n; i++ {
		hits[i] = Hit{ID: ids[i], Score: Dot(query, corpus[i*dim:(i+1)*dim])}
	}
	sort.Slice(hits, func(i, j int) bool { return hits[i].Score > hits[j].Score })
	if topK < len(hits) {
		hits = hits[:topK]
	}
	return hits
}

// recallAt returns |approx ∩ exact| / |exact| on IDs.
func recallAt(approx, exact []Hit) float64 {
	want := make(map[uint64]struct{}, len(exact))
	for _, h := range exact {
		want[h.ID] = struct{}{}
	}
	hit := 0
	for _, h := range approx {
		if _, ok := want[h.ID]; ok {
			hit++
		}
	}
	return float64(hit) / math.Max(1, float64(len(exact)))
}

// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

package ivf

import (
	"fmt"
	"io"
	"sort"
)

// Hit is a single search result: a user-supplied ID plus the similarity
// score (dot product of unit vectors, so ∈ [-1, 1]).
type Hit struct {
	ID    uint64
	Score float32
}

// IVFFlat is an inverted-file-list index with flat (uncompressed) residuals.
// Build sequence:
//
//	ix := New(dim, k)
//	ix.Train(sample)         // pick centroids from a representative sample
//	ix.Add(id, vec)          // or ix.AddBatch
//	ix.Search(query, topK, nprobe)
//
// At query time the index scores the `nprobe` centroids nearest the query
// and brute-force-ranks only the vectors attached to those partitions, so
// wall-clock search is roughly O(K + (N/K) * nprobe) cosine evaluations.
type IVFFlat struct {
	Dim    int
	K      int
	Nprobe int // default nprobe used when Search is called with nprobe<=0

	centroids []float32  // K * Dim, L2-unit
	invlists  []invlist  // K entries
	ntotal    int
	trained   bool

	// Populated only by Open(); kept alive so the Go runtime won't GC the
	// mmap regions underlying ix.centroids and ix.invlists[*].
	mmapHandles []io.Closer
}

type invlist struct {
	ids     []uint64
	vectors []float32 // len = len(ids) * Dim
}

// New constructs an untrained index. dim is the vector dimension, k is the
// number of coarse centroids (rule of thumb: k ≈ 4·√N).
func New(dim, k int) *IVFFlat {
	if dim <= 0 || k <= 0 {
		panic("ivf.New: dim and k must be > 0")
	}
	return &IVFFlat{
		Dim:      dim,
		K:        k,
		Nprobe:   8,
		invlists: make([]invlist, k),
	}
}

// Train picks K centroids from a sample of vectors. `sample` is a flat
// slice of length len(sample)/Dim vectors. Any representative sample of
// the insert distribution works — you don't have to train on the full
// corpus; 30×K-50×K vectors is usually sufficient.
func (ix *IVFFlat) Train(sample []float32, iters int, seed int64) error {
	if len(sample)%ix.Dim != 0 {
		return fmt.Errorf("ivf.Train: sample length %d not divisible by dim %d",
			len(sample), ix.Dim)
	}
	n := len(sample) / ix.Dim
	if n < ix.K {
		return fmt.Errorf("ivf.Train: need at least K=%d sample vectors, got %d",
			ix.K, n)
	}
	if iters <= 0 {
		iters = 20
	}
	ix.centroids = trainKMeans(sample, n, ix.Dim, ix.K, iters, seed)
	ix.trained = true
	return nil
}

// SetCentroids bypasses training by installing precomputed centroids
// directly. Useful for loading a persisted index or sharing centroids
// across shards.
func (ix *IVFFlat) SetCentroids(centroids []float32) error {
	if len(centroids) != ix.K*ix.Dim {
		return fmt.Errorf("ivf.SetCentroids: want %d floats, got %d",
			ix.K*ix.Dim, len(centroids))
	}
	ix.centroids = append(ix.centroids[:0], centroids...)
	ix.trained = true
	return nil
}

// Add assigns a single vector to its nearest centroid's posting list.
func (ix *IVFFlat) Add(id uint64, vec []float32) error {
	if !ix.trained {
		return fmt.Errorf("ivf.Add: index not trained")
	}
	if len(vec) != ix.Dim {
		return fmt.Errorf("ivf.Add: vector dim %d != index dim %d", len(vec), ix.Dim)
	}
	c := ix.nearestCentroid(vec)
	il := &ix.invlists[c]
	il.ids = append(il.ids, id)
	il.vectors = append(il.vectors, vec...)
	ix.ntotal++
	return nil
}

// AddBatch is a convenience for adding n vectors at once. ids must have
// length len(vectors)/Dim.
func (ix *IVFFlat) AddBatch(ids []uint64, vectors []float32) error {
	if !ix.trained {
		return fmt.Errorf("ivf.AddBatch: index not trained")
	}
	if len(vectors)%ix.Dim != 0 {
		return fmt.Errorf("ivf.AddBatch: vectors length %d not divisible by dim %d",
			len(vectors), ix.Dim)
	}
	n := len(vectors) / ix.Dim
	if len(ids) != n {
		return fmt.Errorf("ivf.AddBatch: have %d ids for %d vectors", len(ids), n)
	}
	for i := 0; i < n; i++ {
		if err := ix.Add(ids[i], vectors[i*ix.Dim:(i+1)*ix.Dim]); err != nil {
			return err
		}
	}
	return nil
}

// Ntotal returns the number of vectors added.
func (ix *IVFFlat) Ntotal() int { return ix.ntotal }

// Close releases any mmap regions this index holds. Safe to call on an
// in-memory index (built by New/Train/Add or loaded via Load) — it's a
// no-op in that case. Safe to call multiple times.
func (ix *IVFFlat) Close() error {
	var firstErr error
	for _, h := range ix.mmapHandles {
		if err := h.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	ix.mmapHandles = nil
	return firstErr
}

// Search ranks topK nearest neighbors of `query` by probing the nprobe
// nearest centroids. Pass nprobe=0 to use ix.Nprobe. Returns descending
// by score (dot product).
func (ix *IVFFlat) Search(query []float32, topK, nprobe int) []Hit {
	if !ix.trained || ix.ntotal == 0 {
		return nil
	}
	if len(query) != ix.Dim {
		return nil
	}
	if nprobe <= 0 {
		nprobe = ix.Nprobe
	}
	if nprobe > ix.K {
		nprobe = ix.K
	}

	probed := ix.topCentroids(query, nprobe)

	heap := newTopKHeap(topK)
	for _, cIdx := range probed {
		il := &ix.invlists[cIdx]
		for i, id := range il.ids {
			s := Dot(query, il.vectors[i*ix.Dim:(i+1)*ix.Dim])
			heap.Push(Hit{ID: id, Score: s})
		}
	}
	return heap.SortedDesc()
}

func (ix *IVFFlat) nearestCentroid(vec []float32) int {
	best := 0
	bestScore := Dot(vec, ix.centroids[:ix.Dim])
	for c := 1; c < ix.K; c++ {
		s := Dot(vec, ix.centroids[c*ix.Dim:(c+1)*ix.Dim])
		if s > bestScore {
			bestScore = s
			best = c
		}
	}
	return best
}

// topCentroids returns the indices of the nprobe highest-scoring centroids.
// Used partial sort: a topK min-heap over K centroids beats full sort for
// nprobe ≪ K (the common case: K=4096, nprobe=8..32).
func (ix *IVFFlat) topCentroids(query []float32, nprobe int) []int {
	type cs struct {
		idx   int
		score float32
	}
	// Simple heap over K entries (K fits fully in RAM, ~4KB worth of
	// struct even at K=4096). For very large K we'd use a real heap
	// sized to nprobe; this is O(K log nprobe).
	scores := make([]cs, ix.K)
	for c := 0; c < ix.K; c++ {
		scores[c] = cs{idx: c, score: Dot(query, ix.centroids[c*ix.Dim:(c+1)*ix.Dim])}
	}
	sort.Slice(scores, func(i, j int) bool { return scores[i].score > scores[j].score })
	out := make([]int, nprobe)
	for i := 0; i < nprobe; i++ {
		out[i] = scores[i].idx
	}
	return out
}

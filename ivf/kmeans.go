// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

package ivf

import (
	"math"
	"math/rand"
)

// trainKMeans clusters n×d vectors into k centroids using Lloyd's algorithm
// with k-means++ initialization. Vectors are stored as a single flat
// float32 slice of length n*d. Returns centroids as k*d flat slice.
//
// For L2-unit inputs, Lloyd's step is equivalent whether we pick the
// centroid by min squared-L2 or max dot product (they're monotone
// transforms on the unit sphere), so we use Dot — the asm kernel — for
// both assignment and k-means++ seeding.
func trainKMeans(vectors []float32, n, d, k, iters int, seed int64) []float32 {
	if k <= 0 || d <= 0 || n < k {
		panic("trainKMeans: need n >= k and k,d > 0")
	}
	rng := rand.New(rand.NewSource(seed))
	centroids := kmeansPlusPlus(vectors, n, d, k, rng)
	assign := make([]int, n)
	sums := make([]float32, k*d)
	counts := make([]int, k)

	for it := 0; it < iters; it++ {
		changed := 0
		for i := 0; i < n; i++ {
			vec := vectors[i*d : (i+1)*d]
			best := 0
			bestScore := Dot(vec, centroids[:d])
			for c := 1; c < k; c++ {
				s := Dot(vec, centroids[c*d:(c+1)*d])
				if s > bestScore {
					bestScore = s
					best = c
				}
			}
			if it == 0 || assign[i] != best {
				changed++
			}
			assign[i] = best
		}
		if it > 0 && changed == 0 {
			break
		}

		// Update: centroid = mean of assigned vectors.
		for i := range sums {
			sums[i] = 0
		}
		for i := range counts {
			counts[i] = 0
		}
		for i := 0; i < n; i++ {
			c := assign[i]
			counts[c]++
			src := vectors[i*d : (i+1)*d]
			dst := sums[c*d : (c+1)*d]
			for j := 0; j < d; j++ {
				dst[j] += src[j]
			}
		}
		for c := 0; c < k; c++ {
			if counts[c] == 0 {
				// Empty cluster: reseed from a random training vector.
				pick := rng.Intn(n)
				copy(centroids[c*d:(c+1)*d], vectors[pick*d:(pick+1)*d])
				continue
			}
			inv := 1.0 / float32(counts[c])
			dst := centroids[c*d : (c+1)*d]
			src := sums[c*d : (c+1)*d]
			for j := 0; j < d; j++ {
				dst[j] = src[j] * inv
			}
		}
	}
	// Re-normalize centroids to unit length — keeps the "Dot ⇔ cosine"
	// invariant downstream even though the mean of unit vectors isn't.
	for c := 0; c < k; c++ {
		normalize(centroids[c*d : (c+1)*d])
	}
	return centroids
}

// kmeansPlusPlus picks k initial centroids so that each new seed is chosen
// with probability proportional to its squared distance from the nearest
// already-chosen seed. This dramatically improves convergence vs. random
// init.
func kmeansPlusPlus(vectors []float32, n, d, k int, rng *rand.Rand) []float32 {
	centroids := make([]float32, k*d)

	first := rng.Intn(n)
	copy(centroids[:d], vectors[first*d:(first+1)*d])

	// distances[i] = min squared-L2 distance from vector i to any chosen centroid.
	distances := make([]float32, n)
	for i := 0; i < n; i++ {
		distances[i] = SquaredL2(vectors[i*d:(i+1)*d], centroids[:d])
	}

	for c := 1; c < k; c++ {
		var sum float64
		for _, x := range distances {
			sum += float64(x)
		}
		if sum == 0 {
			// All identical; pick another random one.
			pick := rng.Intn(n)
			copy(centroids[c*d:(c+1)*d], vectors[pick*d:(pick+1)*d])
			continue
		}
		target := rng.Float64() * sum
		cum := 0.0
		chosen := n - 1
		for i, x := range distances {
			cum += float64(x)
			if cum >= target {
				chosen = i
				break
			}
		}
		copy(centroids[c*d:(c+1)*d], vectors[chosen*d:(chosen+1)*d])
		newC := centroids[c*d : (c+1)*d]
		for i := 0; i < n; i++ {
			nd := SquaredL2(vectors[i*d:(i+1)*d], newC)
			if nd < distances[i] {
				distances[i] = nd
			}
		}
	}
	return centroids
}

func normalize(v []float32) {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	if s == 0 {
		return
	}
	inv := float32(1.0 / math.Sqrt(s))
	for i := range v {
		v[i] *= inv
	}
}

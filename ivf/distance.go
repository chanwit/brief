// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

// Package ivf is an inverted-file-list (IVF-Flat) approximate nearest
// neighbor index for dense float32 vectors. Designed for RAG workloads at
// 1M–100M vectors where brute force is too slow but an HNSW graph's memory
// footprint is overkill.
//
// Inputs are expected to be L2-unit vectors; ranking uses plain dot product
// (equivalent to cosine similarity on unit vectors and to argmin ||a-b||²).
package ivf

// Dot returns the dot product of two equal-length float32 slices.
//
// It is a function variable so the amd64 build can replace it with an AVX2
// kernel at init time (see distance_amd64.go). On other architectures
// (including arm64) this stays as dotGeneric.
//
// Why no arm64 NEON kernel: Go's arm64 assembler currently lacks
// 128-bit (.S4 / .D2) encodings for VFMLA, VFMUL, VFADD, and VFADDP —
// those are marked TODO in cmd/asm/internal/asm/testdata/arm64enc.s.
// Only .S2 (64-bit half-vector) variants are implemented. A proper
// NEON kernel is worth revisiting once Go ships .S4 encodings, or by
// emitting WORD-encoded raw instructions. For now the scalar Go path
// is Go's own compiler-vectorized baseline, which Apple Silicon in
// particular handles well.
var Dot func(a, b []float32) float32 = dotGeneric

func dotGeneric(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var s float32
	for i := 0; i < n; i++ {
		s += a[i] * b[i]
	}
	return s
}

// SquaredL2 is the squared Euclidean distance. For L2-unit vectors,
// ||a-b||² = 2 - 2⟨a,b⟩, so callers that already have normalized inputs
// should prefer Dot and convert if they need an L2 number.
func SquaredL2(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var s float32
	for i := 0; i < n; i++ {
		d := a[i] - b[i]
		s += d * d
	}
	return s
}

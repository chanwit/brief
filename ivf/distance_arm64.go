// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

//go:build arm64

package ivf

// dotNEONCore is the hand-written arm64 NEON kernel defined in
// distance_arm64.s. It processes 8 floats per main-loop iteration with two
// independent accumulators to hide FMLA latency, then folds a 4-wide tail
// and a scalar residue.
//
// NEON (Advanced SIMD) is required by the ARMv8-A baseline and is present
// on every Apple Silicon chip and every Linux arm64 target we ship, so no
// runtime feature check is needed — we install this kernel unconditionally
// in init().
//
//go:noescape
func dotNEONCore(a, b *float32, n int) float32

func dotNEON(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	if n == 0 {
		return 0
	}
	return dotNEONCore(&a[0], &b[0], n)
}

func init() {
	Dot = dotNEON
}

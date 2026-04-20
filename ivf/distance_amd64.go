// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

//go:build amd64

package ivf

import "golang.org/x/sys/cpu"

// dotAVX2Core is the hand-written AVX2 kernel defined in distance_amd64.s.
// It processes 16 floats per outer iteration with two independent accumulator
// registers to hide the VADDPS latency, then folds 8/4/scalar tails before
// horizontal-summing.
//
//go:noescape
func dotAVX2Core(a, b *float32, n int) float32

func dotAVX2(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	if n == 0 {
		return 0
	}
	return dotAVX2Core(&a[0], &b[0], n)
}

func init() {
	if cpu.X86.HasAVX2 {
		Dot = dotAVX2
	}
}

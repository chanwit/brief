// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

//go:build arm64

package ivf

// dotNEONCore is the hand-written arm64 NEON kernel defined in
// distance_arm64.s. Processes 4 floats per main-loop iteration via two
// independent .S2 VFMLA accumulators, then a 2-wide tail and a scalar
// residue.
//
// The kernel is deliberately limited to .S2 (64-bit, 2 floats) vector
// operations: Go's arm64 assembler currently implements only two vector
// float mnemonics (VFMLA, VFMLS) and has no VFADD / VFMUL / VADDP at any
// arrangement. That rules out a .S4 kernel and also rules out vector
// horizontal reduction — reduction happens via VMOV (lane → GPR) +
// FMOVS (GPR → FP register) + scalar FADDS. It's ugly but correct.
//
// NEON is mandatory in ARMv8-A so no runtime capability check is needed.
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

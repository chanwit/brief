// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

#include "textflag.h"

// func dotNEONCore(a, b *float32, n int) float32
//
// Plan 9 asm for arm64, ARMv8-A NEON (Advanced SIMD).
//
// Register usage:
//   R0 = &a[0]
//   R1 = &b[0]
//   R2 = remaining element count (mutated through the function)
//   R3 = main-loop iteration count (n >> 3)
//   V0, V1 = two independent float32×4 accumulators — breaking the FMLA
//            dependency chain lets the M1/M2 FPU issue two FMLAs per
//            cycle instead of serializing on ~3-cycle latency.
//   V2..V5 = load temporaries
//   F0..F3 = scalar float temporaries in the tail
//
// Arg frame: 3 × 8 bytes (two pointers + int) + 4 bytes return = 28.
TEXT ·dotNEONCore(SB), NOSPLIT, $0-28
	MOVD a+0(FP), R0
	MOVD b+8(FP), R1
	MOVD n+16(FP), R2

	// Zero both accumulators via XOR-self. VEOR with .B16 view clears the
	// full 128-bit register; any lane-view interpretation afterwards still
	// reads zero floats.
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16

	// Main 8-wide loop with 2-way ILP.
	MOVD R2, R3
	LSR  $3, R3, R3            // R3 = n / 8
	CBZ  R3, after_main

loop8:
	VLD1.P 32(R0), [V2.S4, V3.S4]   // a[i..i+7]
	VLD1.P 32(R1), [V4.S4, V5.S4]   // b[i..i+7]
	VFMLA  V4.S4, V2.S4, V0.S4      // V0 += V2 * V4
	VFMLA  V5.S4, V3.S4, V1.S4      // V1 += V3 * V5
	SUB    $1, R3, R3
	CBNZ   R3, loop8

after_main:
	AND $7, R2, R2             // R2 = n % 8

	// Optional 4-wide step (handles n % 8 ∈ {4, 5, 6, 7}).
	CMP  $4, R2
	BLT  hsum
	VLD1.P 16(R0), [V2.S4]
	VLD1.P 16(R1), [V3.S4]
	VFMLA  V3.S4, V2.S4, V0.S4
	SUB    $4, R2, R2

hsum:
	// Horizontal sum of each accumulator. VADDP with .S4 lane arrangement
	// selects the single-precision FADDP variant: pairwise float add
	// [a,b,c,d] → [a+b, c+d, a+b, c+d]. Two applications collapse a
	// 4-lane vector into "all lanes equal the total". After that F0/F1
	// (which alias V0.S[0] / V1.S[0]) hold the per-accumulator totals.
	VADDP V0.S4, V0.S4, V0.S4
	VADDP V1.S4, V1.S4, V1.S4
	VADDP V0.S4, V0.S4, V0.S4
	VADDP V1.S4, V1.S4, V1.S4
	FADDS F1, F0, F0           // merge the two accumulators

	// Scalar tail for the 0..3 trailing elements.
	CBZ R2, done

scalar:
	FMOVS (R0), F2
	FMOVS (R1), F3
	FMULS F3, F2, F2
	FADDS F2, F0, F0
	ADD   $4, R0, R0
	ADD   $4, R1, R1
	SUB   $1, R2, R2
	CBNZ  R2, scalar

done:
	FMOVS F0, ret+24(FP)
	RET

// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

#include "textflag.h"

// func dotNEONCore(a, b *float32, n int) float32
//
// .S2-only NEON kernel. Constrained by Go's arm64 assembler which
// implements only two vector float mnemonics (VFMLA, VFMLS) and no
// vector float add/subtract/pairwise-add. Every mnemonic used here is
// verified present in cmd/asm/internal/asm/testdata/arm64enc.s.
//
// Register usage:
//   R0 = &a[0]
//   R1 = &b[0]
//   R2 = remaining element count
//   R3 = scratch (loop counter, later GPR shuttle for lane extraction)
//   V0, V1 = two independent 2-float accumulators (ILP: FMLA has ~3-4
//            cycle latency on M1/M2; alternating accumulators lets two
//            retire per cycle)
//   V2..V5 = load temporaries
//   F0 = final scalar accumulator, also return register (aliased with V0.S[0])
//   F1..F3 = scratch FP registers
//
// Arg frame: 3 × 8 byte args + 4 byte return = 28.
TEXT ·dotNEONCore(SB), NOSPLIT, $0-28
	MOVD a+0(FP), R0
	MOVD b+8(FP), R1
	MOVD n+16(FP), R2

	// Zero both accumulators (.B16 view zeros the whole 128-bit reg; we
	// only read/write the low 64 bits via .S2, so lanes 2-3 stay zero
	// forever and don't matter).
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16

	// Main 4-wide loop.
	MOVD R2, R3
	LSR  $2, R3, R3            // R3 = n / 4
	CBZ  R3, tail2

loop4:
	// Load 8 bytes (2 floats) into each of V2, V3 (for a) and V4, V5 (for b).
	// .D1 arrangement is 1×float64 = 8 bytes; we treat those bytes as
	// .S2 (2×float32) in the subsequent VFMLA, which is valid because
	// .D1 and .S2 are both 64-bit arrangements and the load only moves
	// bytes — it doesn't interpret them.
	VLD1.P 8(R0), [V2.D1]
	VLD1.P 8(R1), [V4.D1]
	VFMLA  V4.S2, V2.S2, V0.S2     // V0 += V2 * V4 (2 lanes)

	VLD1.P 8(R0), [V3.D1]
	VLD1.P 8(R1), [V5.D1]
	VFMLA  V5.S2, V3.S2, V1.S2     // V1 += V3 * V5 (2 lanes)

	SUB    $1, R3, R3
	CBNZ   R3, loop4

tail2:
	AND $3, R2, R2             // R2 = n % 4

	// 2-wide tail if remaining ≥ 2.
	CMP  $2, R2
	BLT  reduce
	VLD1.P 8(R0), [V2.D1]
	VLD1.P 8(R1), [V3.D1]
	VFMLA  V3.S2, V2.S2, V0.S2
	SUB    $2, R2, R2

reduce:
	// Horizontal reduce V0 and V1 into scalar F0.
	// Go's arm64 asm has no float vector pairwise-add, so we extract
	// each lane via the integer path:
	//     VMOV Vn.S[i], Rk   (lane i → 32-bit GPR, bit-preserving)
	//     FMOVS Rk, Fk       (GPR bits → FP register, bit-preserving)
	// and accumulate with scalar FADDS.
	VMOV  V0.S[0], R3
	FMOVS R3, F0
	VMOV  V0.S[1], R3
	FMOVS R3, F1
	FADDS F1, F0, F0           // F0 = V0[0] + V0[1]

	VMOV  V1.S[0], R3
	FMOVS R3, F2
	FADDS F2, F0, F0
	VMOV  V1.S[1], R3
	FMOVS R3, F3
	FADDS F3, F0, F0           // F0 = V0[0]+V0[1]+V1[0]+V1[1]

	// Scalar residue (0 or 1 leftover element).
	CBZ R2, done

scalar:
	FMOVS  (R0), F2
	FMOVS  (R1), F3
	FMADDS F3, F2, F0, F0      // F0 = F0 + F2 * F3
	ADD    $4, R0, R0
	ADD    $4, R1, R1
	SUB    $1, R2, R2
	CBNZ   R2, scalar

done:
	FMOVS F0, ret+24(FP)
	RET

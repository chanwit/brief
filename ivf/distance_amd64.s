// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

#include "textflag.h"

// func dotAVX2Core(a, b *float32, n int) float32
//
// Register usage:
//   AX  = &a[0]
//   BX  = &b[0]
//   CX  = remaining element count
//   DX  = main-loop iteration count (n >> 4)
//   Y0,Y1 = two independent float32×8 accumulators (breaks the VADDPS
//           dependency chain so two adds retire per cycle on Haswell+).
//   Y2..Y5 / X2..X3 = load/multiply temporaries
//
// IMPORTANT: VEX-encoded 128-bit VADDPS/VMULPS zero the upper 128 bits of
// the destination YMM register. That means once we've entered the XMM-
// width tail, Y0's high lane is gone — so we must collapse Y0 → X0 first.
TEXT ·dotAVX2Core(SB), NOSPLIT, $0-28
	MOVQ a+0(FP), AX
	MOVQ b+8(FP), BX
	MOVQ n+16(FP), CX

	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1

	MOVQ CX, DX
	SHRQ $4, DX            // DX = n / 16
	JZ   after16

main16:
	VMOVUPS (AX), Y2
	VMOVUPS 32(AX), Y3
	VMOVUPS (BX), Y4
	VMOVUPS 32(BX), Y5
	VMULPS  Y4, Y2, Y2
	VMULPS  Y5, Y3, Y3
	VADDPS  Y2, Y0, Y0
	VADDPS  Y3, Y1, Y1
	ADDQ    $64, AX
	ADDQ    $64, BX
	DECQ    DX
	JNZ     main16

after16:
	VADDPS Y1, Y0, Y0      // fold acc1 into acc0
	ANDQ   $15, CX         // CX = n % 16

	// Process 8 floats while still in YMM land.
	CMPQ CX, $8
	JL   collapse
	VMOVUPS (AX), Y2
	VMOVUPS (BX), Y3
	VMULPS  Y3, Y2, Y2
	VADDPS  Y2, Y0, Y0
	ADDQ    $32, AX
	ADDQ    $32, BX
	SUBQ    $8, CX

collapse:
	// Fold Y0 (8 partial sums) down to X0 (4 partial sums). After this
	// point any VEX.128 op would zero Y0's upper lane, so we must be done
	// reading from it.
	VEXTRACTF128 $1, Y0, X1
	VADDPS  X1, X0, X0

	// Process 4 floats in XMM if available.
	CMPQ CX, $4
	JL   hsum
	VMOVUPS (AX), X2
	VMOVUPS (BX), X3
	VMULPS  X3, X2, X2
	VADDPS  X2, X0, X0
	ADDQ    $16, AX
	ADDQ    $16, BX
	SUBQ    $4, CX

hsum:
	// Horizontal sum of X0[0..3] into X0[0].
	VHADDPS X0, X0, X0     // [a+b, c+d, a+b, c+d]
	VHADDPS X0, X0, X0     // [sum, sum, sum, sum]

	// Scalar tail for the last 0–3 elements.
	TESTQ CX, CX
	JZ    done

scalar:
	MOVSS (AX), X1
	MULSS (BX), X1
	ADDSS X1, X0
	ADDQ  $4, AX
	ADDQ  $4, BX
	DECQ  CX
	JNZ   scalar

done:
	MOVSS      X0, ret+24(FP)
	VZEROUPPER
	RET

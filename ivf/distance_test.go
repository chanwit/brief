// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

package ivf

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"runtime"
	"testing"
)

// TestActiveKernel prints which Dot implementation is in use on this
// host. Not a correctness test — just useful context in CI logs so you
// can tell at a glance that macos-latest actually hit the NEON kernel
// and isn't silently falling back to dotGeneric.
func TestActiveKernel(t *testing.T) {
	name := runtime.FuncForPC(reflect.ValueOf(Dot).Pointer()).Name()
	t.Logf("active Dot kernel: %s  (GOOS=%s GOARCH=%s)", name, runtime.GOOS, runtime.GOARCH)

	// Make the expectation explicit on known-accelerated targets so a CI
	// regression (e.g. init ordering break) surfaces immediately.
	switch runtime.GOARCH {
	case "amd64":
		if !isAccelerated(name) {
			t.Fatalf("amd64 build dispatched to %s — AVX2 kernel not installed", name)
		}
	case "arm64":
		if !isAccelerated(name) {
			t.Fatalf("arm64 build dispatched to %s — NEON kernel not installed", name)
		}
	}
}

func isAccelerated(funcName string) bool {
	// Any implementation other than the scalar fallback counts as
	// accelerated. On amd64 that's dotAVX2; on arm64 that's dotNEON.
	return funcName != "" && funcName != fmt.Sprintf("%s.dotGeneric", "github.com/chanwit/rag-engine/ivf")
}

// TestDotMatchesGeneric: the accelerated Dot (whichever the current arch
// dispatched to) must agree with the scalar fallback on random inputs at
// every length class that stresses the asm tails (head-16, tail-8,
// tail-4, scalar remainder).
func TestDotMatchesGeneric(t *testing.T) {
	lengths := []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 23, 31, 32,
		64, 65, 100, 128, 384, 385, 768, 1000}
	r := rand.New(rand.NewSource(7))

	for _, n := range lengths {
		a := randVec(r, n)
		b := randVec(r, n)
		got := Dot(a, b)
		want := dotGeneric(a, b)
		if !closeEnough(got, want) {
			t.Errorf("n=%d: Dot=%g dotGeneric=%g (diff=%g)",
				n, got, want, math.Abs(float64(got-want)))
		}
	}
}

// TestDotZeroLen: both empty inputs return 0; mismatched lengths use the
// shorter one (defensive, not expected in IVF but keeps the kernel
// crash-proof).
func TestDotEdgeCases(t *testing.T) {
	if Dot(nil, nil) != 0 {
		t.Fatal("Dot(nil, nil) != 0")
	}
	if Dot([]float32{}, []float32{}) != 0 {
		t.Fatal("Dot([], []) != 0")
	}
	a := []float32{1, 2, 3, 4, 5}
	b := []float32{1, 1, 1}
	if got := Dot(a, b); got != 6 {
		t.Fatalf("mismatched len: Dot=%g want 6", got)
	}
}

func randVec(r *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = r.Float32()*2 - 1
	}
	return v
}

// closeEnough uses a relative tolerance because the asm kernel accumulates
// in a different summation order than the scalar loop, so floating-point
// results differ in the last couple of bits.
func closeEnough(a, b float32) bool {
	diff := math.Abs(float64(a - b))
	if diff < 1e-4 {
		return true
	}
	mag := math.Max(math.Abs(float64(a)), math.Abs(float64(b)))
	return diff/mag < 1e-5
}

// BenchmarkDotGeneric / BenchmarkDotAccelerated let you see the asm win.
// Run with: go test -bench=Dot ./ivf/
func BenchmarkDotGeneric384(b *testing.B) {
	benchDotFunc(b, dotGeneric, 384)
}

func BenchmarkDotAccelerated384(b *testing.B) {
	benchDotFunc(b, Dot, 384)
}

func BenchmarkDotGeneric768(b *testing.B) {
	benchDotFunc(b, dotGeneric, 768)
}

func BenchmarkDotAccelerated768(b *testing.B) {
	benchDotFunc(b, Dot, 768)
}

func benchDotFunc(b *testing.B, f func(a, b []float32) float32, n int) {
	r := rand.New(rand.NewSource(1))
	a := randVec(r, n)
	v := randVec(r, n)
	b.SetBytes(int64(n * 4 * 2))
	b.ResetTimer()
	var s float32
	for i := 0; i < b.N; i++ {
		s += f(a, v)
	}
	runtime_keepalive(s)
}

// runtime_keepalive prevents the compiler from dead-code-eliminating the
// benchmark loop body.
var sink float32

func runtime_keepalive(x float32) { sink = x }

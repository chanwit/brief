// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

//go:build (linux || darwin) && (amd64 || arm64)

package ivf

import (
	"math/rand"
	"path/filepath"
	"testing"
)

// TestMmapOpenMatchesLoad: an index opened via mmap must return identical
// top-K results to the same index loaded into heap memory. Guards against
// subtle mistakes in pointer math / slice length calculations.
func TestMmapOpenMatchesLoad(t *testing.T) {
	const (
		dim  = 48
		n    = 800
		k    = 32
		topK = 10
	)
	r := rand.New(rand.NewSource(17))
	corpus := makeClusteredCorpus(r, n, dim, 10)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i*7) + 100
	}

	ix := New(dim, k)
	if err := ix.Train(corpus, 20, 3); err != nil {
		t.Fatal(err)
	}
	if err := ix.AddBatch(ids, corpus); err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(t.TempDir(), "idx")
	if err := ix.Save(dir); err != nil {
		t.Fatal(err)
	}

	loaded, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	opened, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer opened.Close()

	if opened.Ntotal() != loaded.Ntotal() {
		t.Fatalf("ntotal mismatch: mmap=%d load=%d", opened.Ntotal(), loaded.Ntotal())
	}

	// Hit each path over many queries — a mmap offset bug tends to show up
	// as wrong IDs on some queries but not others.
	for q := 0; q < 30; q++ {
		query := randUnitVec(r, dim)
		a := loaded.Search(query, topK, 4)
		b := opened.Search(query, topK, 4)
		if len(a) != len(b) {
			t.Fatalf("q=%d: length mismatch %d vs %d", q, len(a), len(b))
		}
		for i := range a {
			if a[i].ID != b[i].ID {
				t.Errorf("q=%d rank %d: load id=%d mmap id=%d", q, i, a[i].ID, b[i].ID)
			}
			if !closeEnough(a[i].Score, b[i].Score) {
				t.Errorf("q=%d rank %d: score diverged %g vs %g",
					q, i, a[i].Score, b[i].Score)
			}
		}
	}
}

// TestMmapCloseIsIdempotent: double-Close must not panic or return a non-
// nil error — matters because the caller may defer Close and have an
// error path that also closes.
func TestMmapCloseIsIdempotent(t *testing.T) {
	const dim, k = 16, 8
	r := rand.New(rand.NewSource(1))
	corpus := makeClusteredCorpus(r, 64, dim, 4)
	ids := make([]uint64, 64)
	for i := range ids {
		ids[i] = uint64(i)
	}
	ix := New(dim, k)
	if err := ix.Train(corpus, 10, 1); err != nil {
		t.Fatal(err)
	}
	if err := ix.AddBatch(ids, corpus); err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(t.TempDir(), "i")
	if err := ix.Save(dir); err != nil {
		t.Fatal(err)
	}
	opened, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := opened.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if err := opened.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}

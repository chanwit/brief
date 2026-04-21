// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

package engine

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestBM25OnlyLearnAndRecall covers the --embedder none path end-to-end.
// It uses nopEmbedder directly (no ONNX setup, no TestMain-installed
// environment required) to prove the BM25-only path is self-contained.
func TestBM25OnlyLearnAndRecall(t *testing.T) {
	tmp := t.TempDir()
	kdir := filepath.Join(tmp, "k")
	mustMkdir(t, kdir)
	writeFile(t, filepath.Join(kdir, "cars.md"), carsMD)
	writeFile(t, filepath.Join(kdir, "apples.md"), applesMD)

	// Build a BM25-only index via the nop embedder. No ONNX touch.
	cfg := DefaultIndexConfig()
	cfg.ModelKey = NopModelKey
	emb, err := LoadEmbedder(nopModelInfo)
	if err != nil {
		t.Fatalf("LoadEmbedder(nop): %v", err)
	}
	defer emb.Close()

	chunks := ParseKnowledge(kdir, cfg)
	if len(chunks) == 0 {
		t.Fatal("no chunks parsed")
	}
	idx := BuildIndex(chunks, emb, cfg)
	if idx.ModelInfo.Key != NopModelKey {
		t.Fatalf("index ModelInfo = %q, want %q", idx.ModelInfo.Key, NopModelKey)
	}
	for _, c := range idx.Chunks {
		if c.Vector != nil {
			t.Errorf("BM25-only chunk %q has a vector; expected nil", c.Title)
		}
	}

	// Round-trip through disk.
	path := filepath.Join(tmp, "idx.json")
	if err := SaveIndex(idx, path); err != nil {
		t.Fatalf("SaveIndex: %v", err)
	}
	loaded, err := LoadIndex(path)
	if err != nil {
		t.Fatalf("LoadIndex: %v", err)
	}

	// BM25 recall should still rank the right doc even with no vectors.
	qCfg := DefaultQueryConfig()
	qCfg.Mode = "bm25"
	results := SearchBM25(loaded, "how often to change oil", qCfg)
	if len(results) == 0 || !strings.Contains(results[0].File, "cars") {
		t.Fatalf("BM25-only recall top=%v, want cars.md", results)
	}
}

// TestHasEmbeddings sanity-checks the predicate that gates semantic /
// hybrid code paths on the presence of real embeddings.
func TestHasEmbeddings(t *testing.T) {
	var nop nopEmbedder
	if HasEmbeddings(nop) {
		t.Error("nopEmbedder should not report HasEmbeddings")
	}
	info, _ := ResolveModel(DefaultModelKey)
	if info.Dim == 0 {
		t.Fatal("default model should have Dim > 0")
	}
	onnx, err := LoadEmbedder(info)
	if err != nil {
		t.Skipf("default model not available locally: %v", err)
	}
	defer onnx.Close()
	if !HasEmbeddings(onnx) {
		t.Error("onnxEmbedder should report HasEmbeddings")
	}
}

func mustMkdir(t *testing.T, path string) {
	t.Helper()
	if err := os.Mkdir(path, 0o755); err != nil {
		t.Fatal(err)
	}
}

// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

package engine

import (
	"path/filepath"
	"strings"
	"testing"
)

// ---------- stemming ----------

// TestStemmingCollapsesInflections: with Stem=true, a body containing
// "refresh" must match a query for "refreshing" (and vice versa).
// Verifies the tokenizer stems both sides consistently.
func TestStemmingCollapsesInflections(t *testing.T) {
	tmp := t.TempDir()
	writeFile(t, filepath.Join(tmp, "a.md"),
		"# Note\n\n## Procedure\n\nRun a full refresh to recover.\n")

	cfg := DefaultIndexConfig()
	cfg.ModelKey = NopModelKey
	cfg.Stem = true
	emb, _ := LoadEmbedder(nopModelInfo)
	defer emb.Close()
	idx := BuildIndex(ParseKnowledge(tmp, cfg), emb, cfg)

	qCfg := DefaultQueryConfig()
	qCfg.Mode = "bm25"
	qCfg.MaxLinked = 0

	// "refreshing" → stem "refresh" → matches the body token.
	results := SearchBM25(idx, "refreshing", qCfg)
	if len(results) == 0 {
		t.Fatal("expected stemmed query to hit, got no results")
	}
	if !strings.Contains(results[0].Body, "refresh") {
		t.Fatalf("top result body missing 'refresh': %q", results[0].Body)
	}
}

// TestStemmingOffDoesNotMatch: the same corpus built with Stem=false
// must NOT match "refreshing" when the body only has "refresh". Guards
// against accidentally leaving stemming on.
func TestStemmingOffDoesNotMatch(t *testing.T) {
	tmp := t.TempDir()
	writeFile(t, filepath.Join(tmp, "a.md"),
		"# Note\n\n## Procedure\n\nRun a full refresh to recover.\n")

	cfg := DefaultIndexConfig()
	cfg.ModelKey = NopModelKey
	cfg.Stem = false
	emb, _ := LoadEmbedder(nopModelInfo)
	defer emb.Close()
	idx := BuildIndex(ParseKnowledge(tmp, cfg), emb, cfg)

	qCfg := DefaultQueryConfig()
	qCfg.Mode = "bm25"
	qCfg.MaxLinked = 0

	// Without stemming "refreshing" doesn't appear in the doc-freq
	// table, so BM25 has nothing to score — zero hits.
	results := SearchBM25(idx, "refreshing", qCfg)
	if len(results) != 0 {
		t.Fatalf("expected 0 hits without stemming, got %d", len(results))
	}
}

// TestStemmerSkipsIdentifiers: tokens with digits or dashes must pass
// through unchanged so version strings ("v1.2.3") and CLI flags
// ("--use-ivf") don't get mangled by the English stemmer.
func TestStemmerSkipsIdentifiers(t *testing.T) {
	cases := []string{"--use-ivf", "v1", "kube-apiserver", "rag2024"}
	for _, in := range cases {
		got := maybeStem(in, true)
		if got != in {
			t.Errorf("maybeStem(%q)=%q, want unchanged", in, got)
		}
	}
}

// ---------- BM25F title boost ----------

// TestTitleBoostFavorsHeadings: a query term appearing in a chunk's
// title must score higher than the same term appearing only in body
// of another chunk. Core BM25F invariant.
func TestTitleBoostFavorsHeadings(t *testing.T) {
	tmp := t.TempDir()
	// "oil" in first chunk's title; just mentioned in passing in
	// the second chunk's body. Both chunks share the same corpus,
	// same doc lengths.
	writeFile(t, filepath.Join(tmp, "a.md"),
		"# Cars\n\n"+
			"## Changing oil\n\n"+
			"Regular replacement helps.\n\n"+
			"## Other topic\n\n"+
			"Unrelated content that briefly mentions oil near the end.\n")

	cfg := DefaultIndexConfig()
	cfg.ModelKey = NopModelKey
	emb, _ := LoadEmbedder(nopModelInfo)
	defer emb.Close()
	idx := BuildIndex(ParseKnowledge(tmp, cfg), emb, cfg)

	qCfg := DefaultQueryConfig()
	qCfg.Mode = "bm25"
	qCfg.MaxLinked = 0
	qCfg.TitleBoost = 3.0

	results := SearchBM25(idx, "oil", qCfg)
	if len(results) < 2 {
		t.Fatalf("expected at least 2 results, got %d", len(results))
	}
	if !strings.Contains(results[0].Title, "Changing oil") {
		t.Fatalf("top result should be the 'Changing oil' chunk, got %q",
			results[0].Title)
	}
}

// TestTitleBoostScalesScore: increasing TitleBoost must strictly
// increase the score of a chunk whose TITLE contains the query term.
// Confirms the BM25F boost path is wired through scoring.
func TestTitleBoostScalesScore(t *testing.T) {
	tmp := t.TempDir()
	// Use an H2 heading so splitSections creates a section chunk
	// with the title-containing "apple" token. H1 content stays in
	// the body and doesn't exercise the title path.
	writeFile(t, filepath.Join(tmp, "a.md"),
		"# Doc\n\n## Apple pruning\n\nBody text about oranges.\n")

	cfg := DefaultIndexConfig()
	cfg.ModelKey = NopModelKey
	emb, _ := LoadEmbedder(nopModelInfo)
	defer emb.Close()
	idx := BuildIndex(ParseKnowledge(tmp, cfg), emb, cfg)

	qCfg := DefaultQueryConfig()
	qCfg.Mode = "bm25"
	qCfg.MaxLinked = 0
	qCfg.TitleBoost = 1.0 // equivalent to vanilla BM25

	boost1 := SearchBM25(idx, "apple", qCfg)
	qCfg.TitleBoost = 5.0
	boost5 := SearchBM25(idx, "apple", qCfg)

	if len(boost1) == 0 || len(boost5) == 0 {
		t.Fatal("expected non-empty results")
	}
	if boost5[0].Score <= boost1[0].Score {
		t.Errorf("boost=5 score (%g) should exceed boost=1 score (%g)",
			boost5[0].Score, boost1[0].Score)
	}
}

// ---------- frontmatter ----------

// TestParseFrontmatterBasic covers a canonical Obsidian-style block.
func TestParseFrontmatterBasic(t *testing.T) {
	text := "---\n" +
		"title: Drift detection\n" +
		"aliases: [divergence, drift]\n" +
		"tags: [runbook, ops]\n" +
		"---\n" +
		"# Header\n\nBody content.\n"

	fm, body := parseFrontmatter(text)
	if fm.Title != "Drift detection" {
		t.Errorf("title = %q, want %q", fm.Title, "Drift detection")
	}
	if len(fm.Aliases) != 2 || fm.Aliases[0] != "divergence" {
		t.Errorf("aliases = %v", fm.Aliases)
	}
	if len(fm.Tags) != 2 || fm.Tags[0] != "runbook" {
		t.Errorf("tags = %v", fm.Tags)
	}
	if !strings.HasPrefix(body, "# Header") {
		t.Errorf("body should start with '# Header', got %q", body)
	}
}

// TestParseFrontmatterAbsent: file without a leading fence is returned
// unchanged; fm is zero.
func TestParseFrontmatterAbsent(t *testing.T) {
	text := "# Plain markdown\n\nNo frontmatter here.\n"
	fm, body := parseFrontmatter(text)
	if fm.Title != "" || len(fm.Aliases) != 0 || len(fm.Tags) != 0 {
		t.Errorf("expected zero frontmatter, got %+v", fm)
	}
	if body != text {
		t.Errorf("body should be unchanged on no-frontmatter input")
	}
}

// TestParseFrontmatterMalformedFallsBack: an opening fence without a
// closing one must not crash or consume the file — just return the
// original text.
func TestParseFrontmatterMalformedFallsBack(t *testing.T) {
	text := "---\ntitle: Drift\n# forgot the closing fence\n\nbody\n"
	fm, body := parseFrontmatter(text)
	if fm.Title != "" {
		t.Errorf("expected zero fm on malformed input, got %+v", fm)
	}
	if body != text {
		t.Errorf("body should equal original on malformed input")
	}
}

// TestAliasesGetTitleBoost: a query for an alias term must retrieve
// the file's root chunk, proving aliases end up in the TitleTermFreq
// path (and so receive the title multiplier).
func TestAliasesGetTitleBoost(t *testing.T) {
	tmp := t.TempDir()
	writeFile(t, filepath.Join(tmp, "drift.md"),
		"---\n"+
			"aliases: [divergence]\n"+
			"---\n"+
			"# Drift detection\n\n"+
			"When live state stops matching declared state.\n")

	cfg := DefaultIndexConfig()
	cfg.ModelKey = NopModelKey
	emb, _ := LoadEmbedder(nopModelInfo)
	defer emb.Close()
	idx := BuildIndex(ParseKnowledge(tmp, cfg), emb, cfg)

	// Query for "divergence" — not in the body, only in the alias.
	qCfg := DefaultQueryConfig()
	qCfg.Mode = "bm25"
	qCfg.MaxLinked = 0

	results := SearchBM25(idx, "divergence", qCfg)
	if len(results) == 0 {
		t.Fatal("alias 'divergence' did not match any chunk")
	}
	if !strings.Contains(results[0].File, "drift") {
		t.Fatalf("top hit should be drift.md, got %q", results[0].File)
	}
}

// TestTagsAssignedFromFrontmatter: every chunk from a file carrying
// `tags: [X]` must have X in Chunk.Tags.
func TestTagsAssignedFromFrontmatter(t *testing.T) {
	tmp := t.TempDir()
	writeFile(t, filepath.Join(tmp, "a.md"),
		"---\n"+
			"tags: [kubernetes, runbook]\n"+
			"---\n"+
			"# Doc\n\n## One\n\nfirst\n\n## Two\n\nsecond\n")

	cfg := DefaultIndexConfig()
	chunks := ParseKnowledge(tmp, cfg)
	if len(chunks) == 0 {
		t.Fatal("no chunks parsed")
	}
	for _, c := range chunks {
		if !containsStr(c.Tags, "kubernetes") || !containsStr(c.Tags, "runbook") {
			t.Errorf("chunk %q tags=%v, expected kubernetes+runbook", c.Title, c.Tags)
		}
	}
}

// TestTagFilterExcludesMismatch: RequireTags narrows scoring to chunks
// with a matching tag; chunks without the tag are silently dropped
// even when their BM25 score would have been high.
func TestTagFilterExcludesMismatch(t *testing.T) {
	tmp := t.TempDir()
	writeFile(t, filepath.Join(tmp, "k8s.md"),
		"---\ntags: [kubernetes]\n---\n# K8s\n\nHow to scale replicas.\n")
	writeFile(t, filepath.Join(tmp, "aws.md"),
		"---\ntags: [aws]\n---\n# AWS\n\nHow to scale replicas.\n")

	cfg := DefaultIndexConfig()
	cfg.ModelKey = NopModelKey
	emb, _ := LoadEmbedder(nopModelInfo)
	defer emb.Close()
	idx := BuildIndex(ParseKnowledge(tmp, cfg), emb, cfg)

	qCfg := DefaultQueryConfig()
	qCfg.Mode = "bm25"
	qCfg.MaxLinked = 0
	qCfg.RequireTags = []string{"kubernetes"}

	results := SearchBM25(idx, "scale replicas", qCfg)
	if len(results) == 0 {
		t.Fatal("expected at least one kubernetes-tagged hit")
	}
	for _, r := range results {
		if !strings.Contains(r.File, "k8s") {
			t.Errorf("unexpected file %q in tag-filtered results — should have been excluded", r.File)
		}
	}
}

// TestTagFilterEmptyAllowsAll: empty RequireTags must not exclude
// anything. Protects against an accidental "always empty" bug.
func TestTagFilterEmptyAllowsAll(t *testing.T) {
	tmp := t.TempDir()
	writeFile(t, filepath.Join(tmp, "k8s.md"),
		"---\ntags: [kubernetes]\n---\n# K8s\n\nHow to scale.\n")
	writeFile(t, filepath.Join(tmp, "plain.md"),
		"# Plain\n\nHow to scale.\n")

	cfg := DefaultIndexConfig()
	cfg.ModelKey = NopModelKey
	emb, _ := LoadEmbedder(nopModelInfo)
	defer emb.Close()
	idx := BuildIndex(ParseKnowledge(tmp, cfg), emb, cfg)

	qCfg := DefaultQueryConfig()
	qCfg.Mode = "bm25"
	qCfg.MaxLinked = 0
	// RequireTags nil / empty: both chunks must still be reachable.
	results := SearchBM25(idx, "scale", qCfg)
	if len(results) < 2 {
		t.Fatalf("expected >=2 results with no filter, got %d", len(results))
	}
}

func containsStr(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}

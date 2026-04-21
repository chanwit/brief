// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

package engine

import (
	"path/filepath"
	"strings"
	"testing"
)

// TestExtractLinks covers the wikilink regex against common shapes seen
// in Obsidian / Logseq / Foam vaults: plain, aliased, heading-anchored,
// nested inside surrounding text, duplicates, and negative cases that
// must NOT match (bare brackets, code-fence-looking text).
func TestExtractLinks(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"No links here.", nil},
		{"See [[refresh]].", []string{"refresh"}},
		{"Both [[refresh]] and [[bridges]].", []string{"refresh", "bridges"}},
		{"Aliased: [[refresh|re-pull]] works.", []string{"refresh"}},
		{"With anchor: [[Foo#Scaling]] should target Foo.", []string{"foo"}},
		{"Combined: [[Foo#S|show]] strips both.", []string{"foo"}},
		{"Dedup: [[a]] then [[a]] then [[A]].", []string{"a"}},
		{"Single bracket [not a link].", nil},
		{"Empty [[]] is ignored.", nil},
		{"Surrounding *[[bold]]* still works.", []string{"bold"}},
	}
	for _, tc := range cases {
		got := extractLinks(tc.in)
		if len(got) != len(tc.want) {
			t.Errorf("extractLinks(%q) got %v, want %v", tc.in, got, tc.want)
			continue
		}
		for i := range got {
			if got[i] != tc.want[i] {
				t.Errorf("extractLinks(%q)[%d] = %q, want %q", tc.in, i, got[i], tc.want[i])
			}
		}
	}
}

// TestParseKnowledgeCapturesLinks: when a corpus contains wikilinks, the
// parser's output Chunks must carry them in .Links. This is the bridge
// between parsing and recall-time expansion.
func TestParseKnowledgeCapturesLinks(t *testing.T) {
	tmp := t.TempDir()
	writeFile(t, filepath.Join(tmp, "drift.md"),
		"# Drift\n\n## What is drift\n\nSee [[refresh]] and [[bridges]].\n")
	writeFile(t, filepath.Join(tmp, "refresh.md"),
		"# Refresh\n\nRecovery doc, references [[drift]] back.\n")
	writeFile(t, filepath.Join(tmp, "bridges.md"),
		"# Bridges\n\nNo links here.\n")

	cfg := DefaultIndexConfig()
	chunks := ParseKnowledge(tmp, cfg)
	if len(chunks) == 0 {
		t.Fatal("no chunks parsed")
	}

	driftLinks, refreshLinks, bridgesLinks := collectLinksByFile(chunks)
	if want := []string{"refresh", "bridges"}; !equalStringSets(driftLinks, want) {
		t.Errorf("drift.md links = %v, want %v", driftLinks, want)
	}
	if want := []string{"drift"}; !equalStringSets(refreshLinks, want) {
		t.Errorf("refresh.md links = %v, want %v", refreshLinks, want)
	}
	if len(bridgesLinks) != 0 {
		t.Errorf("bridges.md links = %v, want none", bridgesLinks)
	}
}

// TestExpandLinksSurfacesRelatedChunks: the whole point of parsing links
// is to make them accessible at recall time. Build a BM25-only index
// with a wikilink chain, run DispatchSearch, and verify LinkedFrom
// results appear.
func TestExpandLinksSurfacesRelatedChunks(t *testing.T) {
	tmp := t.TempDir()
	writeFile(t, filepath.Join(tmp, "drift.md"),
		"# Drift\n\n## What is drift\n\nDrift happens. See [[refresh]] and [[bridges]].\n")
	writeFile(t, filepath.Join(tmp, "refresh.md"),
		"# Refresh\n\nRun `cub unit refresh` to recover.\n")
	writeFile(t, filepath.Join(tmp, "bridges.md"),
		"# Bridges\n\nBridges compare live state to declared state.\n")

	cfg := DefaultIndexConfig()
	cfg.ModelKey = NopModelKey
	emb, _ := LoadEmbedder(nopModelInfo)
	defer emb.Close()
	idx := BuildIndex(ParseKnowledge(tmp, cfg), emb, cfg)

	qCfg := DefaultQueryConfig()
	qCfg.Mode = "bm25"
	qCfg.MaxLinked = 3

	results := DispatchSearch(idx, nil, "drift happens", qCfg)

	// Expected: at least one primary hit on drift.md, plus two Related
	// entries (refresh.md, bridges.md) with LinkedFrom = "drift.md".
	var linked []SearchResult
	var sawDrift bool
	for _, r := range results {
		if r.LinkedFrom != "" {
			linked = append(linked, r)
		} else if strings.Contains(r.File, "drift") {
			sawDrift = true
		}
	}
	if !sawDrift {
		t.Fatalf("no drift.md primary hit in %v", results)
	}
	if len(linked) != 2 {
		t.Fatalf("expected 2 linked results, got %d: %v", len(linked), linked)
	}
	files := map[string]bool{}
	for _, r := range linked {
		files[r.File] = true
		if r.LinkedFrom != "drift.md" {
			t.Errorf("linked result LinkedFrom=%q, want drift.md", r.LinkedFrom)
		}
	}
	if !files["refresh.md"] || !files["bridges.md"] {
		t.Errorf("expected refresh.md and bridges.md among linked, got %v", files)
	}
}

// TestExpandLinksNoOpWhenDisabled: MaxLinked=0 must short-circuit and
// return the primary hits untouched. Guards against regressions that
// would make the feature impossible to turn off.
func TestExpandLinksNoOpWhenDisabled(t *testing.T) {
	tmp := t.TempDir()
	writeFile(t, filepath.Join(tmp, "a.md"), "# A\n\nSee [[b]].\n")
	writeFile(t, filepath.Join(tmp, "b.md"), "# B\n\nContent.\n")

	cfg := DefaultIndexConfig()
	cfg.ModelKey = NopModelKey
	emb, _ := LoadEmbedder(nopModelInfo)
	defer emb.Close()
	idx := BuildIndex(ParseKnowledge(tmp, cfg), emb, cfg)

	qCfg := DefaultQueryConfig()
	qCfg.Mode = "bm25"
	qCfg.MaxLinked = 0

	results := DispatchSearch(idx, nil, "a", qCfg)
	for _, r := range results {
		if r.LinkedFrom != "" {
			t.Errorf("expected no linked results with MaxLinked=0, got %+v", r)
		}
	}
}

// ---------- helpers ----------

func collectLinksByFile(chunks []Chunk) (drift, refresh, bridges []string) {
	for _, c := range chunks {
		switch c.File {
		case "drift.md":
			drift = append(drift, c.Links...)
		case "refresh.md":
			refresh = append(refresh, c.Links...)
		case "bridges.md":
			bridges = append(bridges, c.Links...)
		}
	}
	return
}

func equalStringSets(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	m := make(map[string]bool, len(a))
	for _, x := range a {
		m[x] = true
	}
	for _, x := range b {
		if !m[x] {
			return false
		}
	}
	return true
}

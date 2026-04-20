// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

package cmd

import (
	"fmt"
	"os"
)

// Default-path resolution — optimized for the agentic-hook use case:
// running `brief recall "prompt"` at a project root should Just Work.

// indexLookupOrder is the priority list of paths brief tries when the
// user doesn't pass --index. Explicit env var > hidden state dir >
// legacy Claude Code layout > convention-named file.
var indexLookupOrder = []string{
	".brief/index.json",
	".claude/knowledge.index.json",
	"brief.index.json",
}

// knowledgeLookupOrder is the priority list of knowledge directories.
var knowledgeLookupOrder = []string{
	".claude/knowledge",
	"knowledge",
	"docs",
}

// DefaultIndexOutput is where `brief learn` writes when --output is
// omitted. Mirrors .git/, .terraform/ as a hidden project state dir.
const DefaultIndexOutput = ".brief/index.json"

// locateIndex returns the first existing index file in lookup order,
// or empty if none was found.
func locateIndex() string {
	if p := os.Getenv("BRIEF_INDEX"); p != "" {
		return p
	}
	for _, c := range indexLookupOrder {
		if fileExists(c) {
			return c
		}
	}
	return ""
}

// locateKnowledge returns the first existing knowledge dir, or empty
// if none was found.
func locateKnowledge() string {
	if p := os.Getenv("BRIEF_KNOWLEDGE"); p != "" {
		return p
	}
	for _, c := range knowledgeLookupOrder {
		if st, err := os.Stat(c); err == nil && st.IsDir() {
			return c
		}
	}
	return ""
}

// errNotFound prints a friendly lookup-order explanation to stderr and
// exits nonzero. We bypass returning an error to cobra because cobra
// would then print a second, less informative line — the structured
// guidance here is what we want the user to see.
func errNotFound(kind, envVar string, order []string) {
	fmt.Fprintf(os.Stderr, "error: no --%s specified and none found at:\n", kind)
	fmt.Fprintf(os.Stderr, "  $%s (unset)\n", envVar)
	for _, c := range order {
		fmt.Fprintf(os.Stderr, "  ./%s\n", c)
	}
	os.Exit(1)
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

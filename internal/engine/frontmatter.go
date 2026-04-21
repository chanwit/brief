// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

package engine

import (
	"strings"

	"gopkg.in/yaml.v3"
)

// frontmatter holds the subset of YAML keys brief recognizes in the
// leading --- block of a markdown file. Any other keys are ignored —
// Obsidian and Foam vaults often carry app-specific fields (created,
// updated, cssclass, …) that aren't retrieval signals.
type frontmatter struct {
	Title   string   `yaml:"title"`
	Aliases []string `yaml:"aliases"`
	Tags    []string `yaml:"tags"`
}

// parseFrontmatter looks for a leading
//
//	---
//	<yaml>
//	---
//
// block at the start of text. If found, returns the parsed struct
// plus the remaining body (after the closing fence). If absent or
// malformed, returns a zero struct and the original text untouched —
// the caller should treat the whole input as body.
func parseFrontmatter(text string) (frontmatter, string) {
	// File must begin with exactly "---" followed by a newline.
	const fence = "---"
	trimmed := text
	if !strings.HasPrefix(trimmed, fence) {
		return frontmatter{}, text
	}
	after := trimmed[len(fence):]
	if !strings.HasPrefix(after, "\n") && !strings.HasPrefix(after, "\r\n") {
		return frontmatter{}, text
	}

	// Find the closing fence — a "---" line after the opening.
	nl := strings.Index(after, "\n")
	body := after[nl+1:]
	end := strings.Index(body, "\n---")
	if end < 0 {
		return frontmatter{}, text
	}
	yamlBlock := body[:end]
	rest := body[end+len("\n---"):]
	// Skip the newline after the closing fence (if any).
	if strings.HasPrefix(rest, "\n") {
		rest = rest[1:]
	} else if strings.HasPrefix(rest, "\r\n") {
		rest = rest[2:]
	}

	var fm frontmatter
	if err := yaml.Unmarshal([]byte(yamlBlock), &fm); err != nil {
		// Malformed YAML: fall back to treating everything as body,
		// preserving the original text unchanged.
		return frontmatter{}, text
	}
	// Normalize tags to lowercase so --tag filters are case-
	// insensitive without callers having to think about it.
	for i, t := range fm.Tags {
		fm.Tags[i] = strings.ToLower(strings.TrimSpace(t))
	}
	// Aliases keep their original case for display; the tokenizer
	// lowercases at index/query time for matching.
	for i, a := range fm.Aliases {
		fm.Aliases[i] = strings.TrimSpace(a)
	}
	return fm, rest
}

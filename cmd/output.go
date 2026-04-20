// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

package cmd

import (
	"fmt"
	"os"
	"strings"

	"github.com/chanwit/brief/internal/engine"
)

// Hook-mode output caps. Keep injected context small enough to coexist
// with the user's prompt in the LLM's context window.
const (
	hookMaxCharsPerChunk = 1800
	hookMaxTotalChars    = 6000
)

// isStdoutTTY reports whether stdout is a terminal. Used to switch
// between human-readable (banner + numbered list) and hook-friendly
// (Markdown block) output without requiring a flag.
func isStdoutTTY() bool {
	fi, err := os.Stdout.Stat()
	if err != nil {
		return false
	}
	return (fi.Mode() & os.ModeCharDevice) != 0
}

// printHookOutput emits a <knowledge>...</knowledge> Markdown block
// designed to be appended to an LLM's context by a UserPromptSubmit
// hook. Truncates long bodies and caps total size.
func printHookOutput(results []engine.SearchResult) {
	if len(results) == 0 {
		return
	}
	fmt.Println("<knowledge>")
	fmt.Println("The following project knowledge was retrieved for your prompt.")
	fmt.Println("Treat it as reference material — cite it when relevant, ignore if off-topic.")
	fmt.Println()

	used := 0
	for _, r := range results {
		body := r.Body
		if len(body) > hookMaxCharsPerChunk {
			cut := strings.LastIndex(body[:hookMaxCharsPerChunk], "\n\n")
			if cut < hookMaxCharsPerChunk/2 {
				cut = hookMaxCharsPerChunk
			}
			body = body[:cut] + "\n…"
		}
		block := fmt.Sprintf("## %s — %s\n\n%s\n\n", r.File, r.Title, body)
		if used+len(block) > hookMaxTotalChars {
			break
		}
		fmt.Print(block)
		used += len(block)
	}
	fmt.Println("</knowledge>")
}

// printHumanOutput emits the interactive banner + numbered ranked list.
func printHumanOutput(idx *engine.Index, query string, cfg engine.QueryConfig, results []engine.SearchResult) {
	fmt.Printf("%s results for: %s (model=%s backend=%s)\n\n",
		cfg.Mode, query, idx.ModelInfo.Key, engine.BackendDescription(idx, cfg))
	for i, r := range results {
		switch cfg.Mode {
		case "hybrid":
			fmt.Printf("%d. [%.4f sem=%.3f bm25=%.3f] %s — %s\n",
				i+1, r.Score, r.Semantic, r.BM25, r.File, r.Title)
		default:
			fmt.Printf("%d. [%.4f] %s — %s\n", i+1, r.Score, r.File, r.Title)
		}
	}
}

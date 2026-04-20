// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

// Package cmd wires up the brief CLI using spf13/cobra. Each public
// subcommand lives in its own file; this file only carries the root
// command and the shared Execute entrypoint.
package cmd

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
	Use:   "brief",
	Short: "A memory engine for agents",
	Long: `brief learns from a corpus and recalls the right passages when your
coding agent asks. Designed for prompt-time hooks: sub-millisecond
search, auto-bootstrapping, and a Markdown output format the LLM
can read directly.

Environment:
  BRIEF_HOME       root dir (default: ~/.brief)
  BRIEF_INDEX      default index path for recall
  BRIEF_KNOWLEDGE  default knowledge dir for learn
  ORT_LIB_PATH     full path to libonnxruntime shared library
  BRIEF_MODELS_DIR where per-model subdirs live`,
	SilenceUsage: true,
}

// Execute runs the cobra tree. Called from main.main.
func Execute() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

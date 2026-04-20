// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

package cmd

import (
	"github.com/spf13/cobra"
)

var tuneCmd = &cobra.Command{
	Use:   "tune",
	Short: "Hyperparameter tuning for brief",
	Long: `Tune retrieval quality against a labeled eval set. Two subcommands:

  brief tune recall  — random-search recall-time knobs (ranking weights,
                       BM25 k1/b, relevance floors). Fast; no re-learn.
  brief tune learn   — grid-search learn-time knobs (chunk strategy,
                       size, overlap). Slow; rebuilds the index per trial.

Use 'tune recall' first — it's cheap and usually solves the recall
target alone. Fall back to 'tune learn' only if it can't.`,
}

func init() {
	rootCmd.AddCommand(tuneCmd)
}

// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

package cmd

import (
	"encoding/json"
	"fmt"

	"github.com/spf13/cobra"
	ort "github.com/yalue/onnxruntime_go"

	"github.com/chanwit/brief/internal/engine"
)

var tuneRecallFlags struct {
	indexPath string
	evalPath  string
	mode      string
	trials    int
	k         int
	objective string
	output    string
}

var tuneRecallCmd = &cobra.Command{
	Use:   "recall",
	Short: "Random-search recall-time knobs against an eval set",
	Long: `Random-searches the full recall-time configuration space and
reports the config that maximizes the chosen objective.

Objectives:
  hit_rate  is the relevant doc anywhere in top-K?  (strict correctness)
  mrr       how high did it rank?                   (rank quality)

Use hit_rate when you need a pass/fail bar; use mrr once hit_rate is
already high.`,
	RunE: runTuneRecall,
}

func runTuneRecall(cmd *cobra.Command, args []string) error {
	if tuneRecallFlags.indexPath == "" {
		tuneRecallFlags.indexPath = locateIndex()
	}
	if tuneRecallFlags.indexPath == "" {
		errNotFound("index", "BRIEF_INDEX", indexLookupOrder) // exits
	}
	if tuneRecallFlags.evalPath == "" {
		return fmt.Errorf("--eval required")
	}

	idx, err := engine.LoadIndex(tuneRecallFlags.indexPath)
	if err != nil {
		return err
	}
	defer idx.Close()
	set, err := engine.LoadEvalSet(tuneRecallFlags.evalPath)
	if err != nil {
		return err
	}

	if err := engine.EnsureSetup(idx.ModelInfo.Key); err != nil {
		return fmt.Errorf("setup: %w", err)
	}
	engine.InitORT()
	defer ort.DestroyEnvironment()
	emb, err := engine.LoadEmbedder(idx.ModelInfo)
	if err != nil {
		return err
	}
	defer emb.Close()

	best, metrics := engine.TuneQueryConfig(idx, emb, set,
		tuneRecallFlags.trials, tuneRecallFlags.mode, tuneRecallFlags.k, tuneRecallFlags.objective)
	fmt.Printf("\nBest hit@%d=%.4f MRR@%d=%.4f recall@%d=%.4f precision@%d=%.4f\n",
		tuneRecallFlags.k, metrics.HitRate,
		tuneRecallFlags.k, metrics.MRR,
		tuneRecallFlags.k, metrics.Recall,
		tuneRecallFlags.k, metrics.Precision)
	data, _ := json.MarshalIndent(best, "", "  ")
	fmt.Println(string(data))
	if tuneRecallFlags.output != "" {
		if err := engine.SaveQueryConfig(best, tuneRecallFlags.output); err != nil {
			return fmt.Errorf("save: %w", err)
		}
		fmt.Printf("Saved best QueryConfig to %s\n", tuneRecallFlags.output)
	}
	return nil
}

func init() {
	f := tuneRecallCmd.Flags()
	f.StringVar(&tuneRecallFlags.indexPath, "index", "", "path to index.json (auto-located if empty)")
	f.StringVar(&tuneRecallFlags.evalPath, "eval", "", "path to eval JSON (required)")
	f.StringVar(&tuneRecallFlags.mode, "mode", "hybrid", "hybrid | bm25 | semantic")
	f.IntVar(&tuneRecallFlags.trials, "trials", 200, "number of random-search trials")
	f.IntVarP(&tuneRecallFlags.k, "k", "k", 5, "top-k used to compute the objective")
	f.StringVar(&tuneRecallFlags.objective, "objective", "hit_rate", "hit_rate (in-top-K at all) | mrr (rank quality)")
	f.StringVarP(&tuneRecallFlags.output, "output", "o", "", "save best QueryConfig JSON here")

	tuneCmd.AddCommand(tuneRecallCmd)
}

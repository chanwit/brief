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

var tuneLearnFlags struct {
	from            string
	evalPath        string
	model           string
	trials          int
	k               int
	objective       string
	queryConfigPath string
	outConfig       string
	outIndex        string
}

var tuneLearnCmd = &cobra.Command{
	Use:   "learn",
	Short: "Grid-search learn-time knobs (chunking) against an eval set",
	Long: `Tries a small set of chunking configurations, rebuilding the index
for each, and reports the one that scores best on the eval set.

This is slower than 'brief tune recall' because each trial re-embeds
the corpus. Run it only after 'tune recall' has plateaued — chunking
changes are usually the smaller lever.`,
	RunE: runTuneLearn,
}

func runTuneLearn(cmd *cobra.Command, args []string) error {
	if tuneLearnFlags.from == "" {
		tuneLearnFlags.from = locateKnowledge()
	}
	if tuneLearnFlags.from == "" {
		errNotFound("from", "BRIEF_KNOWLEDGE", knowledgeLookupOrder) // exits
	}
	if tuneLearnFlags.evalPath == "" {
		return fmt.Errorf("--eval required")
	}

	set, err := engine.LoadEvalSet(tuneLearnFlags.evalPath)
	if err != nil {
		return err
	}

	info, err := engine.ResolveModel(tuneLearnFlags.model)
	if err != nil {
		return err
	}
	if err := engine.EnsureSetup(info.Key); err != nil {
		return fmt.Errorf("setup: %w", err)
	}
	engine.InitORT()
	defer ort.DestroyEnvironment()
	emb, err := engine.LoadEmbedder(info)
	if err != nil {
		return err
	}
	defer emb.Close()

	qCfg := engine.DefaultQueryConfig()
	if tuneLearnFlags.queryConfigPath != "" {
		c, err := engine.LoadQueryConfig(tuneLearnFlags.queryConfigPath)
		if err != nil {
			return err
		}
		qCfg = c
	}
	qCfg.K = tuneLearnFlags.k

	bestCfg, bestIdx, metrics := engine.TuneIndexConfig(
		tuneLearnFlags.from, emb, set,
		tuneLearnFlags.trials, qCfg, tuneLearnFlags.objective)
	fmt.Printf("\nBest hit@%d=%.4f MRR@%d=%.4f recall@%d=%.4f\n",
		tuneLearnFlags.k, metrics.HitRate,
		tuneLearnFlags.k, metrics.MRR,
		tuneLearnFlags.k, metrics.Recall)
	data, _ := json.MarshalIndent(bestCfg, "", "  ")
	fmt.Println(string(data))
	if tuneLearnFlags.outConfig != "" {
		if err := engine.SaveIndexConfig(bestCfg, tuneLearnFlags.outConfig); err != nil {
			return fmt.Errorf("save: %w", err)
		}
		fmt.Printf("Saved best IndexConfig to %s\n", tuneLearnFlags.outConfig)
	}
	if tuneLearnFlags.outIndex != "" {
		if err := engine.SaveIndex(bestIdx, tuneLearnFlags.outIndex); err != nil {
			return fmt.Errorf("save: %w", err)
		}
		fmt.Printf("Saved best index to %s\n", tuneLearnFlags.outIndex)
	}
	return nil
}

func init() {
	f := tuneLearnCmd.Flags()
	f.StringVar(&tuneLearnFlags.from, "from", "", "directory of source documents (auto-located if empty)")
	f.StringVar(&tuneLearnFlags.from, "knowledge", "", "alias for --from")
	_ = f.MarkHidden("knowledge")
	f.StringVar(&tuneLearnFlags.evalPath, "eval", "", "path to eval JSON (required)")
	f.StringVar(&tuneLearnFlags.model, "model", engine.DefaultModelKey, "embedding model key")
	f.IntVar(&tuneLearnFlags.trials, "trials", 8, "number of chunking configs to try")
	f.IntVarP(&tuneLearnFlags.k, "k", "k", 5, "top-k used to compute the objective")
	f.StringVar(&tuneLearnFlags.objective, "objective", "hit_rate", "hit_rate | mrr")
	f.StringVar(&tuneLearnFlags.queryConfigPath, "query-config", "", "QueryConfig JSON used when scoring candidates")
	f.StringVar(&tuneLearnFlags.outConfig, "output-config", "", "save best IndexConfig here")
	f.StringVar(&tuneLearnFlags.outIndex, "output-index", "", "save best index here")

	tuneCmd.AddCommand(tuneLearnCmd)
}

// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

package cmd

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"

	"github.com/chanwit/brief/internal/engine"
)

var initFlags struct {
	model   string
	verbose bool
}

var initCmd = &cobra.Command{
	Use:   "init",
	Short: "Download ONNX runtime and the requested embedding model",
	Long: `Fetches the ONNX runtime shared library and the chosen sentence-
transformer model into ~/.brief (or $BRIEF_HOME). Safe to run repeatedly —
exits silently when the cache is already warm.`,
	RunE: func(cmd *cobra.Command, args []string) error {
		info, err := engine.ResolveModel(initFlags.model)
		if err != nil {
			return err
		}

		// Silent when warm — hook scripts that chain
		//   brief init && brief recall
		// don't want chatter on every turn.
		warm := fileExists(engine.OrtLibPath) &&
			fileExists(filepath.Join(engine.ModelDirFor(info.Key), "model.onnx")) &&
			fileExists(filepath.Join(engine.ModelDirFor(info.Key), "tokenizer.json"))

		if err := engine.EnsureSetup(initFlags.model); err != nil {
			return fmt.Errorf("setup failed: %w", err)
		}
		if warm && !initFlags.verbose {
			return nil
		}
		fmt.Fprintf(os.Stdout, "ONNX runtime: %s\n", engine.OrtLibPath)
		fmt.Fprintf(os.Stdout, "Model:        %s\n", engine.ModelDirFor(info.Key))
		fmt.Fprintln(os.Stdout, "Setup complete.")
		return nil
	},
}

func init() {
	initCmd.Flags().StringVar(&initFlags.model, "model", engine.DefaultModelKey, "model key to download")
	initCmd.Flags().BoolVarP(&initFlags.verbose, "verbose", "v", false, "always print paths (default: silent when cache is warm)")
	rootCmd.AddCommand(initCmd)
}

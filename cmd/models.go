// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

package cmd

import (
	"fmt"

	"github.com/spf13/cobra"

	"github.com/chanwit/brief/internal/engine"
)

var modelsCmd = &cobra.Command{
	Use:   "models",
	Short: "List built-in embedding models",
	RunE: func(cmd *cobra.Command, args []string) error {
		fmt.Println("Built-in ONNX embedding models:")
		for _, k := range engine.ModelKeys() {
			m := engine.KnownModels[k]
			marker := " "
			if k == engine.DefaultModelKey {
				marker = "*"
			}
			fmt.Printf("  %s %-30s dim=%-4d pool=%-4s max_len=%-4d repo=%s\n",
				marker, k, m.Dim, m.Pooling, m.MaxLength, m.HFRepo)
		}
		fmt.Println("\n(* marks the default)")
		return nil
	},
}

func init() {
	rootCmd.AddCommand(modelsCmd)
}

// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

package main

import (
	"fmt"
	"path/filepath"
	"sort"
)

// ModelInfo is everything rag-engine needs to load and run an ONNX embedding
// model: where to fetch it, how to tokenize, how to pool the hidden state,
// and what ONNX input/output names the session should bind.
//
// It is persisted inside every index so queries can reliably rebuild the
// exact same embedding pipeline.
type ModelInfo struct {
	Key           string   `json:"key"`
	HFRepo        string   `json:"hf_repo"`
	Revision      string   `json:"revision"`
	ModelPath     string   `json:"model_path"`
	TokenizerPath string   `json:"tokenizer_path"`
	Dim           int      `json:"dim"`
	MaxLength     int      `json:"max_length"`
	Pooling       string   `json:"pooling"`   // "mean" or "cls"
	Normalize     bool     `json:"normalize"` // L2-normalize output vector
	Inputs        []string `json:"inputs"`
	Outputs       []string `json:"outputs"`
}

// knownModels is the built-in registry. Extend it by adding entries here —
// nothing else in the codebase needs to change. All entries in the default
// registry use BERT-style WordPiece tokenizers so the built-in tokenizer
// code path works for each of them.
var knownModels = map[string]ModelInfo{
	"all-MiniLM-L6-v2": {
		Key: "all-MiniLM-L6-v2", HFRepo: "Xenova/all-MiniLM-L6-v2",
		Revision: "main", ModelPath: "onnx/model.onnx", TokenizerPath: "tokenizer.json",
		Dim: 384, MaxLength: 128, Pooling: "mean", Normalize: true,
		Inputs:  []string{"input_ids", "attention_mask", "token_type_ids"},
		Outputs: []string{"last_hidden_state"},
	},
	"all-MiniLM-L12-v2": {
		Key: "all-MiniLM-L12-v2", HFRepo: "Xenova/all-MiniLM-L12-v2",
		Revision: "main", ModelPath: "onnx/model.onnx", TokenizerPath: "tokenizer.json",
		Dim: 384, MaxLength: 128, Pooling: "mean", Normalize: true,
		Inputs:  []string{"input_ids", "attention_mask", "token_type_ids"},
		Outputs: []string{"last_hidden_state"},
	},
	"multi-qa-MiniLM-L6-cos-v1": {
		Key: "multi-qa-MiniLM-L6-cos-v1", HFRepo: "Xenova/multi-qa-MiniLM-L6-cos-v1",
		Revision: "main", ModelPath: "onnx/model.onnx", TokenizerPath: "tokenizer.json",
		Dim: 384, MaxLength: 512, Pooling: "mean", Normalize: true,
		Inputs:  []string{"input_ids", "attention_mask", "token_type_ids"},
		Outputs: []string{"last_hidden_state"},
	},
	"bge-small-en-v1.5": {
		Key: "bge-small-en-v1.5", HFRepo: "Xenova/bge-small-en-v1.5",
		Revision: "main", ModelPath: "onnx/model.onnx", TokenizerPath: "tokenizer.json",
		Dim: 384, MaxLength: 512, Pooling: "cls", Normalize: true,
		Inputs:  []string{"input_ids", "attention_mask", "token_type_ids"},
		Outputs: []string{"last_hidden_state"},
	},
	"bge-base-en-v1.5": {
		Key: "bge-base-en-v1.5", HFRepo: "Xenova/bge-base-en-v1.5",
		Revision: "main", ModelPath: "onnx/model.onnx", TokenizerPath: "tokenizer.json",
		Dim: 768, MaxLength: 512, Pooling: "cls", Normalize: true,
		Inputs:  []string{"input_ids", "attention_mask", "token_type_ids"},
		Outputs: []string{"last_hidden_state"},
	},
}

const defaultModelKey = "all-MiniLM-L6-v2"

func resolveModel(key string) (ModelInfo, error) {
	if key == "" {
		key = defaultModelKey
	}
	info, ok := knownModels[key]
	if !ok {
		return ModelInfo{}, fmt.Errorf("unknown model %q (run `rag-engine models` to list)", key)
	}
	return info, nil
}

func modelKeys() []string {
	keys := make([]string, 0, len(knownModels))
	for k := range knownModels {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func modelDirFor(key string) string {
	return filepath.Join(modelsRoot, key)
}

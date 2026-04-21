// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

package engine

import (
	"fmt"
	"path/filepath"
	"sort"
)

// ModelInfo is everything brief needs to load and run an ONNX embedding
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

// KnownModels is the built-in registry. Extend it by adding entries here —
// nothing else in the codebase needs to change. All entries in the default
// registry use BERT-style WordPiece tokenizers so the built-in tokenizer
// code path works for each of them.
var KnownModels = map[string]ModelInfo{
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

const (
	DefaultModelKey = "all-MiniLM-L6-v2"

	// NopModelKey is the sentinel that tells ResolveModel to return a
	// no-embeddings ModelInfo. Used when the caller wants a BM25-only
	// index (brief learn --embedder none): no ONNX download, no
	// per-chunk embedding, a much smaller index JSON, and no semantic
	// search at query time.
	NopModelKey = "none"
)

// nopModelInfo describes the BM25-only "model" — a distinguishable
// ModelInfo that carries no embedding parameters. Dim=0 is the key
// invariant: callers gate embedding-dependent paths on HasEmbeddings.
var nopModelInfo = ModelInfo{Key: NopModelKey}

func ResolveModel(key string) (ModelInfo, error) {
	if key == "" {
		key = DefaultModelKey
	}
	if key == NopModelKey {
		return nopModelInfo, nil
	}
	info, ok := KnownModels[key]
	if !ok {
		return ModelInfo{}, fmt.Errorf("unknown model %q (run `brief models` to list)", key)
	}
	return info, nil
}

func ModelKeys() []string {
	keys := make([]string, 0, len(KnownModels))
	for k := range KnownModels {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func ModelDirFor(key string) string {
	return filepath.Join(ModelsRoot, key)
}

// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

package engine

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	ort "github.com/yalue/onnxruntime_go"
)

// Embedder is the abstract interface that turns a piece of text into a
// dense vector. Two implementations ship today:
//
//   - onnxEmbedder wraps a loaded ONNX session + tokenizer and runs a
//     sentence-transformer model locally. See LoadEmbedder.
//   - nopEmbedder is a no-op that always returns nil. Used when the
//     caller wants a BM25-only index (no ONNX setup, no downloads,
//     no embedding cost). Selected by ModelInfo.Key == NopModelKey.
//
// Additional backends (OpenAI, Cohere, Ollama, …) can satisfy this
// interface without touching the rest of the engine.
type Embedder interface {
	// Info returns the ModelInfo that describes the embedding produced
	// by this embedder. Serialized into indexes so queries can verify
	// they're using a compatible embedder.
	Info() ModelInfo
	// Embed returns a vector for the given text. May return nil on a
	// nop embedder; callers that need a vector must check.
	Embed(text string) []float32
	// Close releases any resources held by the embedder.
	Close() error
}

// HasEmbeddings reports whether an embedder actually produces vectors.
// Callers use it to gate semantic / hybrid search paths.
func HasEmbeddings(e Embedder) bool {
	return e != nil && e.Info().Dim > 0
}

func InitORT() {
	if err := InitORTSafe(); err != nil {
		fmt.Fprintf(os.Stderr, "error initializing ONNX runtime: %v\n", err)
		os.Exit(1)
	}
}

func InitORTSafe() error {
	ort.SetSharedLibraryPath(OrtLibPath)
	return ort.InitializeEnvironment()
}

// LoadEmbedder constructs the right Embedder for the given ModelInfo.
// NopModelKey yields a no-op embedder (no ONNX session loaded); any
// other key loads an ONNX session from the on-disk model dir.
func LoadEmbedder(info ModelInfo) (Embedder, error) {
	if info.Key == NopModelKey || info.Dim == 0 {
		return nopEmbedder{info: info}, nil
	}
	dir := ModelDirFor(info.Key)
	modelPath := filepath.Join(dir, "model.onnx")
	session, err := ort.NewDynamicAdvancedSession(modelPath, info.Inputs, info.Outputs, nil)
	if err != nil {
		return nil, fmt.Errorf("load model %s: %w", modelPath, err)
	}
	tok, err := loadTokenizer(filepath.Join(dir, "tokenizer.json"), info.MaxLength)
	if err != nil {
		session.Destroy()
		return nil, err
	}
	return &onnxEmbedder{info: info, session: session, tok: tok}, nil
}

// onnxEmbedder runs a local sentence-transformer via onnxruntime_go.
type onnxEmbedder struct {
	info    ModelInfo
	session *ort.DynamicAdvancedSession
	tok     *Tokenizer
}

func (e *onnxEmbedder) Info() ModelInfo { return e.info }

func (e *onnxEmbedder) Close() error {
	if e != nil && e.session != nil {
		e.session.Destroy()
	}
	return nil
}

func (e *onnxEmbedder) Embed(text string) []float32 {
	return runEmbed(e.session, e.tok, &e.info, text)
}

// nopEmbedder is a zero-cost Embedder that produces no vectors. An
// index built through it relies entirely on BM25 at recall time.
type nopEmbedder struct{ info ModelInfo }

func (n nopEmbedder) Info() ModelInfo         { return n.info }
func (n nopEmbedder) Embed(text string) []float32 { return nil }
func (n nopEmbedder) Close() error            { return nil }

// Tokenizer is a minimal BERT-style WordPiece tokenizer loaded from a
// HuggingFace tokenizer.json file.
type Tokenizer struct {
	Vocab  map[string]int64
	MaxLen int
	ClsID  int64
	SepID  int64
	PadID  int64
	UnkID  int64
}

func loadTokenizer(path string, maxLenOverride int) (*Tokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer %s: %w", path, err)
	}
	var raw struct {
		Model struct {
			Vocab map[string]int `json:"vocab"`
		} `json:"model"`
		Truncation *struct {
			MaxLength int `json:"max_length"`
		} `json:"truncation"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("parse tokenizer %s: %w", path, err)
	}

	vocab := make(map[string]int64, len(raw.Model.Vocab))
	for k, v := range raw.Model.Vocab {
		vocab[k] = int64(v)
	}

	maxLen := 128
	if raw.Truncation != nil && raw.Truncation.MaxLength > 0 {
		maxLen = raw.Truncation.MaxLength
	}
	if maxLenOverride > 0 {
		maxLen = maxLenOverride
	}

	return &Tokenizer{
		Vocab:  vocab,
		MaxLen: maxLen,
		ClsID:  vocab["[CLS]"],
		SepID:  vocab["[SEP]"],
		PadID:  vocab["[PAD]"],
		UnkID:  vocab["[UNK]"],
	}, nil
}

func (t *Tokenizer) Encode(text string) (inputIDs, attentionMask, tokenTypeIDs []int64) {
	text = strings.ToLower(text)
	words := tokenize(text)

	tokens := []int64{t.ClsID}
	for _, word := range words {
		tokens = append(tokens, t.wordPiece(word)...)
	}

	maxTokens := t.MaxLen - 1
	if len(tokens) > maxTokens {
		tokens = tokens[:maxTokens]
	}
	tokens = append(tokens, t.SepID)

	seqLen := len(tokens)
	inputIDs = make([]int64, t.MaxLen)
	attentionMask = make([]int64, t.MaxLen)
	tokenTypeIDs = make([]int64, t.MaxLen)

	copy(inputIDs, tokens)
	for i := 0; i < seqLen; i++ {
		attentionMask[i] = 1
	}
	for i := seqLen; i < t.MaxLen; i++ {
		inputIDs[i] = t.PadID
	}
	return inputIDs, attentionMask, tokenTypeIDs
}

func (t *Tokenizer) wordPiece(word string) []int64 {
	if _, ok := t.Vocab[word]; ok {
		return []int64{t.Vocab[word]}
	}
	var tokens []int64
	start := 0
	for start < len(word) {
		end := len(word)
		found := false
		for end > start {
			substr := word[start:end]
			if start > 0 {
				substr = "##" + substr
			}
			if id, ok := t.Vocab[substr]; ok {
				tokens = append(tokens, id)
				found = true
				start = end
				break
			}
			end--
		}
		if !found {
			tokens = append(tokens, t.UnkID)
			start++
		}
	}
	return tokens
}

func tokenize(text string) []string {
	var words []string
	var current []rune
	for _, r := range text {
		if r == ' ' || r == '\t' || r == '\n' || r == '\r' {
			if len(current) > 0 {
				words = append(words, string(current))
				current = nil
			}
		} else if isPunct(r) {
			if len(current) > 0 {
				words = append(words, string(current))
				current = nil
			}
			words = append(words, string(r))
		} else {
			current = append(current, r)
		}
	}
	if len(current) > 0 {
		words = append(words, string(current))
	}
	return words
}

func isPunct(r rune) bool {
	return (r >= '!' && r <= '/') || (r >= ':' && r <= '@') ||
		(r >= '[' && r <= '`') || (r >= '{' && r <= '~')
}

func runEmbed(session *ort.DynamicAdvancedSession, tok *Tokenizer, info *ModelInfo, text string) []float32 {
	inputIDs, attentionMask, tokenTypeIDs := tok.Encode(text)
	seqLen := int64(tok.MaxLen)
	dim := int64(info.Dim)

	inputShape := ort.Shape{1, seqLen}

	idsTensor, _ := ort.NewTensor(inputShape, inputIDs)
	defer idsTensor.Destroy()
	maskTensor, _ := ort.NewTensor(inputShape, attentionMask)
	defer maskTensor.Destroy()

	tensors := []ort.Value{idsTensor, maskTensor}
	if containsString(info.Inputs, "token_type_ids") {
		typeTensor, _ := ort.NewTensor(inputShape, tokenTypeIDs)
		defer typeTensor.Destroy()
		tensors = append(tensors, typeTensor)
	}

	outputShape := ort.Shape{1, seqLen, dim}
	outputData := make([]float32, seqLen*dim)
	outTensor, _ := ort.NewTensor(outputShape, outputData)
	defer outTensor.Destroy()

	if err := session.Run(tensors, []ort.Value{outTensor}); err != nil {
		fmt.Fprintf(os.Stderr, "error running model: %v\n", err)
		os.Exit(1)
	}

	hidden := outTensor.GetData()
	vec := poolHidden(hidden, attentionMask, int(seqLen), int(dim), info.Pooling)
	// Always L2-normalize so that searches can treat cosine similarity as a
	// plain dot product. ModelInfo.Normalize is retained for documentation
	// but is effectively always true at the search layer.
	l2Normalize(vec)
	return vec
}

func poolHidden(hidden []float32, mask []int64, seqLen, dim int, strategy string) []float32 {
	vec := make([]float32, dim)
	switch strategy {
	case "cls":
		copy(vec, hidden[:dim])
	default: // "mean"
		count := float32(0)
		for i := 0; i < seqLen; i++ {
			if mask[i] == 0 {
				continue
			}
			count++
			offset := i * dim
			for d := 0; d < dim; d++ {
				vec[d] += hidden[offset+d]
			}
		}
		if count > 0 {
			for d := 0; d < dim; d++ {
				vec[d] /= count
			}
		}
	}
	return vec
}

func l2Normalize(vec []float32) {
	var norm float32
	for _, v := range vec {
		norm += v * v
	}
	norm = float32(math.Sqrt(float64(norm)))
	if norm > 0 {
		for d := range vec {
			vec[d] /= norm
		}
	}
}

func containsString(xs []string, s string) bool {
	for _, x := range xs {
		if x == s {
			return true
		}
	}
	return false
}

// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

package main

import (
	"archive/tar"
	"compress/gzip"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

// httpClient has a generous but finite timeout so that a stalled download
// can't hang the process indefinitely. Model files can be 80-300 MB, so the
// ceiling is intentionally high for slow links.
var httpClient = &http.Client{Timeout: 10 * time.Minute}

const userAgent = "rag-engine/" + ortVersion

func httpGet(url string) (*http.Response, error) {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", userAgent)
	return httpClient.Do(req)
}

const ortVersion = "1.22.0"

// Directory layout under $RAG_HOME (default: ~/.rag-engine):
//   lib/libonnxruntime.{so,dylib}
//   models/<model-key>/{model.onnx,tokenizer.json}
//
// Env overrides:
//   RAG_HOME         - overall root
//   ORT_LIB_PATH     - full path to the ONNX runtime shared library
//   RAG_MODELS_DIR   - full path to the models directory
var (
	ragHome     string
	ortLibPath  string
	modelsRoot  string
)

func init() {
	home, _ := os.UserHomeDir()
	ragHome = filepath.Join(home, ".rag-engine")
	if v := os.Getenv("RAG_HOME"); v != "" {
		ragHome = v
	}

	libName := "libonnxruntime.so"
	if runtime.GOOS == "darwin" {
		libName = "libonnxruntime.dylib"
	}
	ortLibPath = filepath.Join(ragHome, "lib", libName)
	if v := os.Getenv("ORT_LIB_PATH"); v != "" {
		ortLibPath = v
	}

	modelsRoot = filepath.Join(ragHome, "models")
	if v := os.Getenv("RAG_MODELS_DIR"); v != "" {
		modelsRoot = v
	}
}

// ensureSetup makes sure the ONNX runtime library and the requested model
// are available locally, downloading each on cache miss. Safe to call many
// times.
func ensureSetup(modelKey string) error {
	if err := ensureORTLibrary(); err != nil {
		return fmt.Errorf("onnx runtime library: %w", err)
	}
	info, err := resolveModel(modelKey)
	if err != nil {
		return err
	}
	if err := ensureModel(info); err != nil {
		return fmt.Errorf("model %s: %w", info.Key, err)
	}
	return nil
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func ensureORTLibrary() error {
	if fileExists(ortLibPath) {
		return nil
	}
	url, err := ortDownloadURL()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(ortLibPath), 0o755); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "rag-engine: downloading ONNX runtime %s for %s/%s\n",
		ortVersion, runtime.GOOS, runtime.GOARCH)
	return downloadAndExtractORT(url, ortLibPath)
}

func ortDownloadURL() (string, error) {
	var asset string
	switch runtime.GOOS + "/" + runtime.GOARCH {
	case "linux/amd64":
		asset = "onnxruntime-linux-x64"
	case "linux/arm64":
		asset = "onnxruntime-linux-aarch64"
	case "darwin/amd64":
		asset = "onnxruntime-osx-x86_64"
	case "darwin/arm64":
		asset = "onnxruntime-osx-arm64"
	default:
		return "", fmt.Errorf("unsupported platform %s/%s", runtime.GOOS, runtime.GOARCH)
	}
	return fmt.Sprintf(
		"https://github.com/microsoft/onnxruntime/releases/download/v%s/%s-%s.tgz",
		ortVersion, asset, ortVersion,
	), nil
}

func downloadAndExtractORT(url, destPath string) error {
	resp, err := httpGet(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("HTTP %d from %s", resp.StatusCode, url)
	}
	gz, err := gzip.NewReader(resp.Body)
	if err != nil {
		return err
	}
	defer gz.Close()
	tr := tar.NewReader(gz)

	suffix := ".so"
	if runtime.GOOS == "darwin" {
		suffix = ".dylib"
	}

	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		if hdr.Typeflag != tar.TypeReg {
			continue
		}
		base := filepath.Base(hdr.Name)
		if !strings.HasPrefix(base, "libonnxruntime.") || !strings.Contains(base, suffix) {
			continue
		}
		f, err := os.OpenFile(destPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o755)
		if err != nil {
			return err
		}
		if _, err := io.Copy(f, tr); err != nil {
			f.Close()
			return err
		}
		return f.Close()
	}
	return fmt.Errorf("no libonnxruntime%s found in %s", suffix, url)
}

func ensureModel(info ModelInfo) error {
	dir := modelDirFor(info.Key)
	wantModel := filepath.Join(dir, "model.onnx")
	wantTok := filepath.Join(dir, "tokenizer.json")
	if fileExists(wantModel) && fileExists(wantTok) {
		return nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	if !fileExists(wantModel) {
		url := fmt.Sprintf("https://huggingface.co/%s/resolve/%s/%s",
			info.HFRepo, info.Revision, info.ModelPath)
		if err := downloadToFile(url, wantModel); err != nil {
			return err
		}
	}
	if !fileExists(wantTok) {
		url := fmt.Sprintf("https://huggingface.co/%s/resolve/%s/%s",
			info.HFRepo, info.Revision, info.TokenizerPath)
		if err := downloadToFile(url, wantTok); err != nil {
			return err
		}
	}
	return nil
}

func downloadToFile(url, dest string) error {
	fmt.Fprintf(os.Stderr, "rag-engine: downloading %s\n", url)
	resp, err := httpGet(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("HTTP %d from %s", resp.StatusCode, url)
	}
	tmp := dest + ".part"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if _, err := io.Copy(f, resp.Body); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, dest)
}

// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

package ivf

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
)

// on-disk layout under <dir>/:
//
//   manifest.json    — dim, k, nprobe, ntotal, counts[]
//   centroids.bin    — K * Dim * float32 (little-endian)
//   invlists.ids     — concatenated uint64 ids for each invlist, in order
//   invlists.vecs    — concatenated float32 vectors, same order
//
// ids[invlist c] are packed sequentially; counts[c] tells how many. Same
// for vecs with element size Dim*4.
//
// All three big files are raw binary so future versions can mmap them
// directly without a codec step.

type manifest struct {
	SchemaVersion int    `json:"schema_version"`
	Dim           int    `json:"dim"`
	K             int    `json:"k"`
	Nprobe        int    `json:"nprobe"`
	Ntotal        int    `json:"ntotal"`
	Counts        []int  `json:"counts"`
}

const ivfSchemaVersion = 1

// Save writes the index to dir. Overwrites existing files.
func (ix *IVFFlat) Save(dir string) error {
	if !ix.trained {
		return fmt.Errorf("ivf.Save: index not trained")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	counts := make([]int, ix.K)
	for c := 0; c < ix.K; c++ {
		counts[c] = len(ix.invlists[c].ids)
	}
	m := manifest{
		SchemaVersion: ivfSchemaVersion,
		Dim:           ix.Dim,
		K:             ix.K,
		Nprobe:        ix.Nprobe,
		Ntotal:        ix.ntotal,
		Counts:        counts,
	}
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), data, 0o644); err != nil {
		return err
	}
	if err := writeFloat32Flat(filepath.Join(dir, "centroids.bin"), ix.centroids); err != nil {
		return err
	}
	return ix.writeInvlists(dir)
}

// Load reads an index from dir. Loads everything into memory; mmap support
// can be added later without changing the on-disk format.
func Load(dir string) (*IVFFlat, error) {
	data, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return nil, err
	}
	var m manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("ivf.Load: parse manifest: %w", err)
	}
	if m.SchemaVersion != ivfSchemaVersion {
		return nil, fmt.Errorf("ivf.Load: schema v%d, this binary understands v%d",
			m.SchemaVersion, ivfSchemaVersion)
	}
	if len(m.Counts) != m.K {
		return nil, fmt.Errorf("ivf.Load: manifest counts len %d != K %d",
			len(m.Counts), m.K)
	}

	ix := New(m.Dim, m.K)
	ix.Nprobe = m.Nprobe
	ix.ntotal = m.Ntotal

	cent, err := readFloat32Flat(filepath.Join(dir, "centroids.bin"), m.K*m.Dim)
	if err != nil {
		return nil, err
	}
	ix.centroids = cent
	ix.trained = true

	return ix, ix.readInvlists(dir, m.Counts)
}

func (ix *IVFFlat) writeInvlists(dir string) error {
	ids, err := os.Create(filepath.Join(dir, "invlists.ids"))
	if err != nil {
		return err
	}
	defer ids.Close()
	vecs, err := os.Create(filepath.Join(dir, "invlists.vecs"))
	if err != nil {
		return err
	}
	defer vecs.Close()

	buf := make([]byte, 8)
	for c := 0; c < ix.K; c++ {
		il := &ix.invlists[c]
		for _, id := range il.ids {
			binary.LittleEndian.PutUint64(buf, id)
			if _, err := ids.Write(buf); err != nil {
				return err
			}
		}
		if len(il.vectors) > 0 {
			if err := writeFloat32Stream(vecs, il.vectors); err != nil {
				return err
			}
		}
	}
	return nil
}

func (ix *IVFFlat) readInvlists(dir string, counts []int) error {
	ids, err := os.Open(filepath.Join(dir, "invlists.ids"))
	if err != nil {
		return err
	}
	defer ids.Close()
	vecs, err := os.Open(filepath.Join(dir, "invlists.vecs"))
	if err != nil {
		return err
	}
	defer vecs.Close()

	buf := make([]byte, 8)
	for c, n := range counts {
		if n == 0 {
			continue
		}
		il := &ix.invlists[c]
		il.ids = make([]uint64, n)
		for i := 0; i < n; i++ {
			if _, err := io.ReadFull(ids, buf); err != nil {
				return fmt.Errorf("read ids for invlist %d: %w", c, err)
			}
			il.ids[i] = binary.LittleEndian.Uint64(buf)
		}
		il.vectors = make([]float32, n*ix.Dim)
		if err := readFloat32Stream(vecs, il.vectors); err != nil {
			return fmt.Errorf("read vecs for invlist %d: %w", c, err)
		}
	}
	return nil
}

// writeFloat32Flat writes a float32 slice to a fresh file as little-endian
// binary, directly addressable via mmap without any codec.
func writeFloat32Flat(path string, v []float32) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return writeFloat32Stream(f, v)
}

func writeFloat32Stream(w io.Writer, v []float32) error {
	const batch = 4096
	buf := make([]byte, batch*4)
	for start := 0; start < len(v); start += batch {
		end := start + batch
		if end > len(v) {
			end = len(v)
		}
		for i, x := range v[start:end] {
			binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(x))
		}
		if _, err := w.Write(buf[:(end-start)*4]); err != nil {
			return err
		}
	}
	return nil
}

func readFloat32Flat(path string, n int) ([]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	out := make([]float32, n)
	return out, readFloat32Stream(f, out)
}

func readFloat32Stream(r io.Reader, out []float32) error {
	const batch = 4096
	buf := make([]byte, batch*4)
	for start := 0; start < len(out); start += batch {
		end := start + batch
		if end > len(out) {
			end = len(out)
		}
		if _, err := io.ReadFull(r, buf[:(end-start)*4]); err != nil {
			return err
		}
		for i := start; i < end; i++ {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[(i-start)*4:]))
		}
	}
	return nil
}

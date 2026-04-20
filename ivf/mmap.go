// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

//go:build (linux || darwin) && (amd64 || arm64)

package ivf

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"unsafe"

	"golang.org/x/sys/unix"
)

// Open memory-maps an IVF index directory and returns an IVFFlat ready for
// search. No vector data is loaded into heap memory; the OS page cache
// serves reads on demand, which means:
//
//   - process RSS stays tiny even for 100M-vector indexes
//   - multiple processes sharing the same index share one copy in RAM
//   - startup is O(read manifest) — the big files are just mapped
//
// Call Close() to unmap. While the returned index is live, the underlying
// files must not be truncated or overwritten.
//
// Build-constrained to linux/darwin on amd64/arm64 because the fast path
// reinterprets mmap'd bytes as []float32 / []uint64 in native byte order,
// which is only correct on little-endian hosts. All our release targets
// qualify; other OS/arch combinations fall back to Load().
func Open(dir string) (*IVFFlat, error) {
	data, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return nil, fmt.Errorf("ivf.Open: read manifest: %w", err)
	}
	var m manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("ivf.Open: parse manifest: %w", err)
	}
	if m.SchemaVersion != ivfSchemaVersion {
		return nil, fmt.Errorf("ivf.Open: schema v%d, this binary understands v%d",
			m.SchemaVersion, ivfSchemaVersion)
	}
	if len(m.Counts) != m.K {
		return nil, fmt.Errorf("ivf.Open: manifest counts len %d != K %d",
			len(m.Counts), m.K)
	}

	ix := New(m.Dim, m.K)
	ix.Nprobe = m.Nprobe
	ix.ntotal = m.Ntotal
	ix.trained = true

	// Centroids are tiny (K*Dim*4 bytes, e.g. 4096*384*4 = 6 MB) but mmap
	// them anyway for layout symmetry and shared-memory semantics.
	centroidsMap, err := mmapRead(filepath.Join(dir, "centroids.bin"))
	if err != nil {
		return nil, fmt.Errorf("ivf.Open: centroids: %w", err)
	}
	expectedBytes := m.K * m.Dim * 4
	if len(centroidsMap.data) < expectedBytes {
		centroidsMap.Close()
		return nil, fmt.Errorf("ivf.Open: centroids.bin is %d bytes, expected ≥%d",
			len(centroidsMap.data), expectedBytes)
	}
	ix.centroids = bytesAsFloat32(centroidsMap.data)[:m.K*m.Dim]
	ix.mmapHandles = append(ix.mmapHandles, centroidsMap)

	// Invlists: one file for IDs, one for vectors. We mmap each whole file
	// once and slice zero-copy views per posting list.
	idsMap, err := mmapRead(filepath.Join(dir, "invlists.ids"))
	if err != nil {
		ix.Close()
		return nil, fmt.Errorf("ivf.Open: invlists.ids: %w", err)
	}
	ix.mmapHandles = append(ix.mmapHandles, idsMap)

	vecsMap, err := mmapRead(filepath.Join(dir, "invlists.vecs"))
	if err != nil {
		ix.Close()
		return nil, fmt.Errorf("ivf.Open: invlists.vecs: %w", err)
	}
	ix.mmapHandles = append(ix.mmapHandles, vecsMap)

	allIDs := bytesAsUint64(idsMap.data)
	allVecs := bytesAsFloat32(vecsMap.data)

	var idOff, vecOff int
	for c, n := range m.Counts {
		il := &ix.invlists[c]
		if n == 0 {
			continue
		}
		if idOff+n > len(allIDs) || vecOff+n*m.Dim > len(allVecs) {
			ix.Close()
			return nil, fmt.Errorf("ivf.Open: invlist %d: file truncated", c)
		}
		il.ids = allIDs[idOff : idOff+n]
		il.vectors = allVecs[vecOff : vecOff+n*m.Dim]
		idOff += n
		vecOff += n * m.Dim
	}
	return ix, nil
}

// mmapFile owns one mmap region and the file descriptor that produced it.
// Close unmaps and closes the fd in that order.
type mmapFile struct {
	data []byte
	f    *os.File
}

var _ io.Closer = (*mmapFile)(nil)

func mmapRead(path string) (*mmapFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	st, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	size := int(st.Size())
	if size == 0 {
		return &mmapFile{f: f}, nil
	}
	data, err := unix.Mmap(int(f.Fd()), 0, size, unix.PROT_READ, unix.MAP_SHARED)
	if err != nil {
		f.Close()
		return nil, err
	}
	// Hint "random access": probes touch a handful of posting lists per
	// query, not sequential scans. The kernel uses this to avoid
	// read-ahead prefetch. Best-effort; ignore errors.
	_ = unix.Madvise(data, unix.MADV_RANDOM)
	return &mmapFile{data: data, f: f}, nil
}

func (m *mmapFile) Close() error {
	var em, ef error
	if len(m.data) > 0 {
		em = unix.Munmap(m.data)
		m.data = nil
	}
	if m.f != nil {
		ef = m.f.Close()
		m.f = nil
	}
	if em != nil {
		return em
	}
	return ef
}

// bytesAsFloat32 reinterprets a little-endian mmap'd byte region as a
// float32 slice without copying. The returned slice aliases the caller's
// mmap; the mmap must outlive every use of the returned slice, which the
// IVFFlat struct guarantees by keeping the mmapFile in mmapHandles.
func bytesAsFloat32(b []byte) []float32 {
	if len(b) < 4 {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), len(b)/4)
}

func bytesAsUint64(b []byte) []uint64 {
	if len(b) < 8 {
		return nil
	}
	return unsafe.Slice((*uint64)(unsafe.Pointer(&b[0])), len(b)/8)
}

// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

//go:build !((linux || darwin) && (amd64 || arm64))

package ivf

// Open falls back to Load on platforms where mmap with zero-copy byte
// reinterpretation isn't supported (non-POSIX or big-endian hosts). Semantics
// remain identical to Load — the vectors just live on the heap instead of
// being page-faulted in on demand.
func Open(dir string) (*IVFFlat, error) {
	return Load(dir)
}

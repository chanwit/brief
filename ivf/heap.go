// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

package ivf

// topKHeap maintains the k highest-scoring Hits seen so far in a min-heap
// keyed on Score. Push is O(log k); final extraction is O(k log k). A
// manual heap beats container/heap here because the interface dispatch
// shows up in profiles at millions of Push calls per query.
type topKHeap struct {
	cap  int
	data []Hit
}

func newTopKHeap(cap int) *topKHeap {
	return &topKHeap{cap: cap, data: make([]Hit, 0, cap)}
}

// Push offers a candidate. If the heap isn't full it's always inserted;
// otherwise it replaces the current minimum iff its score is higher.
func (h *topKHeap) Push(item Hit) {
	if len(h.data) < h.cap {
		h.data = append(h.data, item)
		h.siftUp(len(h.data) - 1)
		return
	}
	if item.Score > h.data[0].Score {
		h.data[0] = item
		h.siftDown(0)
	}
}

// SortedDesc returns the accumulated items sorted by Score descending.
// Destructive — the heap is unusable afterwards.
func (h *topKHeap) SortedDesc() []Hit {
	out := make([]Hit, 0, len(h.data))
	for len(h.data) > 0 {
		out = append(out, h.data[0])
		last := len(h.data) - 1
		h.data[0] = h.data[last]
		h.data = h.data[:last]
		if len(h.data) > 0 {
			h.siftDown(0)
		}
	}
	// out is ascending (min popped first); reverse for descending.
	for i, j := 0, len(out)-1; i < j; i, j = i+1, j-1 {
		out[i], out[j] = out[j], out[i]
	}
	return out
}

func (h *topKHeap) siftUp(i int) {
	for i > 0 {
		p := (i - 1) / 2
		if h.data[p].Score <= h.data[i].Score {
			break
		}
		h.data[p], h.data[i] = h.data[i], h.data[p]
		i = p
	}
}

func (h *topKHeap) siftDown(i int) {
	n := len(h.data)
	for {
		l := 2*i + 1
		if l >= n {
			return
		}
		s := l
		r := l + 1
		if r < n && h.data[r].Score < h.data[l].Score {
			s = r
		}
		if h.data[i].Score <= h.data[s].Score {
			return
		}
		h.data[i], h.data[s] = h.data[s], h.data[i]
		i = s
	}
}

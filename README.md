# brief

A memory engine for agents. `brief` learns from your project's
knowledge and recalls the right passages when your coding agent asks.
Drop-in context injection for Claude Code, Cursor, or anything with a
prompt-time hook. Local ONNX embeddings, hybrid BM25 + semantic
ranking, tunable to 100% hit rate on your own corpus.

[![CI](https://github.com/chanwit/brief/actions/workflows/ci.yml/badge.svg)](https://github.com/chanwit/brief/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/chanwit/brief.svg)](https://pkg.go.dev/github.com/chanwit/brief)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Contents

- [Why brief](#why-brief)
- [Use case: Claude Code hooks](#use-case-claude-code-hooks)
- [Install](#install)
- [Quick start](#quick-start)
- [Commands](#commands)
- [Configuration](#configuration)
- [Hyperparameter tuning](#hyperparameter-tuning)
- [Performance](#performance)
- [Design](#design)
- [Environment](#environment)
- [Development](#development)
- [License](#license)

---

## Why brief

### Built for agentic hooks

Coding agents like Claude Code and Cursor fire a prompt-submit hook on
every user turn. That hook has a tight latency budget and needs to
inject relevant project knowledge before the model thinks. `brief` is
optimized for that exact shape of workload: a short-lived process per
turn, microsecond search, millisecond embedding, and a Markdown output
format the LLM can read directly.

### Fast enough to run on every turn

| Stage                         | Typical latency |
|-------------------------------|-----------------|
| Query embedding (ONNX)        | ~12 ms          |
| Semantic search via IVF-Flat  | ~6 µs           |
| Full hybrid recall, end-to-end | **~15 ms**     |

The IVF hot path runs through hand-written **AVX2** (amd64) and
**NEON** (arm64) kernels — 8.6–9.6× faster than scalar Go on the
dot-product primitive. Hook budgets are usually 50–200 ms; `brief`
fits with headroom to spare.

### Self-contained, batteries included

- **One static binary.** No Python, no Docker, no vector database, no
  API keys, no network calls at query time.
- **Auto-bootstraps.** First invocation downloads the ONNX runtime and
  the requested embedding model into `~/.brief/` (~150 MB). Every
  subsequent run is fully offline.
- **No sidecar.** Indexes are portable files — a JSON for BM25 and
  metadata, plus a directory of mmap-ready binaries for the IVF-Flat
  ANN companion. Copy them anywhere, commit them, ship them in a
  container.
- **Five models built in.** MiniLM L6/L12, BGE small/base, multi-qa
  MiniLM. `brief models` lists them; `--model KEY` switches.
  Adding a new ONNX model is one struct literal.

### UX that respects your time

- **One command to the answer.** `brief recall "prompt"` returns
  ranked Markdown chunks ready to drop into context.
- **Reproducible indexes.** The model, chunking strategy, and every
  build parameter are serialized into the index — queries can't
  silently cross-run a different model's embeddings.
- **Two search backends, one interface.** Flat brute-force for small
  corpora, mmap'd IVF for scale. Dispatch is automatic; the CLI
  doesn't change.
- **`--json` for scripting**, Markdown-block for hooks, human-readable
  for terminals. `brief` detects the output target automatically.
- **Tuned defaults that work.** On a realistic 18-query eval set
  against technical docs, defaults score **1.0 hit@5** and
  **1.0 MRR**. Drop in, don't fiddle.

### Hyperparameter tuning for your corpus

Defaults are strong for English technical prose. For anything else,
point the tuner at a small labeled eval set and it random-searches the
full knob space until your target metric is met:

```sh
brief tune recall --index .brief/index.json --eval eval.json \
    --objective hit_rate --trials 300 --output best-recall.json
```

Two objectives are first-class:

- **`hit_rate`** — strict correctness: is the right doc in top-K at
  all? Use this when you need a pass/fail quality bar.
- **`mrr`** — rank quality: how high did it land? Use this once
  hit-rate is already solved.

Both **recall-time tuning** (fast, no re-learn — ranking weights, BM25
params, relevance floors) and **learn-time tuning** (slower; chunking
strategy, overlap, length caps) are built in.

### CI-tested on four arches

`{linux, darwin} × {amd64, arm64}` on every push and pull request.
Releases ship native tarballs per runner with a `SHA256SUMS` manifest.

---

## Use case: Claude Code hooks

Claude Code fires the `UserPromptSubmit` hook on every turn. The hook
receives the user's prompt, and anything it writes to stdout becomes
additional context for that turn. `brief` was built to be exactly that
binary.

### Wire it up in one line

In `.claude/settings.local.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [{
      "hooks": [{
        "type": "command",
        "command": "brief recall \"$CLAUDE_USER_PROMPT\""
      }]
    }]
  }
}
```

No flags. `brief recall` auto-locates the index in the current working
directory (tries `$BRIEF_INDEX`, then `./.brief/index.json`, then
`./.claude/knowledge.index.json`, then `./brief.index.json`).

### Build the knowledge index once

```sh
brief learn
```

Also no flags. `brief learn` auto-locates the knowledge directory
(`$BRIEF_KNOWLEDGE` → `./.claude/knowledge` → `./knowledge` → `./docs`)
and writes to `./.brief/index.json` by default. IVF is auto-enabled
once the corpus crosses 5 000 chunks; smaller corpora stay on the
brute-force path.

Every turn, Claude Code pipes the user's prompt through
`brief recall`. When stdout is a pipe (hook context) `brief` emits a
`<knowledge>...</knowledge>` Markdown block, ranked and truncated to
fit the agent's context budget. When stdout is a terminal, it prints
a human-readable banner with scores.

### Replace existing Python RAG hooks

Most RAG hook templates shipped with coding agents today are Python
scripts doing TF-IDF cosine in-process, which limits them to tiny
corpora and crude ranking. `brief` is a drop-in upgrade:

- Same stdin/stdout contract.
- Hybrid BM25 + dense-embedding ranking with a relevance gate that
  suppresses off-topic matches.
- IVF-Flat ANN so retrieval stays in the microseconds as your
  knowledge base grows.

---

## Install

### From a release

Download the tarball for your OS and arch from
[Releases](https://github.com/chanwit/brief/releases), extract, and
move the binary onto your `PATH`:

```sh
tar -xzf brief-<version>-<os>-<arch>.tar.gz
mv brief-<version>-<os>-<arch>/brief /usr/local/bin/
```

First invocation downloads the ONNX runtime (~60 MB) and the default
embedding model (~85 MB) into `~/.brief/`. Subsequent runs are
offline.

### From source

Requires Go 1.25 or newer and a C toolchain (the ONNX runtime binding
uses cgo, which means cross-compilation is not supported — build on
the target OS/arch).

```sh
git clone https://github.com/chanwit/brief
cd brief
make build         # produces ./brief
make test          # full test suite; ~15 s on a warm cache
```

---

## Quick start

Zero-config path — drop Markdown into `./.claude/knowledge` (or
`./knowledge` / `./docs`) and run:

```sh
brief learn                             # auto everything
brief recall "how do I rotate credentials"
```

Or be explicit:

```sh
# 1. Learn from a directory of Markdown or text files.
brief learn --from ./docs --output docs.index.json

# 2. Recall passages.
brief recall --index docs.index.json "how do I rotate credentials"

# 3. (Optional) Tune retrieval knobs for your corpus.
brief tune recall \
    --index docs.index.json --eval eval.json \
    --trials 300 --output best-recall.json

# 4. Use the tuned config.
brief recall \
    --index  docs.index.json \
    --config best-recall.json \
    "how do I rotate credentials"
```

IVF-Flat ANN kicks in automatically once your corpus crosses 5 000
chunks; pass `--no-ivf` to force flat search or `--use-ivf` to force
IVF on smaller corpora.

---

## Commands

Every command accepts `-h` for its flag list.

| Command         | Purpose                                                        |
|-----------------|----------------------------------------------------------------|
| `init`          | Download ONNX runtime and the requested embedding model.       |
| `models`        | List built-in embedding models.                                |
| `learn`         | Learn from a directory of documents (builds the index).        |
| `recall`        | Recall relevant passages for a question (queries the index).   |
| `tune learn`    | Grid-search learn-time knobs (chunking) against an eval set.   |
| `tune recall`   | Random-search recall-time knobs against an eval set.           |

### `learn`

Builds a JSON index with BM25 statistics and embedding vectors. With
`--use-ivf` (or auto-enabled above `--auto-ivf-threshold`), vectors
move to a sibling `<index>.ivf/` directory stored as raw float32
invlists and mmap'd at recall time.

```
brief learn \
  [ --from DIR ]                        # auto: $BRIEF_KNOWLEDGE → .claude/knowledge → knowledge → docs
  [ --output PATH ]                     # auto: ./.brief/index.json
  [ --model KEY ]                       # default all-MiniLM-L6-v2
  [ --config INDEXCONFIG.json ]         # JSON config; flags override
  [ --chunk-strategy heading|size ]
  [ --chunk-size N ] [ --chunk-overlap N ]
  [ --min-chunk-chars N ] [ --max-chunk-chars N ]
  [ --include  "*.md,*.txt" ]
  [ --exclude  "drafts/*" ]
  [ --use-ivf | --no-ivf ]              # default: auto at chunks ≥ --auto-ivf-threshold (5000)
  [ --ivf-centroids K ] [ --ivf-nprobe N ]
```

### `recall`

```
brief recall \
  [ --index PATH ]                      # auto: $BRIEF_INDEX → .brief/index.json → .claude/knowledge.index.json → brief.index.json
  [ --config QUERYCONFIG.json ]
  [ --mode hybrid|bm25|semantic ]
  [ -k N ]                              # default 3
  [ --weight-semantic F ] [ --weight-bm25 F ]
  [ --bm25-k1 F ] [ --bm25-b F ]
  [ --semantic-hard-floor F ] [ --semantic-soft-floor F ]
  [ --bm25-min-for-soft-zone F ]
  [ --nprobe N ] [ --n-semantic N ]     # IVF only
  [ --json ]
    "your question"
```

Output format is TTY-aware: when stdout is a terminal you get a
human-readable banner and ranked list; when stdout is a pipe (hook
context) you get a `<knowledge>...</knowledge>` Markdown block
designed to be read by an LLM. `--json` overrides both.

Modes:

- `hybrid` (default): combines normalized BM25 and cosine similarity,
  then applies the relevance gate.
- `bm25`: lexical only. No embedding model loaded.
- `semantic`: cosine similarity only. No gating.

The **relevance gate** (hybrid mode) rejects candidates in three
tiers:

1. `semantic < hard_floor` — always drop.
2. `semantic < soft_floor` — drop unless `bm25_norm ≥ soft_zone` (BM25
   must corroborate).
3. `semantic ≥ soft_floor` — always keep.

This turns off-topic queries into clean empty results instead of
low-confidence noise.

---

## Configuration

Both `learn` and `recall` accept `--config FILE` with JSON matching
the structs below. CLI flags override values from the file.

### IndexConfig

```json
{
  "model": "all-MiniLM-L6-v2",
  "chunk_strategy": "heading",
  "chunk_size": 500,
  "chunk_overlap": 100,
  "min_chunk_chars": 0,
  "max_chunk_chars": 0,
  "embed_max_chars": 1500,
  "include": ["*.md", "*.markdown", "*.txt"],
  "exclude": null,

  "use_ivf": false,
  "ivf_centroids": 0,
  "ivf_nprobe": 0
}
```

| Field             | Meaning                                                |
|-------------------|--------------------------------------------------------|
| `model`           | Model key from the built-in registry.                  |
| `chunk_strategy`  | `heading` splits on `## ` (markdown-aware); `size` produces fixed-size overlapping chunks. |
| `chunk_size`      | Target chars per chunk (size strategy).                |
| `chunk_overlap`   | Char overlap between adjacent chunks.                  |
| `min_chunk_chars` | Drop chunks shorter than this.                         |
| `max_chunk_chars` | Truncate chunk bodies (0 = unbounded).                 |
| `embed_max_chars` | Truncate text before tokenization.                     |
| `include`/`exclude` | File globs against basename and relative path.       |
| `use_ivf`         | Build an IVF-Flat companion index.                     |
| `ivf_centroids`   | IVF `K`. `0` = auto (`4 · √N`, min 16).                |
| `ivf_nprobe`      | Default nprobe stored in the IVF manifest. `0` = `√K`. |

### QueryConfig

Tuned defaults; they hit `1.0 hit@5` and `1.0 MRR` on the reference
technical-docs eval set. Override for other domains via
`brief tune recall` or CLI flags.

```json
{
  "mode": "hybrid",
  "k": 3,
  "weight_semantic": 0.48,
  "weight_bm25": 0.52,
  "bm25_k1": 2.33,
  "bm25_b": 0.97,
  "semantic_hard_floor": 0.35,
  "semantic_soft_floor": 0.43,
  "bm25_min_for_soft_zone": 0.44,
  "min_query_terms_in_corpus": 0,

  "nprobe": 0,
  "n_semantic": 0
}
```

| Field                        | Meaning                                              |
|------------------------------|------------------------------------------------------|
| `mode`                       | `hybrid`, `bm25`, or `semantic`.                     |
| `k`                          | Top-k results.                                       |
| `weight_semantic` / `weight_bm25` | Hybrid combination weights.                     |
| `bm25_k1`                    | BM25 term-frequency saturation. Canonical 1.2.       |
| `bm25_b`                     | BM25 length normalization. Canonical 0.75.           |
| `semantic_hard_floor`        | Cosine threshold for always-reject.                  |
| `semantic_soft_floor`        | Cosine threshold for BM25-corroborated keep.         |
| `bm25_min_for_soft_zone`     | BM25 floor used inside the soft zone.                |
| `min_query_terms_in_corpus`  | Minimum query tokens that must appear in the corpus. `0` = auto. |
| `nprobe`                     | IVF probe count at recall time. `0` = index default. |
| `n_semantic`                 | IVF shortlist size fed into hybrid re-ranking. `0` = `max(k·20, 100)`. |

---

## Hyperparameter tuning

The tuner takes an eval set of labeled queries and optimizes retrieval
quality. Two objectives are supported:

- **`hit_rate`** — "is the relevant doc anywhere in top-K?" Targets
  strict correctness; optimize this for a pass/fail bar like "100% of
  known queries must return their right answer".
- **`mrr`** — "how high does it rank?" Targets a good user experience
  even when hit-rate is already high.

### Eval set format

```json
{
  "queries": [
    {
      "query": "how do I rotate credentials",
      "relevant_files":  ["security/credentials.md"],
      "relevant_titles": ["Rotating credentials"]
    }
  ]
}
```

A result counts as relevant if its `file` matches any entry in
`relevant_files` (equality or suffix) or its `title` matches any entry
in `relevant_titles`.

### Workflow

1. Write 10–50 queries covering the ways your users actually ask.
2. Build a baseline index (`brief learn`).
3. Tune recall-time knobs first — fast, no re-learn:
   ```sh
   brief tune recall --index .brief/index.json --eval eval.json \
       --objective hit_rate --trials 300 --output best-recall.json
   ```
4. If recall is still short, tune chunking (slower; rebuilds the index
   per trial):
   ```sh
   brief tune learn --from ./docs --eval eval.json \
       --query-config best-recall.json --trials 12 \
       --output-config best-learn.json --output-index tuned.json
   ```
5. Ship `best-recall.json` and the tuned index.

---

## Performance

All numbers from a 13th-gen Intel i5-1335U (12 threads, AVX2).

### SIMD dot product

The IVF search hot path routes through a hand-written assembly kernel
selected at init.

| Dim | Scalar Go | AVX2 kernel | Speedup |
|-----|-----------|-------------|---------|
| 384 | 150 ns    | 17 ns       | **8.6×** |
| 768 | 321 ns    | 33 ns       | **9.6×** |

Correctness is cross-validated against the scalar fallback on 21
length classes that stress every tail path (`TestDotMatchesGeneric`).
The arm64 NEON kernel shares that test on macOS and Linux CI runners.

### Search latency (187-chunk technical-docs corpus)

| Backend        | Mode     | Latency | Note                                                |
|----------------|----------|---------|-----------------------------------------------------|
| Flat           | BM25     |  63 µs  | Pure lexical; no ONNX.                              |
| Flat           | Semantic |  60 µs  | Brute-force cosine (AVX2).                          |
| **IVF**        | Semantic |  **5.6 µs** | **10.7× faster** than flat semantic at K=32/nprobe=8. |
| Flat           | Hybrid   |  93 µs  | BM25 + semantic + combine.                          |
| IVF            | Hybrid   |  78 µs  | BM25 is now the bottleneck.                         |
| *query embed*  | *ONNX*   | *12.5 ms* | *Dominates end-to-end CLI latency.*                 |

At small corpora the embedding forward pass is the overall bottleneck.
At 10k+ chunks IVF decisively outperforms flat semantic; the gap
widens roughly linearly with corpus size.

### Retrieval quality

On the scenarios eval set (18 queries × 187 chunks) with tuned
defaults:

| Backend | hit@5     | MRR      |
|---------|-----------|----------|
| Flat    | **1.000** | **1.000**|
| IVF     | **1.000** | **1.000**|

IVF matches flat exactly on this corpus — no approximation penalty
for the 10× semantic speedup.

---

## Design

### Model registry and reproducibility

Every known model has a `ModelInfo` record covering its HuggingFace
repo, file paths, tokenizer, dimension, pooling strategy, and ONNX
input/output names. Indexes embed the full `ModelInfo` and schema
version. `loadIndex` refuses to open a newer schema and refuses to
attach an IVF companion whose manifest disagrees with the index.

### Chunking

Two strategies: heading-aware markdown splitting that respects fenced
code blocks, and fixed-size windowing with overlap. The chunker is
one dimension of the tuner.

### Hybrid ranking with a relevance gate

Normalized BM25 is combined linearly with cosine similarity. A
two-tier semantic floor rejects low-confidence noise before top-K
truncation — the gate must be applied *before* sorting, otherwise a
cluster of high-BM25 but low-semantic hits can evict every eligible
candidate. `TestHybridGateAppliesBeforeTopK` locks that invariant.

### IVF-Flat ANN

- K-means++ coarse quantizer trained on the corpus vectors.
- Invlists stored as raw little-endian float32 for zero-copy mmap.
- `Open(dir)` on `(linux || darwin) × (amd64 || arm64)` memory-maps
  centroids and invlists and reinterprets the bytes as `[]float32` /
  `[]uint64` without copying. Other platforms fall back to an
  in-memory `Load`.
- Recall-time dispatch (`dispatchSearch`) picks flat vs IVF based on
  whether the index has an attached companion — the CLI, the tuner,
  and the tests all go through the same selector.

### SIMD kernels

- **amd64**: AVX2 with two 8-float accumulators to break FMA latency,
  plus YMM/XMM tails and a scalar residue.
- **arm64**: NEON with two 4-float accumulators, a 4-wide tail, and a
  scalar residue. Unconditional install — NEON is in the ARMv8-A
  baseline.
- **other**: pure-Go fallback validated by the same tests.

### Storage layout

```
<index>.json              # JSON: schema, model info, config, chunks, BM25
<index>.json.ivf/         # IVF companion (only if use_ivf=true)
├── manifest.json
├── centroids.bin         # K * Dim * float32
├── invlists.ids          # concatenated uint64 chunk IDs
└── invlists.vecs         # concatenated float32 vectors
```

`~/.brief/` caches the ONNX runtime and per-model directories.
Overridable via `BRIEF_HOME`, `ORT_LIB_PATH`, `BRIEF_MODELS_DIR`.

---

## Environment

| Variable            | Default                                   | Purpose                                 |
|---------------------|-------------------------------------------|-----------------------------------------|
| `BRIEF_HOME`        | `~/.brief`                                | Root for downloaded artifacts.          |
| `BRIEF_INDEX`       | *(auto-located)*                          | Default index path for `recall`.        |
| `BRIEF_KNOWLEDGE`   | *(auto-located)*                          | Default knowledge dir for `learn`.      |
| `ORT_LIB_PATH`      | `$BRIEF_HOME/lib/libonnxruntime.{so,dylib}` | ONNX runtime library path.            |
| `BRIEF_MODELS_DIR`  | `$BRIEF_HOME/models`                      | Per-model cache directory.              |
| `BRIEF_PERF_CORPUS` | *(auto-discovered)*                       | Override for performance test corpus.   |

---

## Development

```sh
make build    # build ./brief
make test     # run every test (auto-downloads alt model on cold cache)
make vet      # go vet
make dist     # package dist/brief-<ver>-<os>-<arch>.tar.gz
make clean
```

### Benchmarks

```sh
go test -run='^$' -bench=Dot       ./ivf/   # SIMD kernels
go test -run='^$' -bench=Scenarios ./       # end-to-end search
```

### CI

A 4-arch matrix runs on every push and pull request:

| Runner             | GOOS/GOARCH     | SIMD kernel |
|--------------------|-----------------|-------------|
| `ubuntu-latest`    | `linux/amd64`   | AVX2        |
| `ubuntu-24.04-arm` | `linux/arm64`   | NEON        |
| `macos-13`         | `darwin/amd64`  | AVX2        |
| `macos-latest`     | `darwin/arm64`  | NEON        |

Tagged releases build native tarballs on each runner and attach them
to a GitHub Release with a `SHA256SUMS` manifest.

---

## License

[MIT](LICENSE)

# rag-engine

A dataset-agnostic Go-native RAG index and query tool. Uses local ONNX
sentence-transformer models plus BM25, with hybrid ranking, a relevance gate,
and built-in hyperparameter tuning against an eval set.

- **Single binary.** No Python, no services, no vector database.
- **Auto-bootstrapping.** First use downloads the ONNX runtime and the
  requested model into `~/.rag-engine/`.
- **Multi-model.** Five built-in BERT-style models; drop-in custom models by
  extending the registry.
- **Reproducible indexes.** The embedding model and all build-time parameters
  are embedded in the index file. Queries can't silently cross-use a
  different model.
- **Fully parameterized.** Every knob that affects retrieval (chunking,
  BM25 `k1`/`b`, hybrid weights, relevance floors) is exposed on the CLI and
  in JSON config files.
- **Tunable.** `tune-query` random-searches query-time knobs against an eval
  set; `tune-index` grid-walks chunking configs. Both emit JSON you can
  feed back via `--config`.

---

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Supported models](#supported-models)
- [Commands](#commands)
- [Configuration files](#configuration-files)
- [Eval set format](#eval-set-format)
- [Tuning guide](#tuning-guide)
- [Environment variables](#environment-variables)
- [Storage layout](#storage-layout)
- [Building from source](#building-from-source)

---

## Install

Download a prebuilt tarball for your OS/arch from the GitHub release page and
extract the `rag-engine` binary onto your `PATH`.

```sh
tar -xzf rag-engine-<version>-<os>-<arch>.tar.gz
mv rag-engine-<version>-<os>-<arch>/rag-engine /usr/local/bin/
```

The first time you run a command that needs a model, rag-engine will
download the ONNX runtime shared library and the model into
`~/.rag-engine/`. No manual setup is required — but you can pre-warm the
cache explicitly:

```sh
rag-engine setup                        # default model
rag-engine setup --model bge-small-en-v1.5
```

---

## Quick start

```sh
# 1. Index a directory of markdown.
rag-engine index \
    --knowledge ./docs \
    --output    ./docs.index.json

# 2. Query it.
rag-engine query --index ./docs.index.json -k 5 "how do I rotate credentials"

# 3. (Optional) Tune query-time knobs against a small eval set.
rag-engine tune-query \
    --index  ./docs.index.json \
    --eval   ./eval.json \
    --trials 200 \
    --output ./best-query.json

# 4. Use the tuned config.
rag-engine query \
    --index  ./docs.index.json \
    --config ./best-query.json \
    "how do I rotate credentials"
```

---

## Supported models

Run `rag-engine models` to list the built-in registry. All models are
BERT-style ONNX exports with WordPiece tokenizers.

| Key                          | Dim | Pool | Max len | Notes                          |
|------------------------------|-----|------|---------|--------------------------------|
| `all-MiniLM-L6-v2` (default) | 384 | mean | 128     | Small, fast, good baseline.    |
| `all-MiniLM-L12-v2`          | 384 | mean | 128     | Slightly more accurate.        |
| `multi-qa-MiniLM-L6-cos-v1`  | 384 | mean | 512     | Tuned for Q&A-style queries.   |
| `bge-small-en-v1.5`          | 384 | cls  | 512     | Strong English retrieval.      |
| `bge-base-en-v1.5`           | 768 | cls  | 512     | Higher quality, 2× vector size.|

Model files are sourced from Hugging Face (`Xenova/<model>`).

To add a new model, append a `ModelInfo` entry to the `knownModels` map in
`models.go` and rebuild. Any WordPiece-tokenized ONNX embedding model that
exports `last_hidden_state` as its output will work.

---

## Commands

All commands accept `-h` / `--help` to print their flag list.

### `setup`

Download the ONNX runtime and a model. Safe to run many times.

```
rag-engine setup [--model KEY]
```

### `models`

List every known model with its dim, pooling, and max length.

### `index`

Build an index from a directory of source documents.

```
rag-engine index \
    --knowledge DIR \
    --output    PATH \
  [ --model KEY ] \
  [ --config  INDEXCONFIG.json ] \
  [ --chunk-strategy {heading|size} ] \
  [ --chunk-size N ] \
  [ --chunk-overlap N ] \
  [ --min-chunk-chars N ] \
  [ --max-chunk-chars N ] \
  [ --embed-max-chars N ] \
  [ --include  "*.md,*.txt" ] \
  [ --exclude  "drafts/*" ] \
  [ --pooling  {mean|cls} ]
```

- `--chunk-strategy heading` (default): split markdown on `## ` headings.
  Fenced code blocks are preserved intact.
- `--chunk-strategy size`: produce overlapping fixed-size chunks. Use this
  for plain text or any non-markdown corpus.
- `--include` / `--exclude`: comma-separated glob patterns matched against
  both basename and relative path.
- CLI flags override values loaded from `--config`.

### `query`

Run a query against an index.

```
rag-engine query \
    --index PATH \
  [ --config QUERYCONFIG.json ] \
  [ --mode {hybrid|bm25|semantic} ] \
  [ -k N ] \
  [ --weight-semantic F ] \
  [ --weight-bm25 F ] \
  [ --bm25-k1 F ] \
  [ --bm25-b F ] \
  [ --semantic-hard-floor F ] \
  [ --semantic-soft-floor F ] \
  [ --bm25-min-for-soft-zone F ] \
  [ --min-query-terms N ] \
  [ --json ] \
    "your query string"
```

Modes:

- `hybrid` (default): combines semantic cosine and BM25 with configurable
  weights, then applies the two-tier relevance floor below.
- `bm25`: classic BM25 only. No embedding model is loaded.
- `semantic`: pure cosine similarity on embeddings. No floor gating.

The query-time **relevance gate** (hybrid mode) rejects top-k candidates
that fall into:

1. `semantic < semantic_hard_floor` — always drop.
2. `semantic_hard_floor ≤ semantic < semantic_soft_floor` — drop unless
   `bm25_norm ≥ bm25_min_for_soft_zone` (BM25 must corroborate).
3. `semantic ≥ semantic_soft_floor` — always keep.

This lets queries that are off-topic for the corpus return an empty list
instead of low-quality top-k noise.

### `tune-query`

Random-search all query-time knobs against an eval set; emit the best
`QueryConfig` as JSON.

```
rag-engine tune-query \
    --index PATH \
    --eval  EVAL.json \
  [ --mode {hybrid|bm25|semantic} ] \
  [ --k N ] \
  [ --trials N ] \
  [ --output best-query.json ]
```

Objective: **MRR@k** over the eval set (see
[Eval set format](#eval-set-format) for the input schema).

The tuner pre-embeds every eval query once and then re-scores candidates
against the fixed chunk vectors, so hundreds of trials take seconds.

### `tune-index`

Grid-walk a small space of chunking configs, rebuilding the index for each
and evaluating the same way as `tune-query`. Slow because every trial
re-embeds the corpus; keep `--trials` small (default 8).

```
rag-engine tune-index \
    --knowledge DIR \
    --eval      EVAL.json \
  [ --model KEY ] \
  [ --trials N ] \
  [ --k N ] \
  [ --query-config QUERYCONFIG.json ] \
  [ --output-config best-index.json ] \
  [ --output-index  tuned.index.json ]
```

Varies: `chunk_strategy ∈ {heading, size}`, `chunk_size ∈ {250,500,750,1000,1500}`,
`chunk_overlap ∈ {0,50,100,200}`, `embed_max_chars ∈ {1000,1500,2500}`.

The chunking config that produces the best MRR@k is saved alongside (or as)
a ready-to-use tuned index.

---

## Configuration files

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
  "pooling": "",
  "normalize": null
}
```

| Field             | Meaning                                                |
|-------------------|--------------------------------------------------------|
| `model`           | Model key from the built-in registry.                  |
| `chunk_strategy`  | `heading` or `size`.                                   |
| `chunk_size`      | Target chars per chunk (size strategy).                |
| `chunk_overlap`   | Char overlap between adjacent chunks.                  |
| `min_chunk_chars` | Drop chunks shorter than this (0 = keep all).          |
| `max_chunk_chars` | Truncate chunk bodies to this length (0 = unbounded).  |
| `embed_max_chars` | Truncate text before tokenization.                     |
| `include`         | File globs to index (basename or relative path).       |
| `exclude`         | File globs to skip.                                    |
| `pooling`         | Override model default (`mean` or `cls`).              |
| `normalize`       | Override model default L2-normalization (`true`/`false`). |

### QueryConfig

```json
{
  "mode": "hybrid",
  "k": 5,
  "weight_semantic": 0.48,
  "weight_bm25": 0.52,
  "bm25_k1": 2.33,
  "bm25_b": 0.97,
  "semantic_hard_floor": 0.35,
  "semantic_soft_floor": 0.43,
  "bm25_min_for_soft_zone": 0.44,
  "min_query_terms_in_corpus": 0
}
```

These defaults came out of running `tune-query --objective hit_rate` on an
18-query eval set against a ~200-chunk technical-docs corpus: the config
above scored 1.0 hit@5 (every query's relevant document in top-5) and 1.0
MRR (every one at rank 1). The BM25 parameters drifted upward from the
canonical `k1=1.2 / b=0.75` because the target corpus had short,
term-dense sections; other domains may want to re-tune. The semantic
floors are also deliberately tight to reject low-cosine noise earlier.

To return to the textbook-BM25 defaults for a different corpus:

    rag-engine query \
        --bm25-k1 1.2 --bm25-b 0.75 \
        --semantic-hard-floor 0.2 --semantic-soft-floor 0.3 \
        --bm25-min-for-soft-zone 0.3 \
        --index idx.json "your query"

| Field                        | Meaning                                              |
|------------------------------|------------------------------------------------------|
| `mode`                       | `hybrid`, `bm25`, or `semantic`.                     |
| `k`                          | Top-k results.                                       |
| `weight_semantic`            | Hybrid weight on cosine similarity.                  |
| `weight_bm25`                | Hybrid weight on BM25 (normalized).                  |
| `bm25_k1`                    | BM25 term-frequency saturation (classic default 1.2). |
| `bm25_b`                     | BM25 length normalization (classic default 0.75).    |
| `semantic_hard_floor`        | Below this cosine, always reject.                    |
| `semantic_soft_floor`        | Between floors, require BM25 corroboration.          |
| `bm25_min_for_soft_zone`     | BM25-norm floor used inside the soft zone.           |
| `min_query_terms_in_corpus`  | Minimum query tokens that must hit the corpus (0 = auto: 1 for short queries, 2 for 3+ tokens). |

---

## Eval set format

```json
{
  "queries": [
    {
      "query": "how do I rotate credentials",
      "relevant_files":  ["security/credentials.md"],
      "relevant_titles": ["Rotating credentials"]
    },
    {
      "query": "how often should oil be changed",
      "relevant_files": ["cars.md"]
    }
  ]
}
```

Each query must specify at least one of `relevant_files` or
`relevant_titles`. A result is counted relevant if either:

- its `file` equals one of `relevant_files` (case-insensitive), *or*
- its `file` has one of `relevant_files` as a suffix (so you can write a
  basename and have it match a sub-directory path), *or*
- its `title` equals one of `relevant_titles`.

Metrics computed:

- **MRR@k** — mean over queries of `1/rank` of the first relevant hit.
- **Recall@k** — fraction of relevant items retrieved (set-valued, capped at 1.0).
- **Precision@k** — fraction of top-k that are relevant.

The tuner optimizes **MRR@k**.

---

## Tuning guide

For best results on a new dataset:

1. **Write a small eval set.** Ten to fifty queries with the filenames or
   section titles you'd expect to come back. More is better but even a
   handful is enough to pick signal from noise.

2. **Build a baseline index** with defaults:

       rag-engine index --knowledge ./docs --output ./docs.index.json

3. **Tune query-time knobs first** — it's fast and the biggest lever:

       rag-engine tune-query \
           --index  ./docs.index.json \
           --eval   ./eval.json \
           --trials 300 \
           --output ./best-query.json

4. **Optionally tune chunking.** Slow (re-embeds per trial) but worth it for
   long documents or non-markdown corpora:

       rag-engine tune-index \
           --knowledge    ./docs \
           --eval         ./eval.json \
           --query-config ./best-query.json \
           --trials       8 \
           --output-config ./best-index.json \
           --output-index  ./docs.tuned.index.json

5. **Use the tuned configs in production:**

       rag-engine query \
           --index  ./docs.tuned.index.json \
           --config ./best-query.json \
           "your question"

Tips:

- Keep **k** the same across tuning and production; otherwise the tuned
  floors may be miscalibrated.
- If your domain uses specialized vocabulary (code, acronyms, identifiers),
  try `bge-small-en-v1.5` or `bge-base-en-v1.5` — they usually beat MiniLM
  on out-of-distribution English.
- For very short corpora, disable the relevance gate by passing
  `--semantic-hard-floor 0 --semantic-soft-floor 0 --bm25-min-for-soft-zone 0`.
- `bm25` mode needs no embedding model, so it's useful for grep-like
  keyword lookups on large corpora.

---

## Environment variables

| Variable         | Default                                   | Purpose                               |
|------------------|-------------------------------------------|---------------------------------------|
| `RAG_HOME`       | `~/.rag-engine`                           | Root for downloaded artifacts.        |
| `ORT_LIB_PATH`   | `$RAG_HOME/lib/libonnxruntime.{so,dylib}` | Override the ONNX runtime library.    |
| `RAG_MODELS_DIR` | `$RAG_HOME/models`                        | Override where model dirs are stored. |

---

## Storage layout

```
$RAG_HOME/
├── lib/
│   └── libonnxruntime.so      # or libonnxruntime.dylib on macOS
└── models/
    ├── all-MiniLM-L6-v2/
    │   ├── model.onnx
    │   └── tokenizer.json
    └── bge-small-en-v1.5/
        ├── model.onnx
        └── tokenizer.json
```

Each index JSON is fully self-describing: it carries the full `ModelInfo`
and `IndexConfig` that produced it, plus a schema version. Queries load the
model identified by the index, so an index built with `bge-small-en-v1.5`
cannot accidentally be queried using a different model's embeddings.

---

## SIMD kernels

The IVF semantic-search hot path (`ivf.Dot`) uses hand-written assembly
kernels that dispatch by GOARCH at init:

| GOARCH  | Kernel                        | Speedup vs scalar Go |
|---------|-------------------------------|----------------------|
| `amd64` | AVX2 (`ivf/distance_amd64.s`) | 8.5–9.6× @ dim 384 / 768 |
| `arm64` | NEON (`ivf/distance_arm64.s`) | exercised in CI on macOS Apple Silicon and Linux arm64 |
| other   | pure-Go fallback (`dotGeneric`) | 1× (baseline) |

Correctness is cross-validated against `dotGeneric` on every length class
that stresses the tail paths (`TestDotMatchesGeneric`), and CI runs this
test on all four of {linux,darwin} × {amd64,arm64}.

## Building from source

Requirements: Go 1.25+, a C toolchain (cgo is used by the ONNX runtime
binding). Cross-compilation is not supported — build on the target OS/arch.

```sh
git clone https://github.com/chanwit/rag-engine
cd rag-engine
make build     # produces ./rag-engine
make test      # runs the full test suite (downloads alt model on cold cache)
make dist      # packages dist/rag-engine-<ver>-<os>-<arch>.tar.gz
```

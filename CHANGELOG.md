# Changelog

Every published version, newest first.  This file is on the publish
allow-list, so it travels with the package: it is the only thing a
consumer deciding whether to upgrade can read.

## 0.1.0 — 2026-09-27

The first implementation of the interface published as 0.0.1: the four
poolings, normalisation and Matryoshka truncation, the three
similarities with top-k and exact search, and int8 and binary
quantisation.

### Changed, breaking

- `int8_dot` and `int8_cosine` answer `EmbDimMismatch` for two code
  lists of different lengths, where the interface's comment named
  `EmbBitsMismatch`.  `EmbBitsMismatch` stays the fault for binary
  vectors.
- ndarray-nv `^0.1.4` in place of the `0.0.1` interface.

### Behaviour the interface left open

- A mask entry that is not zero keeps its row; the mean divides by the
  number of kept rows.
- `cls_pool` answers `EmbEmptyMask` for a matrix with no rows.
  `pooling_of_name` reads sentence-transformers' `lasttoken` as `last`.
- `pool_batch` answers `EmbMaskMismatch` with the number of masks and
  the batch size when the two differ.
- `is_right_padded` is true for a mask that keeps every token or none.
- Normalisation, similarity and quantisation refuse a coordinate that
  is not finite with `EmbNotFinite` at its position, counted row-major
  over a matrix.  `top_k` refuses a score that is not finite.
- `mean_energy_in_prefix` over a matrix with no rows answers
  `EmbZeroVector`, as a zero row does.
- The int8 rule is a scale of `(hi - lo) / 254`, a zero point of
  `round(-lo / scale) - 127`, codes of `round(x / scale) + zero` clamped
  to -127..127, and rounding half away from zero.
- `calibrate` over a matrix with no coordinates answers
  `EmbBadRange(0, 0)`.
- `to_binary` refuses a vector of no coordinates as `EmbZeroVector`.
  `hamming` reads only the first `dim` bits and refuses a vector with
  fewer words than its dimension needs.
- `binary_bytes` is one byte per eight coordinates, rounded up.
- `of_array` maps every `NdFault` variant, as its comment lists.

### Tests

- 21 tests in four suites: hand-worked arithmetic, every refusal, and
  a comparison with numpy over generated inputs.  `tools/gen_vectors.py`
  follows sentence-transformers' `Pooling`, `SimilarityFunction` and
  `quantize_embeddings` where they define the operation.
- Every line under `src/` is executed by the suites;
  `bash tests/coverage.sh` prints the number.

## 0.0.2 — 2026-09-15

README rewritten to the package README style guide (docs/writing-a-readme.md); no change to the interface.

## 0.0.1 — 2026-09-11

The **interface**, before anyone implements it.  Every signature, every
type and every effect row is published; every body is `todo()`, and the
release is stamped `NOT IMPLEMENTED — interface only`.  Adding this
package works and calling it panics.

- Five modules over one dependency.  `embpool` is pooling, `embnorm`
  unit length and Matryoshka truncation, `embsim` similarity and
  top-k, `embquant` int8 and binary, and `embfault` the fault.
- **The mask is not optional.**  `mean_pool` divides by the mask sum
  and never by the row count, and `last_pool` takes the last UNMASKED
  row and never the last row.  Both are the bugs that look right, both
  are invisible at the call site, and each has its own function so
  neither can be written by accident.  `last_index` and
  `is_right_padded` are what a caller checks its padding convention
  with.
- **Higher is closer**, which is the opposite of hnsw-nv's rule, and
  each package states its own once: an index compares by minimum, a
  ranking sorts by maximum.  `EmbEuclidean`'s score is the NEGATED
  distance, as sentence-transformers does it, so `top_k` is one
  function; `embsim.as_distance` is the single line between the two
  conventions.
- **Nothing normalises for the caller.**  A function that did would
  hand back a vector nobody stored, and a pipeline that wanted
  magnitude would find it gone.  `is_normalized` is what the package
  says instead.
- **A prefix of a unit vector is not a unit vector.**
  `truncate_normalized` is truncate plus renormalise and is the one to
  reach for; `energy_in_prefix` is a measurement over the caller's own
  corpus in place of a claim about which models truncate well.
- **The int8 calibration is corpus-wide and the per-vector form does
  not exist.**  A per-vector scale makes every stored vector
  individually accurate and mutually meaningless, and the failure is
  silent — so `calibrate` measures a range, `to_int8` takes one, and
  `int8_dot` answers `EmbScaleMismatch` rather than a plausible
  number.  A coordinate outside the range is clamped, because one
  unusual document should not fail an ingestion.
- **Binary is a sign bit**, packed 64 to a word with the dimension
  carried so the padding is not counted, and `hamming_score` puts the
  distance on `[-1, 1]` so a binary shortlist and a float re-rank
  share a threshold.  At 768 dimensions the ladder is 6,144 bytes,
  768 and 96.
- `ndarray-nv ^0.0.1` is the one dependency, and it is exact: under
  the caret rule a `0.0.x` has no compatible neighbours, which is the
  interface tier working as intended.
- No `@tier(embedded)` claim — ndarray-nv makes none either, and a
  probe is built against a dependency's modules as well as its own.
- `tests/` holds eleven API tests across two files, every one red.
  The pooling fixture's numbers are chosen so that the right answer
  and the two classic wrong ones are three different numbers;
  sentence-transformers over the same matrices becomes a generated run
  beside them when the bodies land.

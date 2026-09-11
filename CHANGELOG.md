# Changelog

Every published version, newest first.  This file is on the publish
allow-list, so it travels with the package: it is the only thing a
consumer deciding whether to upgrade can read.

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

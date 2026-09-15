# embeddings-nv

An embedding is a list of numbers that stands for a piece of text, so that
two texts about the same thing sit close together. A transformer produces
one vector per token; a vector store wants one vector per document,
normalised, sometimes shortened, sometimes a byte or a bit per coordinate.
This package is the arithmetic between those two, in novo-lang. The
reference implementation is
[sentence-transformers](https://github.com/UKPLab/sentence-transformers),
and the names here are its names. It is built on
[ndarray-nv](https://novo-lang.org/packages/ndarray-nv), whose `NdFloat` is
the matrix type every call takes.

**Status: NOT IMPLEMENTED — interface only.** Every function is declared
with its full signature, but every body is a `todo()` that panics when
called. The package is published so its design can be reviewed and
depended on before it is implemented. Version 0.1.0 will be the first
working release.

## What it is

A transformer that has read a sequence hands back its **last hidden
state**: one vector per token, a matrix of `[tokens, dim]`, where `dim` is
the model's width. **Pooling** reduces that matrix to one vector per
document. Which pooling is correct is a property of the checkpoint, fixed
when it was trained and written on its model card.

Sequences in one batch have different lengths, so the short ones are
**padded** to the length of the longest. An **attention mask** is one entry
per token position: non-zero for a real token, zero for padding. Padding
tokens have embeddings, because the model computed something for them, and
those numbers are not zero. Pooling that ignores the mask averages a
variable amount of noise into every document.

Two vectors are compared by a **similarity**. The **cosine similarity** is
the dot product divided by both lengths, and lies between −1 and 1. The
**dot product** is the same number without that division. The **Euclidean
distance** is the straight-line distance between the two points. In this
package every comparison is a similarity and higher always means closer, so
the Euclidean entry is the negated distance.

**L2 normalisation** scales a vector to length 1. On vectors of length 1
the dot product is the cosine, and the Euclidean distance is
`sqrt(2 - 2·cos)`, which falls as the cosine rises. A ranking over
normalised vectors is therefore the same ranking under all three
similarities, and a caller should use the cheapest. On unnormalised vectors
they disagree.

**Matryoshka Representation Learning** trains a model so that every prefix
of its output is itself a usable embedding: the first 64 coordinates of a
768-coordinate vector carry the coarse structure and the rest refine it. A
caller can store the whole vector and search a prefix of it.

**Quantisation** replaces each coordinate with a smaller number. **int8**
maps a range of values onto the codes −127 to 127, an **affine**
quantisation with a **scale**, the size of one step, and a **zero point**,
the code that stands for 0.0. **Binary** keeps only each coordinate's sign,
one bit each, and compares two vectors by **Hamming distance**, the number
of bits that differ.

Every function here is arithmetic over numbers the caller already holds.
Nothing loads a model, opens a socket, reads a clock or draws a random
number. There is not one effect row in the package.

The cost of a corpus of one million vectors of 768 coordinates:

| Stored as | Bytes per vector | One million |
| --- | --- | --- |
| `Float` | 6,144 | 6 GB |
| int8 | 768 | 768 MB |
| binary | 96 | 96 MB |

The poolings, and the checkpoints that were trained for each:

| Pooling | Name in a config | An example checkpoint |
| --- | --- | --- |
| `EmbMeanPool` | `mean` | `sentence-transformers/all-MiniLM-L6-v2` |
| `EmbClsPool` | `cls` | `BAAI/bge-base-en-v1.5` |
| `EmbLastPool` | `last` | `intfloat/e5-mistral-7b-instruct` |
| `EmbMaxPool` | `max` | rare in current checkpoints |

## Install

```
novo pkg add embeddings-nv
```

## Example

```novo
use ndfloat
use embpool
use embnorm
use embsim

// Pool one sequence to a single vector, scale it to unit length, and
// score it against itself. Every step answers the same fault type, so
// `!` carries a failure straight out of the function.
fn embed_and_score(tokens: NdFloat, mask: [Int]) -> Result<Float, EmbFault>
    let pooled = embpool.pool(EmbMeanPool, tokens, mask)!
    let unit = embnorm.l2_normalize(pooled)!
    let s = embsim.score(EmbCosine, unit, unit)!
    Ok(s)

fn main() [io]
    // Two tokens of a three-dimensional model, one row per token: the
    // shape a transformer's last hidden state has.
    match ndfloat.of_list([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [2, 3])
        Err(e) => println(e.message())
        Ok(tokens) =>
            // The second token is padding. A zero in the mask keeps it
            // out of the sum and out of the divisor.
            match embed_and_score(tokens, [1, 0])
                Ok(s)  => println("${s}")
                Err(f) => println(f.message())
```

Build and test with `novo pkg build` and `novo test`. Today `novo test`
fails on purpose: every test reaches a `not implemented` panic. The tests
are the specification the implementation will have to satisfy.

## What the package contains

| Module | Contents |
| --- | --- |
| `embpool` | The four poolings, the mask helpers, and the batch form that pools a rank-3 array to one vector per sequence. |
| `embnorm` | L2 normalisation for one vector and for every row, the length measurements, and Matryoshka truncation with and without renormalisation. |
| `embsim` | The three similarities, one pair at a time and one query against a whole corpus, the top-k selection, and exact search. |
| `embquant` | int8 quantisation against a shared calibration range, binary quantisation, Hamming distance, and the three storage sizes. |
| `embfault` | The eleven ways a call can refuse, their messages, and the one conversion from an ndarray-nv error. |

## How to choose an entry point

**`embpool.pool` takes the pooling as an argument.** Use it when the
pooling comes from a model card at run time, which is the ordinary case.
`mean_pool`, `cls_pool`, `last_pool` and `max_pool` are the same four
directly, for a caller that has already decided.

**`embpool.pool_batch` pools a whole batch.** It takes a rank-3 array of
`[batch, tokens, dim]` and answers rank 2, `[batch, dim]`, which is exactly
what `embnorm.normalize_rows` and `embsim.score_many` take. A batch never
has to be taken apart and put back together.

**`embsim.search` is score-then-select in one call.** Use it for exact
search over a corpus that fits in memory. `score_many` and `top_k` are the
two halves, for a caller that wants the scores as well.
`embsim.pairwise` is every query against every row, and a thousand queries
against a million rows is a billion numbers.

**`embquant.binary_search` is the shortlist.** The usual arrangement is
binary over the whole corpus, then `embsim.search` over the few hundred
rows it answers, against the float vectors. `hamming_score` is on the same
scale as `embsim`'s scores, so one threshold means the same thing in both
stages.

## The rules a user needs

1. **Pass the mask.** A padded batch has padding embeddings in it, and they
   are not zero. Without the mask the same document embeds differently
   depending on what else was in its batch, and nothing about the result
   looks wrong. `embpool.no_mask` is how a caller with genuinely one
   unpadded sequence says so on purpose.
2. **Mean pooling divides by the number of kept tokens, never by the number
   of rows.** sentence-transformers' `Pooling` module divides by the mask
   sum. Zeroing the padding and then dividing by the row count scales every
   embedding down by however much padding it happened to get. On normalised
   output that scaling cancels, so the mistake survives until someone
   switches to dot-product similarity.
3. **Last-token pooling takes the last unmasked row, never the last row.**
   On a right-padded batch the last row is a pad token's embedding. On a
   left-padded batch the two agree, which is why the mistake appears when a
   pipeline moves between tokenizers. `embpool.last_index` and
   `embpool.is_right_padded` report what the mask actually says.
4. **Use the pooling the checkpoint was trained for.** The table above
   gives the four names sentence-transformers' `1_Pooling/config.json`
   uses, and `embpool.pooling_of_name` reads one. The wrong pooling gives
   vectors of the right shape that cluster plausibly and retrieve badly.
5. **Higher is closer, everywhere in this package.**
   `EmbEuclidean`'s score is the negated distance, as sentence-transformers'
   `SimilarityFunction` defines it, so `top_k` is one function.
   `embsim.euclidean_distance` is the positive number, for reporting.
6. **An index compares by minimum and this package sorts by maximum.**
   [hnsw-nv](https://novo-lang.org/packages/hnsw-nv) takes distances, where
   smaller is closer. `embsim.as_distance` converts a cosine similarity to
   the cosine distance an index wants. Mixing the two conventions returns
   the least relevant documents and raises nothing.
7. **Nothing here normalises for you.** A function that normalised its
   input would hand back a vector the caller never stored, and a pipeline
   that uses magnitude deliberately would lose it.
   `embnorm.is_normalized` and `embnorm.rows_normalized` are what the
   package offers instead, and 0.0001 is a sensible tolerance.
8. **A prefix of a unit vector is not a unit vector.** Its length is the
   square root of the fraction of the energy that fell in the prefix.
   Truncating and comparing by cosine is safe, because cosine divides the
   length out. Truncating and comparing by dot product favours whichever
   vectors kept more energy early. `embnorm.truncate_normalized` truncates
   and renormalises, and is the one to reach for.
9. **Truncation only works on a model trained for it, and this package
   cannot tell.** Kusupati et al., *Matryoshka Representation Learning*
   (2022), is the training method. `embnorm.mean_energy_in_prefix` over
   your own corpus is the measurement that decides: a trained model at half
   its dimensions keeps most of its energy, and an untrained one keeps
   about `dim / full`, which is what any prefix keeps by chance.
10. **An int8 calibration range is measured over a corpus, never over one
    vector.** Two vectors quantised against different ranges cannot be
    compared, and a dot product between them is a number with no meaning.
    The per-vector form does not exist here. `embquant.calibrate` measures
    a range over a sample of rows, `to_int8` takes that range, and
    `int8_dot` answers `EmbScaleMismatch` when the two ranges differ.
11. **A coordinate outside the calibration range is clamped, not refused.**
    One unusual document should not fail an ingestion. `embquant.covers`
    is the occasional check on live data, and a rising rate of misses means
    the range should be measured again.
12. **Binary quantisation is the sign bit and nothing else.** No threshold
    and no per-coordinate centring. `embquant.hamming_score` is
    `1 - 2 * hamming / dim`, which is the cosine similarity of the two sign
    vectors, so a binary shortlist and a float re-rank share a scale.
13. **A zero vector is refused under cosine, under normalisation and under
    binary quantisation.** There is no direction to keep, and every
    substitute answer is wrong in a way that shows up as poor retrieval
    rather than as an error.
14. **`embsim.top_k` keeps row order on a tie.** Duplicate rows rank the
    way they were stored. A result list that reorders between identical
    runs cannot be tested.
15. **`EmbHit.index` is a row position in the matrix that was ranked**, and
    not an identifier this package invented. A caller joining back to its
    own records keeps its identifiers in the same order as its rows.
16. **Faults split into two kinds.** `embfault.is_shape` is true for the
    shape and parameter faults, which a caller fixes by looking at its
    pipeline, and false for `EmbZeroVector`, `EmbNotFinite` and
    `EmbEmptyMask`, which are rows an ingestion loop should count and skip.

## What is not included

- **A model.** `embpool` takes the last hidden state as an argument. Where
  it came from is the caller's business.
- **A tokenizer.** The attention mask arrives with the matrix.
  [tokenizers-nv](https://novo-lang.org/packages/tokenizers-nv) is where a
  mask is produced.
- **A store.** Nothing here writes anything down.
  [vectorstore-nv](https://novo-lang.org/packages/vectorstore-nv) keeps
  documents and [hnsw-nv](https://novo-lang.org/packages/hnsw-nv) indexes
  vectors.
- **Approximate search.** `embsim.search` is exact and its cost is linear
  in the corpus. hnsw-nv is the approximate one.
- **Automatic normalisation.** See rule 7.
- **Per-vector int8 calibration.** See rule 10.
- **A microcontroller build.** No module here is declared to build for a
  device with no heap allocator, and ndarray-nv makes no such claim either.
  A device matching one embedding against a handful of stored ones wants
  the binary half of `embquant` over integers, with no `NdFloat` in it,
  which is a different module.

## Related packages

- [ndarray-nv](https://novo-lang.org/packages/ndarray-nv) is the
  n-dimensional array. Pooling is a reduction along an axis and one query
  against a corpus is a matrix-vector product, so the matrix type is the
  right argument and ndarray-nv owns it.
- [hnsw-nv](https://novo-lang.org/packages/hnsw-nv) is approximate nearest
  neighbour search over a graph. It takes a row as a `[Float]` rather than
  an `NdFloat`, because its subject is a graph over rows of one dimension
  it already knows. `ndfloat.to_list` of one row is the seam.
- [vectorstore-nv](https://novo-lang.org/packages/vectorstore-nv) keeps
  documents beside their vectors and ranks them. It is the layer above
  this one.
- [gguf-nv](https://novo-lang.org/packages/gguf-nv) reads the model file
  the hidden state came out of. The two packages never meet.
- [tokenizers-nv](https://novo-lang.org/packages/tokenizers-nv) turns text
  into the token identifiers and the mask a model is fed.
- `std.vec` in the standard library is dot, cosine, norm, add and scale
  over a plain list of floats. It is the right tool for a handful of
  vectors and has no pooling, no quantisation and no top-k.
- `std.nn` in the standard library is the neural-network layers that
  produce a hidden state, and `std.tensor` is the arithmetic underneath
  them.
- `std.store` in the standard library is an in-memory vector store for
  retrieval-augmented generation.

## Tests

```bash
novo test tests/embpool_tests.nv     # 6 tests: pooling, normalisation, truncation
novo test tests/embsim_tests.nv      # 5 tests: the three similarities, top-k, quantisation
```

The expected numbers are arithmetic, written out beside each assertion. The
pooling fixture is a three-token, two-dimensional matrix whose third row is
padding, chosen so that the right answer and the two classic wrong answers
are three different vectors: the mask-weighted mean is `[2, 2]`, dividing
by the row count gives `[1.33, 1.33]`, and ignoring the mask gives
`[4.33, 4.33]`. The similarity fixtures are the orthogonal pair `[1, 0]`
and `[0, 1]`, and `[0.6, 0.8]`, which is the 3-4-5 triangle normalised. The
identity `d² = 2 − 2·cos` is asserted directly, because it is the reason a
ranking over unit vectors is the same under all three similarities.

sentence-transformers is the oracle, and it arrives with the bodies: its
`Pooling`, its `SimilarityFunction` and its `quantize_embeddings` over the
same matrices, compared coordinate by coordinate and bit by bit.

The tests compile today and fail at run. Some assertions reach ndarray-nv's
`todo()` before they reach this package's, because ndarray-nv is an
interface release too. That is the expected state. They turn green one at a
time as bodies land.

## Implementation status

| Item | Implemented |
| --- | --- |
| `embpool.pooling_name`, `.pooling_of_name` | no |
| `embpool.pool`, `.mean_pool`, `.cls_pool`, `.last_pool`, `.max_pool`, `.pool_batch` | no |
| `embpool.no_mask`, `.kept_count`, `.last_index`, `.is_right_padded` | no |
| `embnorm.l2_normalize`, `.normalize_rows`, `.norm`, `.norm_of_rows` | no |
| `embnorm.is_normalized`, `.rows_normalized` | no |
| `embnorm.truncate`, `.truncate_normalized`, `.truncate_rows` | no |
| `embnorm.energy_in_prefix`, `.mean_energy_in_prefix` | no |
| `embsim.metric_name`, `.metric_of_name` | no |
| `embsim.score`, `.cosine`, `.dot`, `.euclidean_distance`, `.euclidean_squared` | no |
| `embsim.score_many`, `.pairwise`, `.top_k`, `.search` | no |
| `embsim.hit_scores`, `.hit_indices`, `.as_distance` | no |
| `embquant.bits_per_word`, `.binary_words`, `.int8_bytes`, `.binary_bytes`, `.float_bytes` | no |
| `embquant.calibrate`, `.covers`, `.to_int8`, `.to_int8_rows`, `.from_int8` | no |
| `embquant.int8_scale`, `.int8_zero`, `.int8_dot`, `.int8_cosine` | no |
| `embquant.to_binary`, `.to_binary_rows`, `.from_binary` | no |
| `embquant.hamming`, `.hamming_score`, `.binary_search` | no |
| `embfault.is_shape`, `.of_array`, `EmbFault.message` | no |

## Licence

Apache-2.0. See `LICENSE`.

<!-- docs/writing-a-readme.md is the style guide for this page. -->

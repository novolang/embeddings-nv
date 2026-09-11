# embeddings-nv

**Status: NOT IMPLEMENTED — interface only.**

Every public function below is published with its signature and its
effect row, and every body is `todo()`.  Installing this package works;
calling it panics with `not implemented`.

## What this is

**The arithmetic between a model and a store.**  A transformer hands
back one vector per token; a vector store wants one vector per document,
normalised, possibly truncated, possibly a byte or a bit per coordinate.
Everything in that sentence is arithmetic, and all of it is here.

No model, no client, no store.  One dependency: ndarray-nv.

| surface | module | reach for it when |
| --- | --- | --- |
| **pooling** | `embpool` | you have a model's last hidden state |
| **unit length & Matryoshka** | `embnorm` | you have a vector and a storage budget |
| **similarity & top-k** | `embsim` | you have a query and a corpus in memory |
| **int8 & binary** | `embquant` | your corpus stopped fitting |
| the **faults** | `embfault` | you are telling somebody why their batch did not embed |

## Adding it, and checking it

```bash
novo pkg add embeddings-nv         # into your novo.toml
novo pkg build                     # type- and effect-check the package
novo test --isolate tests/embsim_tests.nv
```

`novo test` is red today and that is the point of the release.  Some
assertions fail on **ndarray-nv's** stubs rather than this package's,
because ndarray-nv is an interface too — that is the interface tier
composing, and it is worth seeing before the first body lands.

## The one example that will work

```novo
use embsim
use ndfloat

fn main() [io]
    match ndfloat.of_list([1.0, 0.0], [2])
        Err(e) => println(e.message())
        Ok(a)  =>
            match embsim.score(EmbDot, a, a)
                Ok(s)  => println("${s}")
                          // : 1
                Err(f) => println(f.message())
```

## The load-bearing interface

```novo norun:pseudo
pub fn mean_pool(tokens: NdFloat, mask: [Int]) -> Result<NdFloat, EmbFault>
```

**The mask is not optional, and this is where everyone is wrong.**

A batch is padded to its longest sequence, and the padding tokens have
embeddings — the model computed something for them, and it is not zero.
A mean over all the rows therefore averages in a variable amount of
noise that depends on *what else was in the batch*, so the same document
embeds differently depending on its neighbours.  Nothing about that
failure announces itself: the vectors are the right shape, they cluster
plausibly, and retrieval is a few points worse than it should be
forever.

Two specific errors follow from it, and each has its own function here
so that neither can be written by accident:

**Mean pooling divides by the mask sum, not the row count.**
Multiplying by the mask and then dividing by `tokens` is the bug that
*looks* right — it zeroes the padding and then divides by a number that
includes it, so every embedding is scaled down by however much padding
it happened to get.  On a normalised output the scaling cancels, which
is why it survives review; the day somebody switches to dot-product
similarity for speed, it does not.

**Last-token pooling takes the last unmasked row, not the last row.**
On a right-padded batch those differ and the last row is a pad token's
embedding.  On a left-padded one they agree — which is why this bug
appears exactly when a pipeline moves from a decoder tokenizer to an
encoder's.  `embpool.last_index` and `embpool.is_right_padded` are what
a caller checks its own convention with.

And the pooling itself is not a preference.  It is a property of the
checkpoint, decided when it was trained:
`sentence-transformers/all-MiniLM-L6-v2` is mean, `BAAI/bge-*` is CLS,
`intfloat/e5-mistral-7b-instruct` is last token.  `EmbPooling`'s names
are the ones a `1_Pooling/config.json` uses, so a caller can map one
across without a table.

## Higher is closer, and that is the opposite of hnsw-nv

An index compares by **minimum**, because that is what a priority queue
and a greedy descent want.  A ranking sorts by **maximum**, because that
is what a result list is.  Mixing them is a silent reversal that returns
the least relevant documents, so each package states its rule once and
keeps it everywhere — and `embsim.as_distance` is the single line
between them.

Euclidean is the one that has to be bent to fit, and it is bent the way
sentence-transformers bends it: `EmbEuclidean`'s **score** is the
negated distance, so `top_k` is one function rather than three.
`euclidean_distance` is the positive number, for reporting.

**On unit vectors the three agree on the order.**  The dot product *is*
the cosine, and the Euclidean distance is `sqrt(2 - 2·cos)` — monotonic
in it.  So a ranking over normalised embeddings is the same under all
three metrics and a caller should use the cheapest.  On unnormalised
ones they disagree, and which is right is a property of how the model
was trained.

**This package never normalises for you.**  A function that normalised
its input would hand back a vector the caller did not have, and a caller
doing dot-product similarity over deliberately unnormalised vectors —
where magnitude carries confidence, which some rerankers rely on — would
find it gone.  `embnorm.is_normalized` is what the package says instead.

## A prefix of a unit vector is not a unit vector

Matryoshka Representation Learning trains a model so that every *prefix*
of its output is itself a usable embedding: the first 64 of a 768-vector
carry the coarse structure and the rest refine it.  So a caller can
store 768 and search 64, or store 256 and lose a couple of points of
recall for a third of the memory.

The step everyone forgets is in the heading.  A prefix's norm is the
square root of the fraction of the energy that fell in it — large for a
trained Matryoshka model, never 1.  Truncating and comparing by cosine
is fine, because cosine renormalises.  Truncating and comparing by **dot
product** is a systematic bias towards whichever vectors kept more
energy early.  `embnorm.truncate_normalized` is the one to reach for,
and `embnorm.truncate` is there for the caller who is about to
renormalise anyway.

**Truncation only works on a model trained for it**, and this package
cannot tell.  What it offers instead is a measurement:
`embnorm.mean_energy_in_prefix` over a corpus.  A trained Matryoshka
model at half its dimensions keeps most of its energy; a model that was
not keeps roughly `dim / full`, which is what an arbitrary vector's
prefix keeps by chance.  Those two numbers are far apart, and telling
them apart on *your* data is the decision.

## The calibration is the corpus, never the vector

The obvious int8 quantiser takes each vector's own minimum and maximum
and maps them to ±127.  Every vector is then individually as accurate as
int8 allows — and **no two of them can be compared**, because each
carries a different scale, so a dot product between two of them is a
number with no meaning.

The failure is silent.  The codes are integers, the arithmetic runs, the
results are plausible, and recall is quietly a fraction of what it should
be.  So the per-vector form does not exist here at all:

```novo norun:pseudo
let (lo, hi) = embquant.calibrate(sample)!        // over a CORPUS
let q = embquant.to_int8(v, lo, hi)!              // against THAT range
embquant.int8_dot(a, b)                           // EmbScaleMismatch if they differ
```

A few thousand rows is enough to calibrate, and a range measured on them
holds for a corpus that grows — embedding coordinates are distributed
the way the model trained them rather than the way the corpus happens to
be.  `embquant.covers` is how to check that rather than assume it, and a
coordinate outside the range is **clamped rather than refused**, because
one unusual document should not fail an ingestion.

**Binary is a sign bit and nothing more.**  No threshold, no
per-coordinate centring — both help slightly on some models, both need a
calibration of their own, and the plain sign is what every published
binary-embedding result uses.  Hamming distance over the packed bits
then ranks almost as well as cosine over the floats, which is the
surprising fact the whole technique rests on, and
`embquant.hamming_score` puts it on the same `[-1, 1]` scale as
`embsim`'s so that a binary shortlist and a float re-rank share a
threshold.

The ladder at 768 dimensions, which is why this module is in an
embeddings package rather than a storage one:

| | bytes per vector | a million of them |
| --- | --- | --- |
| `Float` | 6,144 | 6 GB |
| int8 | 768 | 768 MB |
| binary | 96 | 96 MB |

## Where this sits

```
    a model
       │  last hidden state: [tokens, dim]
       ▼
    embpool ──► embnorm ──► embquant ──► a store
       │            │           │
       └────────────┴───────────┴──► embsim, for a corpus in memory
```

**hnsw-nv** is the next step past `embsim.search`: exact search is
linear in the corpus, and the crossover is lower than it looks — a
hundred thousand 768-vectors is three hundred megabytes of arithmetic
per query.  The seam is one line, because hnsw-nv takes `[Float]` rows
rather than `NdFloat`:

```novo norun:pseudo
let row = ndfloat.to_list(ndfloat.index_axis(matrix, 0, i)!)
ix = hnswgraph.insert(ix, id, row, u)!
```

That asymmetry is deliberate on both sides.  This package's subject is a
token-embedding **matrix**, and a shape is exactly what that is —
pooling is a reduction along an axis, one-against-many is a
matrix-vector product, a batch of queries is a matmul.  hnsw-nv's
subject is a **graph** over N rows of one dimension it already knows, and
a shape per row would restate a fact it holds once.

**gguf-nv** is where the model that produced the matrix comes from, and
the two packages never meet: a GGUF reader hands out byte ranges and an
inference engine turns them into a hidden state.

## The layer, and the claim this package does not make

`core`, and there is not one effect row in the package.

**No `@tier(embedded)` claim.**  A device probe would not build:
ndarray-nv makes no embedded claim either, and a `core` package's probe
is built against its dependencies' modules as well as its own.  The
audit's `core-embedded` row passes and says the package makes no claim.

If an edge consumer appears — a device matching a wake-word embedding
against a handful of stored ones — the shape to build is not this
package at the embedded tier.  It is `embquant`'s binary half over
integers, with no `NdFloat` anywhere in it, which is a different module
and probably a different row.

## Naming

Every public type and every enum variant starts `Emb`, and every module
file starts `emb`.  Struct and enum identity is keyed by **name** across
a whole assembly, so `EmbCosine` and hnsw-nv's `HnswCosine` are the same
idea under two names on purpose — the two packages compute the same
three distances and a program that used both would otherwise be refused.

## What is the technique, and what is this package's choice

**sentence-transformers, and binding**: the four poolings and their
names; the mask-weighted mean; normalisation before comparison; the
negated Euclidean score; the int8 affine quantisation with a shared
range; and the binary sign-bit quantisation with Hamming ranking.

**This package's choice**: no per-vector int8 calibration at all, and
`EmbScaleMismatch` rather than a plausible number; clamping an
out-of-range coordinate rather than refusing the row; refusing a zero
vector under cosine, under normalisation and under binary quantisation;
`truncate` and `truncate_normalized` as two functions rather than a
flag; `energy_in_prefix` as a measurement in place of a claim about
which models truncate well; and excluding masked rows from `max_pool`
rather than zeroing them.

## The reference implementation

[sentence-transformers](https://github.com/UKPLab/sentence-transformers)
— its `Pooling` module, its `SimilarityFunction`, and its
`quantize_embeddings`.  The Matryoshka behaviour is from Kusupati et al.,
*Matryoshka Representation Learning* (2022), and the binary-embedding
result from the `mixedbread-ai` and Cohere published evaluations.

The oracle arrives with the bodies: sentence-transformers' own pooling,
normalisation and quantisation over the same matrices, compared
coordinate by coordinate and bit by bit, as a generated run beside the
two suites in `tests/`.

Apache-2.0.

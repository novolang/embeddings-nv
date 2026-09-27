#!/usr/bin/env python3
"""Write the numpy vectors in tests/embvectors_tests.nv.

Every expected number is computed here with numpy, following
sentence-transformers where it defines the operation: its `Pooling`
module for the four poolings, its `SimilarityFunction` for the three
scores (the Euclidean score is the negated distance), and its
`quantize_embeddings` for the binary sign bit.  The int8 codes follow
this package's corpus-wide affine rule, which the module comment of
`embquant` states.  Run it from the package root:

    python3 tools/gen_vectors.py

The generator is seeded, so the output is the same on every run.
"""
import math

import numpy as np

rng = np.random.default_rng(20260927)

def flo(xs):
    return ",".join("%.17g" % float(x) for x in np.ravel(xs))

def ints(xs):
    return ",".join(str(int(x)) for x in np.ravel(xs))

def round_away(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)

pool_rows = []
for _ in range(40):
    t = int(rng.integers(1, 7)); d = int(rng.integers(1, 9))
    tokens = rng.normal(size=(t, d))
    mask = rng.integers(0, 2, size=t)
    if mask.sum() == 0:
        mask[int(rng.integers(0, t))] = 1
    m = mask[:, None].astype(float)
    mean = (tokens * m).sum(0) / m.sum()
    kept = tokens[mask != 0]
    last = tokens[np.nonzero(mask)[0][-1]]
    mx = kept.max(0)
    unit = mean / np.linalg.norm(mean)
    pool_rows.append("%d,%d|%s|%s|%s|%s|%s|%s|%s" % (
        t, d, flo(tokens), ints(mask), flo(mean), flo(tokens[0]), flo(last), flo(mx), flo(unit)))

sim_rows = []
for _ in range(40):
    d = int(rng.integers(2, 12))
    a = rng.normal(size=d); b = rng.normal(size=d)
    k = int(rng.integers(1, d + 1))
    cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    energy = (a[:k] @ a[:k]) / (a @ a)
    sim_rows.append("%s|%s|%.17g|%.17g|%.17g|%d|%.17g" % (
        flo(a), flo(b), cos, a @ b, np.linalg.norm(a - b), k, energy))

quant_rows = []
for _ in range(30):
    rows = int(rng.integers(2, 6)); d = int(rng.integers(1, 140))
    corpus = rng.normal(size=(rows, d))
    lo, hi = corpus.min(), corpus.max()
    scale = (hi - lo) / 254.0
    zero = int(round_away(-lo / scale)) - 127
    codes = np.clip(round_away(corpus / scale) + zero, -127, 127).astype(np.int64)
    int8_dot = float(((codes[0] - zero) * (codes[1] - zero)).sum()) * scale * scale
    bits = corpus > 0
    def words(row):
        out = []
        for w in range((d + 63) // 64):
            v = 0
            for i in range(w * 64, min(d, w * 64 + 64)):
                if row[i]:
                    v |= 1 << (i - w * 64)
            if v >= 1 << 63:
                v -= 1 << 64
            out.append(v)
        return out
    hamming = int((bits[0] != bits[1]).sum())
    quant_rows.append("%d,%d|%s|%.17g|%.17g|%s|%d|%.17g|%s|%s|%d" % (
        rows, d, flo(corpus), lo, hi, ints(codes[0]), zero, int8_dot,
        ints(words(bits[0])), ints(words(bits[1])), hamming))

rank_rows = []
for _ in range(30):
    rows = int(rng.integers(1, 12)); d = int(rng.integers(1, 6))
    # Rounded to one decimal so that ties occur, with no zero rows.
    corpus = rng.normal(size=(rows, d)).round(1)
    corpus[np.all(corpus == 0, axis=1)] = 0.5
    q = rng.normal(size=d).round(1)
    if np.all(q == 0):
        q[0] = 0.5
    k = int(rng.integers(1, rows + 1))
    # Summed left to right in Python floats, the order the package sums
    # in, so that two equal scores are equal in both and the tie order
    # can be compared.
    def seq_dot(x, y):
        acc = 0.0
        for u, v in zip(x, y):
            acc += float(u) * float(v)
        return acc
    for metric in ("cosine", "dot", "euclidean"):
        s = []
        for row in corpus:
            if metric == "cosine":
                s.append(seq_dot(row, q) / (math.sqrt(seq_dot(row, row)) * math.sqrt(seq_dot(q, q))))
            elif metric == "dot":
                s.append(seq_dot(row, q))
            else:
                diff = [float(u) - float(v) for u, v in zip(row, q)]
                s.append(-math.sqrt(seq_dot(diff, diff)))
        order = sorted(range(rows), key=lambda i: (-s[i], i))[:k]
        rank_rows.append("%s|%d,%d|%s|%s|%d|%s|%s" % (
            metric, rows, d, flo(corpus), flo(q), k, flo(s), ints(order)))

def splice(body, fn, rows):
    marker = "fn %s() -> [Str]\n    [\n" % fn
    start = body.index(marker) + len(marker)
    end = body.index("]\n\n", start)
    lines = ['        "%s",' % r for r in rows]
    lines[-1] = lines[-1].rstrip(",")
    return body[:start] + "\n".join(lines) + body[end:]

path = "tests/embvectors_tests.nv"
body = open(path).read()
body = splice(body, "pool_vectors", pool_rows)
body = splice(body, "sim_vectors", sim_rows)
body = splice(body, "quant_vectors", quant_rows)
body = splice(body, "rank_vectors", rank_rows)
open(path, "w").write(body)

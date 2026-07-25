"""Logistic-regression play-probability model for DMB setlists.

Trained causally: every (song, show) event uses only information available
before that show. Features encode the discoveries from analysis.py:

- smoothed running tour rate (the workhorse)
- gap-since-last-played buckets (rotation / "due" behavior)
- rate x gap-1 interaction (core songs repeat back-to-back at 77%,
  regular songs at 20% — a global penalty is wrong)
- never-played-this-tour flag + prior-tour rate (deep-cut returns)

Pure stdlib: IRLS Newton solver on ~30k events x 10 features.
"""

from __future__ import annotations

from collections import defaultdict

SMOOTH = 3.0
GAPS = [("g1", lambda g: g == 1), ("g2", lambda g: g == 2),
        ("g3", lambda g: g == 3), ("g4", lambda g: g == 4),
        ("g5_8", lambda g: 5 <= g <= 8), ("g9p", lambda g: g >= 9)]
FEATS = (["bias", "rate", "prior_rate", "never", "rate_x_g1"]
         + [name for name, _ in GAPS])


def featurize(rate, prior_rate, gap):
    """gap: shows since last played this leg, or None."""
    x = {"bias": 1.0, "rate": rate, "prior_rate": prior_rate,
         "never": 1.0 if gap is None else 0.0,
         "rate_x_g1": rate if gap == 1 else 0.0}
    for name, test in GAPS:
        x[name] = 1.0 if (gap is not None and test(gap)) else 0.0
    return [x[f] for f in FEATS]


def gap_stats(indices):
    """(mean own gap, gap sd) from a song's play indices, or None."""
    if len(indices) < 3:
        return None
    gaps = [b - a for a, b in zip(indices, indices[1:])]
    mu = sum(gaps) / len(gaps)
    var = sum((g - mu) ** 2 for g in gaps) / len(gaps)
    return mu, var ** 0.5


def leg_events(leg, prior_rate, candidates, min_show=3):
    """Yield raw (rate, prior_rate, gap, own, label) causally per leg.

    own = (mean own gap, sd) once the song has >= 4 plays this leg,
    else None — feeds the song-period timing signal.
    """
    idx = defaultdict(list)
    for i, show in enumerate(leg):
        tonight = set(show["songs"])
        if i >= min_show:
            for song in candidates:
                ii = idx[song]
                gap = i - ii[-1] if ii else None
                rate = ((len(ii) + SMOOTH * prior_rate[song])
                        / (i + SMOOTH))
                yield (rate, prior_rate[song], gap, gap_stats(ii),
                       1.0 if song in tonight else 0.0)
        for song in tonight:
            idx[song].append(i)


def candidate_pool(leg, prior_rate, min_prior=0.05):
    pool = set()
    for s in leg:
        pool.update(s["songs"])
    pool.update(s for s, r in prior_rate.items() if r >= min_prior)
    return pool


def solve(a, b):
    """Gaussian elimination for small dense systems."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[piv] = m[piv], m[col]
        if abs(m[col][col]) < 1e-12:
            m[col][col] += 1e-8
        div = m[col][col]
        m[col] = [v / div for v in m[col]]
        for r in range(n):
            if r != col and m[r][col]:
                f = m[r][col]
                m[r] = [v - f * w for v, w in zip(m[r], m[col])]
    return [m[i][n] for i in range(n)]


def train(events, iters=8, ridge=1e-3):
    import math
    k = len(FEATS)
    w = [0.0] * k
    xs = [featurize(e[0], e[1], e[2]) for e in events]
    ys = [e[-1] for e in events]
    for _ in range(iters):
        grad = [0.0] * k
        hess = [[ridge if i == j else 0.0 for j in range(k)]
                for i in range(k)]
        for x, y in zip(xs, ys):
            z = sum(wi * xi for wi, xi in zip(w, x))
            z = max(-30, min(30, z))
            p = 1 / (1 + math.exp(-z))
            g = p - y
            s = p * (1 - p) + 1e-9
            for i in range(k):
                if x[i]:
                    grad[i] += g * x[i]
                    for j in range(i, k):
                        if x[j]:
                            hess[i][j] += s * x[i] * x[j]
        for i in range(k):
            for j in range(i):
                hess[i][j] = hess[j][i]
        step = solve(hess, grad)
        w = [wi - si for wi, si in zip(w, step)]
    return w


def prob(w, rate, prior_rate, gap):
    import math
    z = sum(wi * xi for wi, xi in
            zip(w, featurize(rate, prior_rate, gap)))
    return 1 / (1 + math.exp(-max(-30, min(30, z))))


# --- Direct 2-D empirical table: P(play | rate tier, gap bucket) ---

RATE_EDGES = [0.08, 0.15, 0.25, 0.35, 0.50]


def rate_tier(rate):
    for i, e in enumerate(RATE_EDGES):
        if rate < e:
            return i
    return len(RATE_EDGES)


def gap_bucket(gap):
    if gap is None:
        return "never"
    if gap <= 4:
        return str(gap)
    return "5-8" if gap <= 8 else "9+"


def rel_bucket(gap, own):
    """Gap relative to the song's own period + regularity class."""
    mu, sd = own
    rel = gap / mu
    b = ("<0.6" if rel < 0.6 else "0.6-0.9" if rel < 0.9 else
         "0.9-1.2" if rel < 1.2 else "1.2-1.6" if rel < 1.6 else ">1.6")
    return b, "reg" if sd < 1.0 else "loose"


class TableModel:
    """Empirical conditional play-probability table with shrinkage.

    Songs with an established personal period (>= 4 plays this leg) are
    binned by (rate tier, gap / own period, regularity); others by
    (rate tier, absolute gap bucket). 'Never this tour' events bin by
    prior-tour rate tier (deep-cut returns depend on career frequency).
    Sparse cells shrink toward the tier's overall play rate.
    """

    def __init__(self, k=5.0):
        self.k = k
        self.cells = defaultdict(lambda: [0.0, 0.0])
        self.tier_base = defaultdict(lambda: [0.0, 0.0])

    def key(self, rate, prior_rate, gap, own):
        if gap is None:
            return ("never", rate_tier(prior_rate))
        if gap == 1 or own is None:
            # gap 1 ALWAYS uses the absolute cell: the back-to-back taboo
            # is a hard rule that the relative buckets would dilute
            return (gap_bucket(gap), rate_tier(rate))
        b, reg = rel_bucket(gap, own)
        return (f"rel{b}|{reg}", rate_tier(rate))

    def fit(self, events):
        """events: (rate, prior_rate, gap, own, label[, weight]).

        Weights let recent eras dominate — DMB's no-repeat culture
        hardened from ~5 back-to-back repeats/night in 2015-2019 to ~0.8
        in 2026, so old years must not set today's gap-1 cells.
        """
        for ev in events:
            rate, pr, gap, own, y = ev[:5]
            w = ev[5] if len(ev) > 5 else 1.0
            key = self.key(rate, pr, gap, own)
            self.cells[key][0] += w * y
            self.cells[key][1] += w
            tb = self.tier_base[key[1]]
            tb[0] += w * y
            tb[1] += w
        return self

    def prob(self, rate, prior_rate, gap, own=None):
        key = self.key(rate, prior_rate, gap, own)
        hits, n = self.cells[key]
        b_hits, b_n = self.tier_base[key[1]]
        base = b_hits / b_n if b_n else 0.05
        return (hits + self.k * base) / (n + self.k)

"""Exploratory analyses over DMB setlist history.

Each analysis prints evidence for/against a candidate predictive signal.
Run: python3 analysis.py <name> | all
"""

from __future__ import annotations

import datetime
import statistics
import sys
from collections import defaultdict

from predict import (encore, load_shows, main_set, segments)


def tours(shows):
    """Tour legs with >= 10 shows, most recent last."""
    return [leg for leg in segments(shows) if len(leg) >= 10]


def overlap_structure(shows):
    """How much of tonight's set appeared k shows ago?"""
    print("=== Set overlap with k-shows-ago (share of tonight's songs) ===")
    for leg in tours(shows):
        year = leg[0]["date"][:4]
        by_k = defaultdict(list)
        for i, s in enumerate(leg):
            tonight = set(s["songs"])
            for k in range(1, 8):
                if i - k >= 0:
                    prev = set(leg[i - k]["songs"])
                    by_k[k].append(len(tonight & prev) / len(tonight))
        row = " ".join(f"k={k}:{statistics.mean(v):.2f}"
                       for k, v in sorted(by_k.items()))
        print(f"{year} ({len(leg)} shows): {row}")

    print("\n=== Union coverage: share of tonight in union of last N shows ===")
    for leg in tours(shows):
        year = leg[0]["date"][:4]
        by_n = defaultdict(list)
        for i, s in enumerate(leg):
            tonight = set(s["songs"])
            for n in range(1, 9):
                if i - n >= 0:
                    pool = set()
                    for j in range(i - n, i):
                        pool |= set(leg[j]["songs"])
                    by_n[n].append(len(tonight & pool) / len(tonight))
        row = " ".join(f"N={n}:{statistics.mean(v):.2f}"
                       for n, v in sorted(by_n.items()))
        print(f"{year}: {row}")


def hazard_curves(shows):
    """P(played | gap) split by song frequency tier."""
    print("=== Hazard by frequency tier (tier = tour plays/show) ===")
    events = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for leg in tours(shows):
        n = len(leg)
        rate = defaultdict(float)
        for s in leg:
            for song in set(s["songs"]):
                rate[song] += 1 / n
        last = {}
        for i, s in enumerate(leg):
            tonight = set(s["songs"])
            for song, li in last.items():
                gap = i - li
                r = rate[song]
                tier = ("core" if r >= 0.55 else
                        "regular" if r >= 0.3 else
                        "rotational" if r >= 0.12 else "rare")
                g = str(gap) if gap <= 4 else ("5-8" if gap <= 8 else "9+")
                cell = events[tier][g]
                cell[0] += 1 if song in tonight else 0
                cell[1] += 1
            for song in tonight:
                last[song] = i
    order = {"1": 1, "2": 2, "3": 3, "4": 4, "5-8": 5, "9+": 6}
    for tier in ["core", "regular", "rotational", "rare"]:
        cells = events[tier]
        row = " ".join(
            f"g={g}:{cells[g][0]/cells[g][1]:.2f}(n={cells[g][1]})"
            for g in sorted(cells, key=order.get))
        print(f"{tier:10s} {row}")


def exclusion_pairs(shows):
    """Song pairs that co-occur far less than independence predicts."""
    print("=== Same-night exclusion among common songs ===")
    for leg in tours(shows)[-2:]:
        year = leg[0]["date"][:4]
        n = len(leg)
        plays = defaultdict(int)
        both = defaultdict(int)
        for s in leg:
            songs = sorted(set(s["songs"]))
            for x in songs:
                plays[x] += 1
            for i, x in enumerate(songs):
                for y in songs[i + 1:]:
                    both[(x, y)] += 1
        common = [x for x, c in plays.items() if c >= n * 0.3]
        rows = []
        for i, x in enumerate(common):
            for y in common[i + 1:]:
                exp = plays[x] * plays[y] / n
                obs = both[(min(x, y), max(x, y))]
                if exp >= 3:
                    rows.append((obs / exp, x, y, obs, round(exp, 1)))
        rows.sort()
        print(f"--- {year} (most mutually exclusive) ---")
        for ratio, x, y, obs, exp in rows[:12]:
            print(f"  {ratio:.2f}  {x} + {y}  obs={obs} exp={exp}")


def venue_repeat(shows):
    """Does last year's setlist at the same venue depress songs?"""
    print("=== Venue-repeat effect ===")
    by_venue = defaultdict(list)
    for s in shows:
        by_venue[s["venue"]].append(s)
    ratios = []
    for venue, ss in by_venue.items():
        for i, s in enumerate(ss):
            for prev in ss[:i]:
                d1 = datetime.date.fromisoformat(prev["date"])
                d2 = datetime.date.fromisoformat(s["date"])
                if 200 < (d2 - d1).days < 500:  # same venue, prior year
                    tonight = set(s["songs"])
                    prev_set = set(prev["songs"])
                    ratios.append(len(tonight & prev_set) / len(tonight))
    if ratios:
        print(f"pairs={len(ratios)}  mean overlap with last year's set at "
              f"same venue: {statistics.mean(ratios):.2f}")


def weekday_effect(shows):
    print("=== Set length / hit share by weekday ===")
    hits = {"Ants Marching", "Crash Into Me", "Two Step", "Grey Street",
            "What Would You Say", "Tripping Billies", "Satellite"}
    by_dow = defaultdict(lambda: [0, 0, 0])
    for leg in tours(shows):
        for s in leg:
            dow = datetime.date.fromisoformat(s["date"]).strftime("%a")
            cell = by_dow[dow]
            cell[0] += len(s["songs"])
            cell[1] += len(set(s["songs"]) & hits)
            cell[2] += 1
    for dow in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
        if by_dow[dow][2]:
            n = by_dow[dow][2]
            print(f"{dow}: shows={n:3d} avg_len={by_dow[dow][0]/n:.1f} "
                  f"avg_hits={by_dow[dow][1]/n:.2f}")


def oracle_ceiling(shows):
    """Upper bound: score of a prophet knowing each song's true
    play-probability tonight (approximated by leave-one-out tour rate with
    gap adjustment is still a model; here we cheat maximally — rank songs by
    whether they were actually played, capped by in-tour plausibility is
    meaningless. Instead: expected hits of the best possible 20-song pick
    given per-show probabilities estimated from the whole tour including
    the target (in-sample).)"""
    print("=== In-sample oracle: sum of top-20 in-sample song rates ===")
    for leg in tours(shows)[-2:]:
        year = leg[0]["date"][:4]
        n = len(leg)
        rate = defaultdict(float)
        for s in leg:
            for song in set(s["songs"]):
                rate[song] += 1 / n
        top20 = sorted(rate.values(), reverse=True)[:20]
        # expected hits if every night you pick the season's top 20
        hits = []
        top_songs = {s for s, _ in
                     sorted(rate.items(), key=lambda kv: -kv[1])[:20]}
        for s in leg:
            hits.append(len(top_songs & set(s["songs"])))
        print(f"{year}: top-20 static in-sample pick: "
              f"mean hits {statistics.mean(hits):.1f} / 20 "
              f"(sum of top-20 rates: {sum(top20):.1f})")


ANALYSES = {
    "overlap": overlap_structure,
    "hazard": hazard_curves,
    "exclusion": exclusion_pairs,
    "venue": venue_repeat,
    "weekday": weekday_effect,
    "oracle": oracle_ceiling,
}

if __name__ == "__main__":
    shows = load_shows()
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in ANALYSES.items():
        if which in ("all", name):
            fn(shows)
            print()

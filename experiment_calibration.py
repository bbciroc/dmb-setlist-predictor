"""Memory-light gap-bucket calibration study.

Trains the hazard table ONCE on history before a target leg, then scores
each show of that leg causally (tour-rate/gap/own-period computed per
show). Reports, for the top-20 picks per show, predicted probability vs
actual hit rate by gap bucket — and evaluates correction multipliers
derived from the 2025 leg on the 2026 leg.

Usage: python3 experiment_calibration.py
"""

from __future__ import annotations

import statistics
from collections import defaultdict

import hazard_model
from predict import SMOOTH_SHOWS, load_shows, segments

CORR_2025 = None  # filled from the 2025 pass, applied to 2026


def bucket_of(gap):
    if gap is None:
        return "never"
    if gap <= 3:
        return str(gap)
    return "4-6" if gap <= 6 else "7+"


def fixed_table(train_legs, target_year):
    def stream():
        for li, leg in enumerate(train_legs):
            pr = defaultdict(float)
            prev = [s for lg in train_legs[:li] for s in lg][-90:]
            for s in prev:
                for song in set(s["songs"]):
                    pr[song] += 1 / max(len(prev), 1)
            pool = hazard_model.candidate_pool(leg, pr)
            w = max(0.6 ** (target_year - int(leg[0]["date"][:4])), 0.05)
            for ev in hazard_model.leg_events(leg, pr, pool):
                yield ev + (w,)
    return hazard_model.TableModel().fit(stream())


def score_leg(leg, prior_shows, table, corrections=None):
    """Yield (bucket, prob, hit) for each top-20 pick of each show i>=10."""
    pr = defaultdict(float)
    prior_win = prior_shows[-90:]
    for s in prior_win:
        for song in set(s["songs"]):
            pr[song] += 1 / len(prior_win)
    idx = defaultdict(list)
    for i, show in enumerate(leg):
        tonight = set(show["songs"])
        if i >= 10:
            cand = {}
            pool = set(idx) | {s for s, r in pr.items() if r >= 0.05}
            for song in pool:
                ii = idx[song]
                gap = i - ii[-1] if ii else None
                rate = (len(ii) + SMOOTH_SHOWS * pr[song]) / (i + SMOOTH_SHOWS)
                p = table.prob(rate, pr[song], gap,
                               hazard_model.gap_stats(ii))
                sel = p * (corrections or {}).get(bucket_of(gap), 1.0) \
                    if corrections else p
                cand[song] = (sel, p, bucket_of(gap))
        if i >= 10:
            top = sorted(cand.items(), key=lambda kv: -kv[1][0])[:20]
            for song, (_, p, b) in top:
                yield b, p, song in tonight
        for song in tonight:
            idx[song].append(i)


def report(name, rows):
    agg = defaultdict(lambda: [0, 0.0, 0])
    for b, p, hit in rows:
        agg[b][0] += 1
        agg[b][1] += p
        agg[b][2] += hit
    total_picks = sum(v[0] for v in agg.values())
    total_hits = sum(v[2] for v in agg.values())
    n_shows = total_picks / 20
    print(f"--- {name}: {total_hits / n_shows:.1f} hits/show ---")
    corr = {}
    for b in ["1", "2", "3", "4-6", "7+", "never"]:
        n, sp, h = agg[b]
        if n >= 20 and sp:
            corr[b] = (h / n) / (sp / n)
            print(f"  gap {b:>5}: picks={n:4d} pred={sp/n:.0%} "
                  f"hit={h/n:.0%} corr={corr[b]:.2f}")
    return corr


def main():
    shows = load_shows()
    legs = [l for l in segments(shows) if len(l) >= 10]
    leg26 = legs[-1]
    leg25 = legs[-2]
    i25 = legs.index(leg25)

    t25 = fixed_table(legs[:i25], int(leg25[0]["date"][:4]))
    prior25 = [s for lg in legs[:i25] for s in lg]
    corr = report("2025 leg (derive)", score_leg(leg25, prior25, t25))

    t26 = fixed_table(legs[:-1], int(leg26[0]["date"][:4]))
    prior26 = [s for lg in legs[:-1] for s in lg]
    report("2026 leg, uncorrected", score_leg(leg26, prior26, t26))
    capped = {b: min(max(c, 0.5), 1.6) for b, c in corr.items()}
    print("applying 2025-derived corrections:",
          {b: round(c, 2) for b, c in capped.items()})
    report("2026 leg, corrected", score_leg(leg26, prior26, t26, capped))


if __name__ == "__main__":
    main()

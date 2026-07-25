"""Predict the next Dave Matthews Band setlist from scraped history.

Scoring is tour-centric and empirical: every parameter (gap multipliers,
slot propensities, set lengths) is estimated from the data.

  score(song) = smoothed tour frequency x gap multiplier

- Tour frequency: plays this tour / shows this tour, smoothed toward the
  prior year's frequency so early-tour data isn't over-trusted.
- Gap multiplier: how much more/less likely a song is given it was last
  played k shows ago, estimated from (gap, played?) events pooled across
  recent tours and normalized by song frequency. DMB's famous reluctance to
  repeat a song on consecutive nights shows up as a low k=1 multiplier.
- Slots: opener / main-set closer / encore chosen by score x empirical slot
  propensity; the middle of the set is ordered by each song's average
  position percentile within main sets this tour.

Usage: python3 predict.py [--target-date YYYY-MM-DD] [--backtest N]
Writes data/prediction.json unless --backtest is given.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
from collections import defaultdict

ROOT = pathlib.Path(__file__).parent
SMOOTH_SHOWS = 3.0     # pseudo-shows pulling tour freq toward prior-year freq
MIN_GAP_EVENTS = 25    # min events per gap bucket before trusting it
HALF_LIFE = float("inf")   # recency decay disabled: backtests showed flat
                           # tour frequency predicts better (pool is stable)

def bucket(gap: int) -> str:
    if gap <= 3:
        return str(gap)
    if gap <= 6:
        return "4-6"
    if gap <= 12:
        return "7-12"
    return "13+"


def normalize(name: str) -> str:
    """Merge annotated variants: 'Too Much\xa0[fake]' -> 'Too Much'."""
    name = name.replace("\xa0", " ")
    return re.sub(r"\s*\[[^\]]*\]\s*$", "", name).strip()


def load_shows():
    shows = json.loads((ROOT / "data" / "shows.json").read_text())
    for s in shows:
        s["songs"] = [normalize(x) for x in s["songs"]]
    return [s for s in shows if s["songs"]]


def main_set(show) -> list:
    e = show["encore_from"]
    return show["songs"][:e] if e is not None else show["songs"]


def encore(show) -> list:
    e = show["encore_from"]
    return show["songs"][e:] if e is not None else []


def segments(shows, max_gap_days=21):
    """Split chronologically sorted shows into contiguous tour legs."""
    import datetime
    legs, leg = [], []
    prev = None
    for s in shows:
        d = datetime.date.fromisoformat(s["date"])
        if prev is not None and (d - prev).days > max_gap_days:
            legs.append(leg)
            leg = []
        leg.append(s)
        prev = d
    if leg:
        legs.append(leg)
    return legs


def gap_multipliers(legs) -> dict:
    """m[bucket] = P(played | gap in bucket) / P(played | song's own rate).

    Events are (gap since last play, played tonight?, song leg rate),
    collected within each tour leg so gaps never span an offseason.
    """
    events = defaultdict(list)  # bucket -> list of (played, rate)
    for year_shows in legs:
        n = len(year_shows)
        if n < 8:
            continue
        rate = defaultdict(float)
        for s in year_shows:
            for song in set(s["songs"]):
                rate[song] += 1.0 / n
        last_played = {}
        for i, s in enumerate(year_shows):
            tonight = set(s["songs"])
            for song, last_i in last_played.items():
                events[bucket(i - last_i)].append(
                    (1.0 if song in tonight else 0.0, rate[song]))
            for song in tonight:
                last_played[song] = i
    mult = {}
    for b, evs in events.items():
        if len(evs) < MIN_GAP_EVENTS:
            continue
        p_obs = statistics.mean(e[0] for e in evs)
        p_exp = statistics.mean(e[1] for e in evs)
        mult[b] = p_obs / p_exp if p_exp > 0 else 1.0
    return mult


def segue_rules(history, min_obs=8, coverage=0.6):
    """Songs with concentrated follower distributions in main sets.

    Returns {A: [B1, B2, ...]} — if A is played (non-final), it is followed
    by one of these songs in >= `coverage` of observations. Captures pairs
    like Anyone Seen the Bridge -> Too Much and Pantala Naga Pampa ->
    {Rapunzel, Pig}.
    """
    nxt = defaultdict(lambda: defaultdict(int))
    apps = defaultdict(int)
    for s in history:
        seq = main_set(s)
        for a, b in zip(seq, seq[1:]):
            nxt[a][b] += 1
            apps[a] += 1
    rules = {}
    for a, n in apps.items():
        if n < min_obs:
            continue
        followers = sorted(nxt[a].items(), key=lambda kv: -kv[1])[:3]
        cum, keep = 0, []
        for b, c in followers:
            if cum / n >= coverage:
                break
            keep.append(b)
            cum += c
        if cum / n >= coverage:
            rules[a] = keep
    return rules


def position_stats(tour, prior):
    """song -> median percentile position within main sets.

    Tour-only positions order sets markedly better than blending prior
    tours (backtested: avg rank corr 0.29 vs 0.12) — slotting habits are
    tour-specific. Prior tours only fill in songs unseen this tour.
    """
    def medians(shows_):
        vals = defaultdict(list)
        for s in shows_:
            ms = main_set(s)
            if len(ms) < 2:
                continue
            for i, song in enumerate(ms):
                vals[song].append(i / (len(ms) - 1))
        return {s: statistics.median(v) for s, v in vals.items()}

    pos = medians(prior)
    pos.update(medians(tour))
    return pos


def encore_position(history):
    """song -> (opens, closes, apps) within encores."""
    stats = defaultdict(lambda: [0, 0, 0])
    for s in history:
        e = encore(s)
        if not e:
            continue
        stats[e[0]][0] += 1
        stats[e[-1]][1] += 1
        for song in e:
            stats[song][2] += 1
    return stats


def apply_segues(order, rules, prob, protected):
    """Reorder/insert so each constrained song is followed by a valid
    follower. Returns (new_order, inserted_count)."""
    order = list(order)
    inserted = 0
    i = 0
    while i < len(order) - 1:  # last main-set song has no follower
        a = order[i]
        followers = rules.get(a)
        if followers and order[i + 1] not in followers:
            present = [b for b in followers
                       if b in order[i + 1:] and b not in protected]
            if present:
                b = max(present, key=lambda x: prob.get(x, 0))
                order.remove(b)
                order.insert(i + 1, b)
            else:
                order.insert(i + 1, followers[0])
                inserted += 1
        i += 1
    # trim inserted overflow: drop lowest-prob unconstrained, unprotected
    while inserted > 0:
        constrained = set(rules) | {b for bs in rules.values() for b in bs}
        free = [x for x in order[1:-1]
                if x not in constrained and x not in protected]
        if not free:
            break
        order.remove(min(free, key=lambda x: prob.get(x, 0)))
        inserted -= 1
    return order


def predict(shows, target_date: str) -> dict:
    history = [s for s in shows if s["date"] < target_date]
    if not history:
        raise SystemExit("no shows before target date")
    legs = segments(history)
    tour = legs[-1]          # the current contiguous tour leg
    tour_ids = {id(s) for s in tour}
    prior = [s for s in history if id(s) not in tour_ids]
    n_tour, n_prior = len(tour), max(len(prior), 1)

    plays_tour = defaultdict(int)
    plays_prior = defaultdict(int)
    weighted = defaultdict(float)   # recency-weighted tour plays
    weight_total = 0.0
    for i, s in enumerate(tour):
        w = 0.5 ** ((n_tour - 1 - i) / HALF_LIFE)
        weight_total += w
        for song in set(s["songs"]):
            plays_tour[song] += 1
            weighted[song] += w
    for s in prior:
        for song in set(s["songs"]):
            plays_prior[song] += 1

    songs = set(plays_tour) | set(plays_prior)
    freq_prior = {s: plays_prior[s] / n_prior for s in songs}
    base = {s: (weighted[s] + SMOOTH_SHOWS * freq_prior[s])
               / (weight_total + SMOOTH_SHOWS) for s in songs}

    mult = gap_multipliers(legs)

    last_played = {}
    for i, s in enumerate(tour):
        for song in set(s["songs"]):
            last_played[song] = i
    never_this_tour_mult = min(mult.values()) if mult else 0.3

    prob = {}
    for song in songs:
        if song in last_played:
            g = bucket(n_tour - last_played[song])
            m = mult.get(g, 1.0)
        else:
            m = never_this_tour_mult
        prob[song] = min(0.99, base[song] * m)

    # slot propensities (tour-weighted, prior year as light backfill)
    def slot_counts(extract):
        c = defaultdict(float)
        for s in tour:
            for song in extract(s):
                c[song] += 1.0
        for s in prior:
            for song in extract(s):
                c[song] += 0.25
        return c

    openers = slot_counts(lambda s: s["songs"][:1])
    closers = slot_counts(lambda s: main_set(s)[-1:])
    encores = slot_counts(encore)
    total_plays = slot_counts(lambda s: set(s["songs"]))
    enc_prop = {s: encores.get(s, 0.0) / total_plays[s]
                for s in total_plays}

    pos = position_stats(tour, prior)
    enc_pos = encore_position(history)
    rules = segue_rules(history)

    n_main = int(statistics.median(len(main_set(s)) for s in tour))
    n_enc = int(statistics.median(len(encore(s)) for s in tour)) or 1

    chosen = set()

    def take(pool_score, n):
        picks = []
        for song, _ in sorted(pool_score.items(), key=lambda kv: -kv[1]):
            if song not in chosen and len(picks) < n:
                picks.append(song)
                chosen.add(song)
        return picks

    opener = take({s: prob[s] * openers[s] for s in openers}, 1)
    closer = take({s: prob[s] * closers[s] for s in closers}, 1)
    enc_songs = take({s: prob[s] * enc_prop.get(s, 0.0) ** 0.5 * encores[s]
                      for s in encores}, n_enc)
    middle = take(prob, n_main - 2)
    middle.sort(key=lambda s: pos.get(s, 0.5))

    # encore runs in its own order: openers (by open share) before closers
    def enc_key(song):
        opens, closes, apps = enc_pos.get(song, (0, 0, 0))
        return (closes - opens) / apps if apps else 0.0
    enc_songs.sort(key=enc_key)

    main_order = apply_segues(opener + middle + closer, rules, prob,
                              protected={*opener, *closer})
    middle = [s for s in main_order if s not in (*opener, *closer)]

    setlist = ([{"song": s, "slot": "opener"} for s in opener]
               + [{"song": s, "slot": "main"} for s in middle]
               + [{"song": s, "slot": "closer"} for s in closer]
               + [{"song": s, "slot": "encore"} for s in enc_songs])
    for item in setlist:
        item["prob"] = round(prob.get(item["song"], 0.0), 3)
        item["plays_tour"] = plays_tour[item["song"]]
        li = last_played.get(item["song"])
        item["shows_since_played"] = None if li is None else n_tour - li

    bubble = [{"song": s, "prob": round(prob[s], 3),
               "plays_tour": plays_tour[s]}
              for s, _ in sorted(prob.items(), key=lambda kv: -kv[1])
              if s not in chosen][:10]

    return {"target_date": target_date,
            "n_tour_shows": n_tour, "n_prior_shows": len(prior),
            "gap_multipliers": {k: round(v, 3) for k, v in mult.items()},
            "set_length": {"main": n_main, "encore": n_enc},
            "setlist": setlist, "bubble": bubble}


def backtest(shows, n_holdout: int) -> None:
    targets = segments(shows)[-1][-n_holdout:]
    hits_model, hits_base, total = 0, 0, 0
    order_corrs = []
    for t in targets:
        pred = predict(shows, t["date"])
        predicted = {x["song"] for x in pred["setlist"]}
        actual = set(t["songs"])
        # naive baseline: most-played N songs of the tour leg so far
        hist = segments([s for s in shows if s["date"] < t["date"]])[-1]
        counts = defaultdict(int)
        for s in hist:
            for song in set(s["songs"]):
                counts[song] += 1
        naive = {s for s, _ in sorted(counts.items(), key=lambda kv: -kv[1])
                 [:len(predicted)]}
        hm = len(predicted & actual)
        hb = len(naive & actual)
        hits_model += hm
        hits_base += hb
        total += len(actual)

        # ordering quality: Spearman rank corr on correctly predicted songs
        pred_order = [x["song"] for x in pred["setlist"]]
        act_order = list(dict.fromkeys(t["songs"]))
        common = [s for s in pred_order if s in actual and s in act_order]
        corr = None
        if len(common) >= 3:
            pr = {s: i for i, s in enumerate(common)}
            ar = sorted(range(len(common)),
                        key=lambda i: act_order.index(common[i]))
            n = len(common)
            d2 = sum((ar.index(i) - i) ** 2 for i in range(n))
            corr = 1 - 6 * d2 / (n * (n * n - 1))
            order_corrs.append(corr)
        print(f'{t["date"]} {t["venue"][:30]:30s} actual={len(actual):2d} '
              f'model={hm:2d} naive={hb:2d} '
              f'order_corr={"-" if corr is None else f"{corr:.2f}"}')
    print(f"model hit rate: {hits_model/total:.1%}  "
          f"naive baseline: {hits_base/total:.1%}  "
          f"avg order corr: {statistics.mean(order_corrs):.2f}"
          if order_corrs else "no order data")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-date", default="2026-07-25")
    ap.add_argument("--backtest", type=int, default=0)
    args = ap.parse_args()
    shows = load_shows()
    if args.backtest:
        backtest(shows, args.backtest)
    else:
        result = predict(shows, args.target_date)
        out = ROOT / "data" / "prediction.json"
        out.write_text(json.dumps(result, indent=1))
        print(json.dumps(result, indent=1))

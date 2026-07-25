# DMB Setlist Prediction — Research Report

Data: 322 full-band shows, 2021–2026, scraped from dmbalmanac.com.
Evaluation: rolling backtest — every prediction uses only shows played
before the target night; 20 songs predicted per show.

## Signals tested

| Signal | Verdict | Evidence |
|---|---|---|
| Tour frequency | strong (baseline) | naive top-20 ≈ 27% hit rate |
| Gap since last played | strong | global multipliers ×0.43 (gap 1) → ×1.34 (gap 4–6) |
| Frequency-tier × gap interaction | strong | core songs repeat back-to-back at 77%, regular songs at 20% — a global back-to-back penalty is wrong |
| Song's own period (gap ÷ own mean gap) | real | regular-cycle songs (gap sd < 1) at 1.2–1.6× their period play at 51% vs 30% for loose songs |
| Recency-weighted frequency (half-life) | HURTS | 34–38% falling as decay strengthens; tour pools are stable |
| Segue chains (PNP→Rapunzel, ASTB→Too Much) | real (ordering) | ASTB→Too Much 100% after name normalization |
| Encore grammar | near-deterministic | Peace on Earth opens 16/16 encores; Two Step closes 15/15 |
| Position-in-set (tour-only median) | real (ordering) | rank corr 0.29 vs 0.12 when blending prior tours |
| Same-night exclusion pairs (slot rivals) | real but NOISY | Ants+Tripping Billies obs/exp ≈ 0.26, but using it in selection cost 2 pts (small samples) |
| Venue-repeat suppression (last year's set) | weak | overlap 0.18 vs ~0.25 baseline |
| Weekday | none | no stable pattern |
| Setlist templates (nearest-neighbor) | none | best historical neighbor shares only 45% on average, with hindsight |

## Model evolution (20-show rolling backtest, 2026 tour)

1. Naive most-played-20: **27.6%** (≈5.5/20)
2. Freq × global gap multiplier + slots/segues: **36.7%** (≈7.3/20)
3. Logistic regression (rate, gap buckets, rate×gap1, prior rate): 35.7%
4. 2-D empirical hazard table (tier × gap): 36.2%
5. + own-period relative-gap cells: ~35–37%
6. + membership-first assembly (top-20 by probability IS the pick;
   slots assigned within it): **37.0%** hits, order corr 0.25 → **0.39**
7. Ensemble of table + multiplier models: no change (7.57 = 7.57) — the
   engines agree on picks; they are extracting the same signal.

All engines converge at ≈7–7.6 of 20. The differences between them are
within backtest noise; the constraint is the process, not the model.

## The ceiling

DMB's nightly set is ~20 songs drawn from an active pool of 80–100 with
heavy rotation. Measured entropy bounds:

- **Union of the last 8 shows covers only 74%** of tonight's songs (2026).
  A predictor restricted to the recent pool caps at ~15/20 even if it
  picked perfectly within it.
- **Calibration**: predicted probabilities track observed frequencies
  (0.2-bucket → 0.31, 0.4 → 0.39, 0.6 → 0.58); expected hits 6.2 vs actual
  7.3 — the model is honest, slightly conservative.
- **In-sample oracle**: a hazard table trained *on the target shows
  themselves* (full leakage) averages **7.9/20, best single night 10**.
- P(≥16 hits) for 20 picks at realistic per-song probabilities is ~10⁻⁸.

**Conclusion: ~8/20 average (10–11 on a good night) is the information
ceiling of historical setlist data. 16/20 would require non-public
information (soundcheck lists, printed setlists) — no statistical model of
past shows can get there, including one allowed to cheat.**

## What would move the needle

- Soundcheck reports (antsmarching forum threads, night-of) — strongly
  correlated with the printed setlist.
- Tour-opener effects after breaks, album-cycle debuts (marginal).
- Live-updating during the show (first songs known → conditional resample).

## Training-depth experiment (2015 vs 2018 vs 2021 cutoff)

Same 15-show rolling backtest on the 2026 leg, training history cut at
three depths:

| Cutoff | Training shows | Hit rate |
|---|---|---|
| 2021 | 322 | 36.2% |
| 2018 | 474 | 35.6% |
| 2015 | 653 | 36.2% |

Depth is a wash: the hazard shape (how core/regular/rare songs cycle) is
stable across eras and saturates with ~5 years of data, while the features
that drive picks are tour-local. One side effect mattered: un-windowed
slot pools let 11 years of history outvote the current tour on
opener/closer/encore choices — fixed by capping all prior windows at the
most recent 90 shows, making predictions invariant to scrape depth. The
full 2015–2026 dataset is kept for analyses.

## Encore two-role grammar (user-spotted, verified)

Recent encores are structurally [slow/solo opener -> full-band closer]:
88 of ~123 multi-song encores 2024-2026 are exactly that pattern, and
nearly all others still open with an O-type and close with a C-type.
Role pools: openers = Peace on Earth (16), Some Devil (15), Rye Whiskey
(14), Just Breathe, Sister; closers = Watchtower (16), Two Step (15),
Grey Street (9), The Last Stop, Ants Marching, Crush. Encore construction
now picks one song per role (prob x role count, windowed to ~90 shows).
Effect: backtest 36.2% -> 36.9%, order corr 0.24 -> 0.44, and encores
always contain a closer-type banger.

## Back-to-back bug + era drift (user-spotted)

Warehouse and #41 were predicted the night after they played. Two causes:
(1) the relative-gap bucket absorbed gap-1 events, diluting the taboo —
gap 1 now always uses the absolute cell; (2) the table trained on
2015-2019 data where back-to-back repeats were normal (5-7/night,
promo-cycle songs like Samurai Cop repeated 39x in 2018) while 2026 is at
0.8/night (nearly all interludes: PNP, Too Much post-ASTB). Training
events now carry era-decay weights (0.6^years-ago), so today's no-repeat
culture sets today's cells. Backtest unchanged (36.9%); predictions no
longer include last night's songs.

## Second-stage gap calibration (user-spotted overpricing)

The raw table overprices gap-2 picks and underprices the 3-6 "due" zone —
consistent in BOTH tours (2025: pred 37% vs hit 24% at gap 2; 2026: 32%
vs 27%). Correction multipliers derived from the 2025 leg only
(gap-2 x0.65, gap-3 x1.10, 4-6 x1.05, 7+ x0.72, gap-1 x0.5) and validated
out-of-sample on 2026: 7.6 -> 7.9 hits/show. Full-pipeline backtest:
37.2% -> 38.1% (naive 26.9%). Derivation: experiment_calibration.py
(single-fit, memory-light).

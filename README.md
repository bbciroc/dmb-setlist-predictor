# DMB Setlist Predictor

Predicts the next Dave Matthews Band setlist from scraped show history.

## Usage

```bash
python3 scraper.py                     # scrape dmbalmanac.com -> data/shows.json (cached)
python3 predict.py                     # predict next show -> data/prediction.json
python3 predict.py --target-date 2026-07-29   # predict a specific date
python3 predict.py --backtest 10       # hold-out accuracy vs naive baseline
python3 build_site.py                  # render site/index.html
cd site && python3 -m http.server 8742 # serve the site
```

No dependencies beyond Python 3 stdlib and curl.

## How the model works

Each song's play probability comes from an empirical hazard table
(`hazard_model.py`) trained causally on 653 shows, 2015-2026 (windowed priors: last ~90 prior shows drive rates and slots; depth beyond that measured neutral) — full
analysis in `docs/analysis-report.md`:

- **P(played | frequency tier, gap)** with learned interactions: core
  songs repeat back-to-back 77% of the time, regular rotation songs 20%.
- **Own-period timing**: songs with a stable personal cycle (gap sd < 1)
  peak at 51% probability when 1.2-1.6x overdue vs their own mean gap.
- **Tour legs** detected automatically (>21-day break splits a leg);
  smoothing toward prior-tour rates handles early-tour uncertainty and
  deep-cut returns.
- **Membership-first assembly**: the top-18 main-set songs by probability
  are the pick; encore slots reserved for true encore-propensity songs.
- **Slots** — opener / main-set closer / encore picked by score x empirical
  slot frequency; middle of the set ordered by median position percentile
  from *this tour only* (backtested better than blending prior tours);
  set length = tour median.
- **Segue rules** — songs whose follower distribution is concentrated
  (>=60% coverage in <=3 followers, >=8 observations) must be followed by
  one of those followers (Anyone Seen the Bridge -> Too Much, Pantala Naga
  Pampa -> Rapunzel/Pig); the follower is inserted if not already selected.
- **Encore grammar** — encore order = encore-opener vs encore-closer
  propensity (Peace on Earth opens 16/16 of its encores; Two Step closes
  15/15).
- Song names are normalized ("Too Much [fake]" -> "Too Much").

Rolling backtest (last 20 shows of the 2026 summer tour, each predicted
from prior history only): 37% of played songs predicted vs 27% for a
most-played-songs baseline; mean Spearman rank correlation of predicted vs
actual order 0.33-0.39. For scale, an oracle trained on the target shows
themselves averages just 40% (7.9/20) — DMB's nightly draw from a ~90-song
active pool caps ANY statistical predictor near 8/20 (see
`docs/analysis-report.md` for the ceiling evidence).

Data: dmbalmanac.com (setlist.fm and antsmarching.org sit behind bot
challenges; dmbalmanac is plainly fetchable). Scraped pages are cached in
`data/cache/` — delete a year page to pick up new shows.

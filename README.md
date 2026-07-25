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

Each song is scored `smoothed tour frequency x gap multiplier`:

- **Tour frequency** — share of shows on the current tour leg featuring the
  song, smoothed toward prior-tour frequency (3 pseudo-shows). Tour legs are
  detected automatically (>21-day break splits a leg).
- **Gap multiplier** — estimated from history: how the chance of playing a
  song changes with how many shows ago it was last played. On current data:
  x0.43 if played last night (the "never back-to-back" rotation), rising to
  x1.34 when rested 4-6 shows.
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

Backtest (last 10 shows of the 2026 summer tour): 37% of played songs
predicted vs 29% for a most-played-songs baseline; mean Spearman rank
correlation of predicted vs actual order 0.28. DMB rotates ~170 songs a
tour, so those gaps are meaningful.

Data: dmbalmanac.com (setlist.fm and antsmarching.org sit behind bot
challenges; dmbalmanac is plainly fetchable). Scraped pages are cached in
`data/cache/` — delete a year page to pick up new shows.

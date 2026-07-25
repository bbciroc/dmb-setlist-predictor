"""Render site/index.html from data/prediction.json."""

from __future__ import annotations

import html
import json
import pathlib

ROOT = pathlib.Path(__file__).parent

SLOT_LABELS = {"opener": "OPENER", "closer": "SET CLOSER", "encore": "ENCORE"}


def row(item, i):
    name = html.escape(item["song"])
    pct = round(item["prob"] * 100)
    slot = item["slot"]
    tag = (f'<span class="tag {slot}">{SLOT_LABELS[slot]}</span>'
           if slot in SLOT_LABELS else "")
    since = item.get("shows_since_played")
    since_txt = ("never this tour" if since is None
                 else "played last show" if since == 1
                 else f"{since} shows ago")
    return f"""<li>
  <span class="num">{i}</span>
  <span class="song">{name} {tag}</span>
  <span class="meta">{item["plays_tour"]} plays this tour &middot; last: {since_txt}</span>
  <span class="bar"><span style="width:{pct}%"></span></span>
  <span class="pct">{pct}%</span>
</li>"""


def main() -> None:
    pred = json.loads((ROOT / "data" / "prediction.json").read_text())
    shows = json.loads((ROOT / "data" / "shows.json").read_text())
    played = [s for s in shows if s["songs"]]
    last = [s for s in played if s["date"] < pred["target_date"]][-1]
    tonight = next((s for s in shows if s["date"] == pred["target_date"]),
                   None)
    where = (f'{tonight["venue"]}, {tonight["city"]}' if tonight
             else "next show")

    main_rows = [row(x, i + 1) for i, x in enumerate(pred["setlist"])
                 if x["slot"] != "encore"]
    enc_rows = [row(x, i + 1) for i, x in enumerate(
        [x for x in pred["setlist"] if x["slot"] == "encore"])]
    bubble = "".join(
        f'<li><span class="song">{html.escape(b["song"])}</span>'
        f'<span class="meta">{b["plays_tour"]} plays this tour</span>'
        f'<span class="pct">{round(b["prob"] * 100)}%</span></li>'
        for b in pred["bubble"])

    gaps = pred["gap_multipliers"]
    gap_txt = " &middot; ".join(
        f"{k} show{'s' if k != '1' else ''} ago: &times;{v}"
        for k, v in sorted(gaps.items(),
                           key=lambda kv: int(kv[0].split("-")[0]
                                              .rstrip("+"))))

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DMB Setlist Predictor &mdash; {pred["target_date"]}</title>
<style>
:root {{ color-scheme: dark; }}
* {{ box-sizing: border-box; margin: 0; }}
body {{ font: 16px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif;
  background: #14161a; color: #e8e6e1; max-width: 860px;
  margin: 0 auto; padding: 2rem 1rem 4rem; }}
h1 {{ font-size: 1.6rem; }} h2 {{ font-size: 1.1rem; margin: 2rem 0 .6rem;
  color: #f0b429; text-transform: uppercase; letter-spacing: .08em; }}
.sub {{ color: #9aa0a8; margin-bottom: 1.5rem; }}
ol, ul {{ list-style: none; padding: 0; }}
li {{ display: grid; grid-template-columns: 2rem 1fr 9rem 3rem;
  grid-template-areas: "num song bar pct" "num meta bar pct";
  gap: 0 .75rem; align-items: center; padding: .45rem .6rem;
  border-bottom: 1px solid #23262c; }}
.num {{ grid-area: num; color: #6b7280; text-align: right; }}
.song {{ grid-area: song; font-weight: 600; }}
.meta {{ grid-area: meta; color: #8a919b; font-size: .78rem; }}
.bar {{ grid-area: bar; background: #23262c; border-radius: 4px;
  height: 8px; overflow: hidden; }}
.bar span {{ display: block; height: 100%; background: #4f8fda; }}
.pct {{ grid-area: pct; text-align: right; font-variant-numeric: tabular-nums;
  color: #c8ccd2; }}
.tag {{ font-size: .65rem; padding: .1rem .4rem; border-radius: 3px;
  vertical-align: middle; letter-spacing: .05em; }}
.tag.opener {{ background: #2d5a2d; }} .tag.closer {{ background: #5a2d2d; }}
.tag.encore {{ background: #4a3a6b; }}
.method {{ color: #9aa0a8; font-size: .88rem; }}
.method p {{ margin: .5rem 0; }}
code {{ background: #23262c; padding: .1rem .3rem; border-radius: 3px; }}
</style></head><body>
<h1>Dave Matthews Band &mdash; Predicted Setlist</h1>
<p class="sub">{html.escape(where)} &middot;
{pred["target_date"]} &middot; predicted from {pred["n_tour_shows"]}
shows this tour + {pred["n_prior_shows"]} prior shows &middot;
last show: {last["date"]}, {last["venue"]}, {last["city"]}</p>

<h2>Main Set</h2>
<ol>{"".join(main_rows)}</ol>

<h2>Encore</h2>
<ol>{"".join(enc_rows)}</ol>

<h2>On the Bubble</h2>
<ul>{bubble}</ul>

<h2>How it works</h2>
<div class="method">
<p>Every song is scored by <code>smoothed tour frequency &times; gap
multiplier</code>. Tour frequency is how often the song has appeared this
tour, smoothed toward last tour's rate. The gap multiplier is estimated from
the data &mdash; how DMB's chance of playing a song changes with how recently
it was played: {gap_txt}. The famous &ldquo;almost never
back-to-back&rdquo; rotation shows up as the low multiplier at gap&nbsp;1.</p>
<p>Opener, set closer, and encore are chosen from empirical slot pools
(score &times; how often the song appears in that slot). The middle of the
set is ordered by each song's median position within main sets
<em>this tour</em> (tour-only positions order sets twice as well as blending
older tours &mdash; slotting habits are tour-specific). Set length is the
tour median ({pred["set_length"]["main"]} main set songs,
{pred["set_length"]["encore"]} encore).</p>
<p>Two sequencing models learned from the data: <b>segue rules</b> &mdash;
songs whose next song is concentrated in a few followers (e.g. Anyone Seen
the Bridge &rarr; Too Much 100%, Pantala Naga Pampa &rarr; Rapunzel/Pig)
must be followed by one of them, inserting it if needed; and <b>encore
grammar</b> &mdash; encore order comes from encore-opener vs encore-closer
propensity (Peace on Earth has opened all 16 of its encores; Two Step and
Watchtower close).</p>
<p>Percentages are the model's estimate that the song is played tonight
(not that it lands in that exact slot). Backtested on the last 10 shows of
this tour: 37% of played songs predicted (vs 29% for a naive
most-played-songs baseline), average rank correlation of song order 0.28.
Data: dmbalmanac.com.</p>
</div>
</body></html>"""

    site = ROOT / "site"
    site.mkdir(exist_ok=True)
    (site / "index.html").write_text(page)
    print(f"wrote {site / 'index.html'}")


if __name__ == "__main__":
    main()

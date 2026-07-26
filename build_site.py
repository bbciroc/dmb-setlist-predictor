"""Render site/index.html and site/sheet.html from data/prediction.json.

Design: amphitheater-at-dusk. A CC-licensed live photo fades into the
night-black page; headers are hand-written (Permanent Marker, inlined);
the setlist reads like a printed cue sheet with hairline amber probability
lines. Fire-dancer red is reserved for the encore.
"""

from __future__ import annotations

import html
import json
import pathlib

ROOT = pathlib.Path(__file__).parent

BG = "#0e0c10"          # night stage
CREAM = "#f3e9d6"       # house lights
AMBER = "#e8a33c"       # tungsten wash
RED = "#c14434"         # fire dancer (encore only)
HAZE = "#8b93a3"        # dusk haze (meta)
LINE = "#2a2733"        # hairlines

PHOTO_CREDIT = ('Photo: Jake Waage, Bonnaroo 2010, '
                '<a href="https://commons.wikimedia.org/wiki/'
                'File:Dave-matthews-band-bonnaroo-2010.jpg">CC BY-SA 3.0</a>')


def base_css(marker_b64: str) -> str:
    return f"""
@font-face {{
  font-family: 'Marker';
  src: url(data:font/woff2;base64,{marker_b64}) format('woff2');
}}
* {{ box-sizing: border-box; margin: 0; }}
:root {{ color-scheme: dark; }}
body {{
  background: {BG}; color: {CREAM};
  font: 15px/1.55 -apple-system, "Segoe UI", Roboto, sans-serif;
}}
a {{ color: {AMBER}; }}
a:focus-visible, button:focus-visible {{
  outline: 2px solid {AMBER}; outline-offset: 3px;
}}
.marker {{ font-family: Marker, "Marker Felt", cursive; font-weight: 400; }}
.mono {{
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
  font-variant-numeric: tabular-nums;
}}
"""


def row(item, i):
    name = html.escape(item["song"])
    pct = round(item["prob"] * 100)
    slot = item["slot"]
    note = {"opener": "opener", "closer": "set closer"}.get(slot, "")
    if item.get("deep_cut"):
        note = "deep cut"
    note_html = f' <span class="slot">{note}</span>' if note else ""
    since = item.get("shows_since_played")
    since_txt = ("first of the tour" if since is None
                 else f"rested {since} show{'s' if since != 1 else ''}")
    cls = ' class="enc"' if slot == "encore" else ""
    return f"""<li{cls}>
  <span class="num mono">{i:02d}</span>
  <span class="song">{name}{note_html}
    <span class="meta mono">{item["plays_tour"]}x this tour &middot; {since_txt}</span>
  </span>
  <span class="odds mono"><i style="width:{pct}%"></i><b>{pct}</b></span>
</li>"""


def main() -> None:
    pred = json.loads((ROOT / "data" / "prediction.json").read_text())
    shows = json.loads((ROOT / "data" / "shows.json").read_text())
    marker_b64 = (ROOT / "assets" / "marker.b64").read_text().strip()
    played = [s for s in shows if s["songs"]]
    last = [s for s in played if s["date"] < pred["target_date"]][-1]
    tonight = next((s for s in shows if s["date"] == pred["target_date"]),
                   None)
    if tonight is None:
        sched_path = ROOT / "data" / "schedule.json"
        if sched_path.exists():
            tonight = next((s for s in json.loads(sched_path.read_text())
                            if s["date"] == pred["target_date"]), None)
    where = (f'{tonight["venue"]}, {tonight["city"]}' if tonight
             else "next show")
    city = (tonight["city"] if tonight and tonight["city"] else "TONIGHT")

    main_items = [x for x in pred["setlist"] if x["slot"] != "encore"]
    enc_items = [x for x in pred["setlist"] if x["slot"] == "encore"]
    main_rows = "".join(row(x, i + 1) for i, x in enumerate(main_items))
    enc_rows = "".join(row(x, len(main_items) + 1 + i)
                       for i, x in enumerate(enc_items))
    bubble = "".join(
        f'<li><span class="song">{html.escape(b["song"])}'
        f'<span class="meta mono">{b["plays_tour"]}x this tour</span></span>'
        f'<span class="odds mono"><i style="width:{round(b["prob"]*100)}%">'
        f'</i><b>{round(b["prob"]*100)}</b></span></li>'
        for b in pred["bubble"])

    index = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DMB Setlist Predictor &mdash; {pred["target_date"]}</title>
<meta property="og:title" content="DMB Predicted Setlist &mdash; {pred["target_date"]}">
<meta property="og:description" content="{html.escape(where)} &middot; rotation-model prediction from {pred["n_tour_shows"]} shows this tour">
<meta property="og:image" content="https://bbciroc.github.io/dmb-setlist-predictor/sheet.png">
<meta name="twitter:card" content="summary_large_image">
<style>
{base_css(marker_b64)}
.hero {{
  position: relative; min-height: 46vh;
  background: url(bg.jpg) center 68% / cover no-repeat {BG};
  display: flex; flex-direction: column; justify-content: flex-end;
}}
.hero::after {{
  content: ""; position: absolute; inset: 0;
  background: linear-gradient(180deg,
    rgba(14,12,16,.5) 0%, rgba(14,12,16,.15) 40%,
    rgba(14,12,16,.82) 78%, {BG} 100%);
}}
.hero-inner {{
  position: relative; z-index: 1; width: min(680px, 92vw);
  margin: 0 auto; padding: 3.5rem 0 1.2rem;
}}
h1 {{ font-size: clamp(2rem, 6vw, 3.1rem); letter-spacing: .02em;
     text-shadow: 0 2px 18px rgba(0,0,0,.8); }}
.show-line {{ color: {CREAM}; opacity: .85; font-size: .92rem;
  margin-top: .35rem; text-shadow: 0 1px 8px rgba(0,0,0,.9); }}
.wrap {{ width: min(680px, 92vw); margin: 0 auto; padding-bottom: 4rem; }}
.sheet-link {{
  display: inline-block; margin: 1.1rem 0 0; padding: .5rem .9rem;
  border: 1px solid {AMBER}; color: {AMBER}; text-decoration: none;
  font-size: .85rem; letter-spacing: .04em;
}}
.sheet-link:hover {{ background: {AMBER}; color: {BG}; }}
h2 {{ font-size: 1.35rem; margin: 2.4rem 0 .5rem; color: {AMBER}; }}
h2.enc-h {{ color: {RED}; }}
ol, ul {{ list-style: none; padding: 0; }}
li {{
  display: grid; grid-template-columns: 2.2rem 1fr 8.5rem;
  align-items: center; column-gap: .8rem;
  padding: .52rem 0; border-bottom: 1px solid {LINE};
}}
.num {{ color: {HAZE}; font-size: .78rem; text-align: right; }}
.song {{ font-weight: 600; font-size: 1.02rem; }}
.slot {{ color: {AMBER}; font-size: .72rem; font-weight: 400;
  letter-spacing: .08em; text-transform: uppercase; margin-left: .4rem; }}
li.enc .slot, li.enc .num {{ color: {RED}; }}
.meta {{ display: block; color: {HAZE}; font-size: .72rem;
  font-weight: 400; margin-top: .1rem; }}
.odds {{ display: flex; align-items: center; gap: .55rem;
  justify-self: end; width: 100%; }}
.odds i {{ display: block; height: 2px; background: {AMBER};
  margin-left: auto; }}
li.enc .odds i {{ background: {RED}; }}
.odds b {{ font-weight: 500; font-size: .85rem; color: {CREAM};
  min-width: 2ch; text-align: right; }}
.odds b::after {{ content: "%"; color: {HAZE}; font-size: .7em; }}
ul li {{ grid-template-columns: 1fr 8.5rem; }}
.method {{ color: {HAZE}; font-size: .85rem; margin-top: .8rem; }}
.method p {{ margin: .55rem 0; }}
.method b {{ color: {CREAM}; font-weight: 600; }}
footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid {LINE};
  color: {HAZE}; font-size: .74rem; }}
footer a {{ color: {HAZE}; }}
</style></head><body>

<div class="hero">
  <div class="hero-inner">
    <h1 class="marker">DMB &mdash; {html.escape(city.upper())}</h1>
    <p class="show-line">{html.escape(where)} &middot; {pred["target_date"]}
    &middot; predicted from {pred["n_tour_shows"]} shows this tour &middot;
    last show {last["date"]}, {html.escape(last["venue"])}</p>
  </div>
</div>

<div class="wrap">
<a class="sheet-link" href="sheet.html">The stage sheet &rarr;</a>

<h2 class="marker">Main set</h2>
<ol>{main_rows}</ol>

<h2 class="marker enc-h">Encore</h2>
<ol>{enc_rows}</ol>

<h2 class="marker">On the bubble</h2>
<ul>{bubble}</ul>

<h2 class="marker">How it works</h2>
<div class="method">
<p>Every song's probability comes from an empirical hazard model trained on
653 shows (2015&ndash;2026): how likely a song is given how often it's
played this tour, how many shows it has rested, and its own rotation cycle.
The famous rules fall out of the data &mdash; almost never back-to-back
(&times;0.5), most likely when a song is 1.2&ndash;1.6&times; overdue on its
own cycle (up to 51%).</p>
<p>The 20 most probable songs are the pick; opener, set closer, and encore
are assigned within them from slot history. The encore follows the grammar
the band actually uses: a quiet solo opener into a full-band closer
(Peace on Earth has opened 16 of its 16 encores). Segue pairs
(Pantala Naga Pampa &rarr; Rapunzel, Big Eyed Fish &rarr; Bartender) are
learned and enforced when both halves make the cut.</p>
<p><b>Track record:</b> rolling backtest over the last 20 shows &mdash; 38%
of the songs played each night were on this page beforehand, vs 27% for
guessing the tour's most-played songs. A model allowed to cheat with the
answers only reaches 40%: rotation entropy caps anyone near 8 of 20.</p>
<p>Data: <a href="https://dmbalmanac.com">dmbalmanac.com</a>. Refreshes
daily at 10am ET after setlists post.</p>
</div>

<footer>{PHOTO_CREDIT} &middot; fan project, not affiliated with the band
</footer>
</div>
</body></html>"""

    sheet = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>The Stage Sheet &mdash; {pred["target_date"]}</title>
<style>
{base_css(marker_b64)}
body {{ display: flex; flex-direction: column; align-items: center;
  min-height: 100vh; padding: 2rem 1rem 3rem; }}
img {{ width: min(460px, 94vw); height: auto;
  box-shadow: 0 30px 80px rgba(0,0,0,.7); }}
.actions {{ display: flex; gap: 1rem; align-items: center;
  margin: 1.6rem 0 0; }}
.dl {{
  background: {AMBER}; color: {BG}; text-decoration: none;
  padding: .65rem 1.2rem; font-weight: 600; font-size: .9rem;
  letter-spacing: .03em;
}}
.dl:hover {{ filter: brightness(1.1); }}
.back {{ color: {HAZE}; font-size: .85rem; text-decoration: none; }}
.back:hover {{ color: {CREAM}; }}
</style></head><body>
<img src="sheet.png" alt="Predicted setlist for {pred["target_date"]},
written like the band's handwritten stage sheet">
<div class="actions">
  <a class="dl" href="sheet.png" download="dmb-predicted-setlist-{pred["target_date"]}.png">Download the PNG</a>
  <a class="back" href="index.html">&larr; back to the prediction</a>
</div>
</body></html>"""

    site = ROOT / "site"
    site.mkdir(exist_ok=True)
    (site / "index.html").write_text(index)
    (site / "sheet.html").write_text(sheet)
    print(f"wrote {site / 'index.html'} and sheet.html")


if __name__ == "__main__":
    main()

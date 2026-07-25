"""Scrape Dave Matthews Band setlists from dmbalmanac.com into data/shows.json.

For each year page (TourShow.aspx?where=YYYY) collect show rows
(id, tid, date), skipping non-full-band shows (Dave solo / Dave & Tim /
guest appearances, flagged with a detailgreen note in the row). Each show
page (TourShowSet.aspx) is server-rendered with a setlist table whose
row-number cell background color encodes the slot; encore rows use
#660000 / #CC0000. Pages are cached on disk so re-runs don't re-fetch.
"""

from __future__ import annotations

import html as htmllib
import json
import pathlib
import re
import subprocess
import sys
import time

BASE = "https://dmbalmanac.com"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

ROOT = pathlib.Path(__file__).parent
CACHE = ROOT / "data" / "cache"
DATA = ROOT / "data"

YEARS = [2021, 2022, 2023, 2024, 2025, 2026]
ENCORE_COLORS = {"#660000", "#CC0000"}
NON_BAND = re.compile(r"Dave solo|Dave &amp; Tim|Guest", re.I)

ROW_RE = re.compile(
    r"class='setheadercell sticky-num'[^>]*"
    r"style='background-color:(#[0-9A-Fa-f]{6})[^']*'[^>]*>"
    r"\s*([^<]*?)\s*</td>"
    r'<td[^>]*class="setheadercell sticky-song"(.*?)</td>', re.S)
SONG_RE = re.compile(r'href="javascript:void\(0\);">([^<]*)</a>')
LINK_RE = re.compile(
    r'<a href="/TourShowSet\.aspx\?id=(\d+)&tid=(\d+)&where=\d+">'
    r'\s*<span class="text-resp\s*">\s*([\d.]+)\s*</span>')


def fetch(url: str, cache_key: str) -> str:
    path = CACHE / cache_key
    if path.exists() and path.stat().st_size > 5000:
        return path.read_text()
    for attempt in range(4):
        out = subprocess.run(
            ["curl", "-s", "-L", "--compressed", "-A", UA, url],
            capture_output=True, text=True, timeout=60)
        text = out.stdout
        time.sleep(1.5)
        if len(text) > 5000:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            return text
        print(f"  retry {attempt + 1} for {url}", file=sys.stderr)
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}")


def year_rows(year: int) -> list[tuple[str, str, str]]:
    """Return (show_id, tid, iso_date) for full-band shows in a year."""
    page = fetch(f"{BASE}/TourShow.aspx?where={year}", f"year-{year}.html")
    rows = []
    for m in LINK_RE.finditer(page):
        show_id, tid, date = m.groups()
        if NON_BAND.search(page[m.start():m.start() + 1200]):
            continue
        mm, dd, yy = date.split(".")
        rows.append((show_id, tid, f"20{yy}-{mm}-{dd}"))
    return rows


def parse_show(page: str) -> dict:
    songs, encore_from = [], None
    for color, _num, cell in ROW_RE.findall(page):
        name = SONG_RE.search(cell)
        if not name:
            continue
        if color.upper() in {c.upper() for c in ENCORE_COLORS} \
                and encore_from is None:
            encore_from = len(songs)
        songs.append(htmllib.unescape(name.group(1)).strip())

    # The venue anchor uses single-quoted href (songs use double quotes)
    # and is followed by "City, ST" text.
    venue, city = "", ""
    m = re.search(r"href='javascript:void\(0\);'>([^<]+)</a>"
                  r"(?:\s|&nbsp;)*([A-Za-z .'&;]+,\s*[A-Z]{2})", page)
    if m:
        venue = htmllib.unescape(m.group(1)).strip()
        city = htmllib.unescape(m.group(2)).strip()
    return {"venue": venue, "city": city, "songs": songs,
            "encore_from": encore_from}


def main() -> None:
    shows = []
    for year in YEARS:
        rows = year_rows(year)
        print(f"{year}: {len(rows)} full-band shows listed", file=sys.stderr)
        for show_id, tid, date in rows:
            url = f"{BASE}/TourShowSet.aspx?id={show_id}&tid={tid}&where={year}"
            show = parse_show(fetch(url, f"show-{show_id}.html"))
            show.update({"date": date, "id": show_id, "url": url})
            print(f'  {date} {show["venue"][:40]:40s} '
                  f'{len(show["songs"]):2d} songs enc@{show["encore_from"]}',
                  file=sys.stderr)
            shows.append(show)

    shows.sort(key=lambda s: s["date"])
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "shows.json").write_text(json.dumps(shows, indent=1))
    played = sum(1 for s in shows if s["songs"])
    print(f"wrote {len(shows)} shows ({played} with setlists)",
          file=sys.stderr)


if __name__ == "__main__":
    main()

"""Render the predicted setlist as a shareable PNG in Dave's
handwritten stage-sheet style. Reads data/prediction.json, writes
site/sheet.png. Deterministic (fixed jitter table) so re-runs are stable.
"""

from __future__ import annotations

import json
import pathlib

from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).parent
FONT = str(ROOT / "assets" / "PermanentMarker-Regular.ttf")

SHORT = {
    "Ants Marching": "ANTS", "Pantala Naga Pampa": "PNP",
    "Rapunzel": "RAPUNZAL", "Madman's Eyes": "MADMAN'S",
    "Fool in the Rain": "FOOL", "What Would You Say": "WHAT WOULD",
    "Dancing Nancies": "NANCIES", "The Stone": "STONE",
    "Don't Drink the Water": "DON'T DRINK", "Grey Street": "GREY ST",
    "The Best of What's Around": "BEST", "Tripping Billies": "TRIPPING",
    "You Might Die Trying": "DIE TRYING", "Jimi Thing": "JIMI",
    "All Along the Watchtower": "WATCHTOWER", "Two Step": "TWO STEP",
    "Crash Into Me": "CRASH", "So Much to Say": "SMTS",
    "Anyone Seen the Bridge": "BRIDGE", "Lie in Our Graves": "LIE IN",
    "Typical Situation": "TYPICAL", "When the World Ends": "WTWE",
    "Stay or Leave": "STAY OR LEAVE", "Walk Around the Moon": "WATM",
    "All That I Wanted": "ALL THAT I WANTED",
}

JITTER = [(-1.2, 0), (0.8, 4), (-0.5, -2), (1.4, 2), (-1.6, 0), (0.3, 6),
          (-0.9, -4), (1.1, 0), (-0.3, 4), (0.6, -2), (-1.4, 2), (0.9, 0),
          (-0.6, -4), (1.5, 4), (-1.1, 0), (0.5, 2), (-0.8, -2), (1.2, 0),
          (-1.0, 2), (0.7, -2)]

PAPER, INK, RULE, MARGIN = "#f7f4ea", "#17161a", "#b9cdd9", "#d96a5f"


def short(name: str) -> str:
    return SHORT.get(name, name.upper())


def draw_rotated_text(img, xy, text, font, fill, angle):
    pad = 30
    box = font.getbbox(text)
    w, h = box[2] - box[0] + 2 * pad, box[3] - box[1] + 2 * pad
    tmp = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(tmp).text((pad - box[0], pad - box[1]), text,
                             font=font, fill=fill)
    tmp = tmp.rotate(angle, resample=Image.BICUBIC, expand=True)
    img.alpha_composite(tmp, (int(xy[0] - pad), int(xy[1] - pad)))


def main() -> None:
    pred = json.loads((ROOT / "data" / "prediction.json").read_text())
    main_songs = [x["song"] for x in pred["setlist"] if x["slot"] != "encore"]
    encore = [x["song"] for x in pred["setlist"] if x["slot"] == "encore"]

    rule = 88                      # 2x scale for sharpness
    top_pad, bottom_pad = int(rule * 1.4), int(rule * 0.9)
    n_lines = 1 + len(main_songs) + 1 + len(encore)   # header + cut line
    pw, ph = 860, top_pad + n_lines * rule + bottom_pad
    W, H = pw + 200, ph + 260

    img = Image.new("RGBA", (W, H), "#171412")
    # subtle stage-light vignette
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse([-W * 0.3, -H * 0.25, W * 1.3, H * 0.55],
                                 fill=(66, 55, 40, 60))
    img.alpha_composite(glow)

    paper = Image.new("RGBA", (pw, ph), PAPER)
    pd = ImageDraw.Draw(paper)
    for y in range(top_pad, ph, rule):
        pd.line([(0, y), (pw, y)], fill=RULE, width=2)
    pd.line([(80, 0), (80, ph)], fill=MARGIN, width=3)

    px, py = (W - pw) // 2, 130
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        [px + 10, py + 14, px + pw + 10, py + ph + 14], 6, fill=(0, 0, 0, 140))
    img.alpha_composite(shadow.filter(__import__(
        "PIL.ImageFilter", fromlist=["GaussianBlur"]).GaussianBlur(12)))
    img.alpha_composite(paper, (px, py))

    # gaffer tape
    tape = Image.new("RGBA", (190, 58), (217, 208, 170, 110))
    for cx, ang in ((px + 60, -8), (px + pw - 240, 6)):
        img.alpha_composite(tape.rotate(ang, expand=True), (cx, py - 34))

    head_font = ImageFont.truetype(FONT, 34)
    song_font = ImageFont.truetype(FONT, 56)
    cap_font = ImageFont.truetype(FONT, 26)

    x0 = px + 108
    y = py + top_pad - rule + 14
    draw_rotated_text(img, (x0, y + 18),
                      f"DMB - {pred['target_date']}  HARTFORD CT",
                      head_font, "#1c1b1e99", -0.5)
    y += rule
    for i, s in enumerate(main_songs):
        rot, dx = JITTER[i % len(JITTER)]
        draw_rotated_text(img, (x0 + dx, y + 8), short(s), song_font,
                          INK, rot)
        y += rule
    # encore cut line
    cd = ImageDraw.Draw(img)
    cd.line([(x0 - 10, y + rule // 2), (x0 + int(pw * 0.62), y + rule // 2 - 6)],
            fill=INK, width=9)
    y += rule
    for j, s in enumerate(encore):
        rot, dx = JITTER[(j + 7) % len(JITTER)]
        draw_rotated_text(img, (x0 + dx, y + 8), short(s), song_font,
                          INK, rot)
        y += rule

    draw_rotated_text(img, (px + 40, py + ph + 40),
                      "PREDICTED SETLIST - NOT THE REAL SHEET",
                      cap_font, "#8a8378", 0)

    out = ROOT / "site" / "sheet.png"
    img.convert("RGB").save(out, "PNG")
    print(f"wrote {out} ({W}x{H})")


if __name__ == "__main__":
    main()

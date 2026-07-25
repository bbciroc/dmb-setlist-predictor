"""Measure whether deeper training history helps or hurts.

Usage: python3 experiment_depth.py <earliest-date> [n_backtest]
Filters history to shows >= earliest-date, then runs the rolling backtest
on the 2026 tour leg and prints the summary line.
"""

import contextlib
import io
import sys

import predict

cutoff = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 15

shows = [s for s in predict.load_shows() if s["date"] >= cutoff]
n_hist = len(shows)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    predict.backtest(shows, n)
print(f"cutoff={cutoff} ({n_hist} shows): "
      + buf.getvalue().strip().splitlines()[-1])

#!/bin/sh
# Refresh data and prediction after new shows are posted.
# Year index pages are re-fetched (show pages stay cached), the model
# retrains on the updated history, and the site regenerates.
cd "$(dirname "$0")" || exit 1
rm -f data/cache/year-2026.html
python3 scraper.py || exit 1
python3 predict.py "$@" || exit 1
python3 build_site.py
echo "Serve with: cd site && python3 -m http.server 8742"

#!/bin/bash
# Full re-run of the sanction-plan plot extraction (after the OCR and cell files exist). Stops on any error.
set -eo pipefail
cd "$(dirname "$0")/.."
rm -f data/sanction-2021/blacklist.json
for i in 1 2; do
  python3 -W ignore tools/sanction_plots.py | tail -3
  python3 tools/sanction_collisions.py | tail -2
done
python3 -W ignore tools/sanction_plots.py | tail -3
python3 tools/sanction_regions.py | tail -1
python3 -W ignore tools/sanction_roads.py
python3 -W ignore tools/sanction_road_rects.py
python3 tools/sanction_review.py
python3 tools/sanction_site.py

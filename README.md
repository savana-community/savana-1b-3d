# Savana — 3D site model

An interactive 3D model of the Adarsh Savana layout (Bettahalli, Bangalore), built from the revised sanction
plan (BIAAPA, 2021, drawn at 1:1750). It runs entirely in the browser, with no build step.

**View it:** https://savana-community.github.io/savana-1b-3d/

**Check it against the plan:** https://savana-community.github.io/savana-1b-3d/data/sanction-2021/review/
(plots, roads and parks drawn over the scanned sanction plan; hover for details)

## What's in the model
- All 1,404 plots, placed at the numbers printed on the plan and sized from its own tables (four standard sizes
  plus 255 odd-shaped plots with their sanctioned areas). Plot 1405 is the club house.
- Roads as straight rectangles, numbered R1, R2, ... — 18 m roads are darkest and have a centre median, then 12 m
  and 9 m. Click a road for its number, width and length.
- Parks, civic-amenity areas and the club house, traced from the drawing.
- 3 ft white picket fences on the plots whose owners have confirmed one (`fences.js`).
- Tools: find a plot by number, measure distances, an original-plan overlay, and export to `.glb` (metres) or `.json`.

## Contributing
The most useful contributions are corrections. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Layout
| Path | What it is |
|---|---|
| `index.html` | the viewer (Three.js) |
| `site-data.js` | generated: plots, roads, regions, plus the plan image for the overlay |
| `fences.js` | hand-maintained list of plots with a picket fence |
| `data/sanction-2021/row-hints.json` | all hand corrections: plot rows, traced shapes, roads, merges |
| `data/sanction-2021/sanction.json`, `plots.csv` | the plan's tables: every plot's size class and area |
| `data/sanction-2021/review/` | the review page (model over the scan) |
| `source/sanction-2021/scan.jpg` | the sanction plan scan |
| `tools/sanction_*.py`, `tools/sanction_run.sh` | the pipeline |

## Rebuilding the data (Python 3.11)
```
pip install numpy scipy opencv-contrib-python shapely pillow rapidocr_onnxruntime
python3 tools/build_sanction.py                 # plan tables -> data/sanction-2021/sanction.json
python3 tools/sanction_ocr_map.py source/sanction-2021/scan.jpg data/sanction-2021/map-ocr.json   # slow, once
python3 tools/sanction_cells.py source/sanction-2021/scan.jpg data/sanction-2021/cells.npz        # once
./tools/sanction_run.sh                         # everything else, ~15 s
```

This is an unofficial community model and is not affiliated with the developer or the planning authority.

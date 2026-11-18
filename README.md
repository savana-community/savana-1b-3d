# Savana Phase 1B — 3D site model

An interactive 3D model of the Adarsh Savana (Bettahalli, Bangalore) Phase 1B layout, traced from the
published master plan. It runs entirely in the browser, with no build step.

**View it:** open `index.html` in a browser (it needs internet access for the Three.js CDN), or use the hosted version at https://savana-community.github.io/savana-1b-3d/

## What's in the model
- About 1,230 plots with straight-edged boundaries traced from the drawing's divider lines, plot numbers
  (read by OCR; grey italic = inferred), and measured sizes. Colouring by size class is an optional overlay.
- Red laterite soil with overgrown weeds on empty plots, lush parks, asphalt roads, slim paver footpaths,
  hedges, avenue trees and white plot corner markers.
- 3 ft white picket fences on the plots whose owners have confirmed one (`fences.js`), with gates.
- The 7 ft concrete compound wall along the project boundary.
- Tools: find a plot by number, measure distances, an original-plan overlay, and export to `.glb` (metres) or `.json`.

The area south of Central Avenue (Future Expansion) is not modelled.

## Contributing
The most useful contributions are corrections to the data. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Layout
| Path | What it is |
|---|---|
| `index.html` | the viewer (Three.js) |
| `site-data.js` | generated: plots, regions, wall, landmarks, plus the plan image for the overlay |
| `fences.js` | hand-maintained list of plots with a picket fence |
| `tools/ocr_1b.py` → `tools/extract_1b.py` → `tools/build_data.py` | pipeline: OCR, tracing, bundling |
| `data/site.json`, `data/ocr-1b.json` | pipeline outputs |

## Rebuilding the data (Python 3.11)
```
pip install numpy scipy opencv-python scikit-image shapely pillow rapidocr_onnxruntime
python3 tools/ocr_1b.py                     # source/master-plan-1b.png -> data/ocr-1b.json
python3 -W ignore tools/extract_1b.py source/master-plan-1b.png data/ocr-1b.json data/site.json
python3 tools/build_data.py                 # -> site-data.js
```

## Known gaps
- Some plot strips are mis-split or lack numbers: 21–30, 40–46, 72–79, 456–459, 628–640, 1112–1120. Block 151–158
  has a second column (about 58–65) that isn't traced yet, and plots 617 and 1128 are mis-split.
- Plots 1242, 1243 and 1397 (confirmed fences) are in the excluded southern area.
- Sizes are measured from the drawing at about 0.34 m per pixel; expect ±0.5 m.

This is an unofficial community model and is not affiliated with the developer.

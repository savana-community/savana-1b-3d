# Contributing

Pull requests are welcome, especially corrections from people who know the site.

## Add or change a picket fence
Edit `fences.js` and add a line such as `{ plot:'123', gate:'single' }`.
- `gate` is `'single'`, `'double'` (10 ft double gate) or `null` (no gate).
- `plot` is the plot id the viewer shows when you click the plot. A duplicated number shows a suffix, e.g. `1069-2`.

Please don't add owners' names or contact details. The repository is public.

## Fix a plot boundary or number
Most plots are traced automatically by `tools/extract_1b.py`. For a block the tracer gets wrong, add a
hand-traced override like the existing one for plots 1137–1146 (search the file for `hand-traced override`).
Use crop-pixel coordinates from `data/plan-crop.png`. Then run the pipeline (see the README) and commit the
regenerated `data/site.json` and `site-data.js`.

## Checking your change
Open `index.html`, search for the plot number, and check it against the original-plan overlay slider.

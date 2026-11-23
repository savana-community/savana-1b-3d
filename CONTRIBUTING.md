# Contributing

Pull requests are welcome, especially corrections from people who know the site. You can also just post the
correction in the group with the plot or road number.

## Add or change a picket fence
Edit `fences.js` and add a line such as `{ plot:'123', gate:'single' }`.
- `gate` is `'single'`, `'double'` (10 ft double gate) or `null` (no gate).
- `plot` is the plot number.

Please don't add owners' names or contact details. The repository is public.

## Fix a plot or a road
All hand corrections live in `data/sanction-2021/row-hints.json` (coordinates are pixels on
`source/sanction-2021/scan.jpg`):
- `rows`: plots that sit side by side in one row, in order, with a direction (`up`, `down`, `left`, `right`, ...).
- `shapes`: a hand-traced outline for a plot that isn't a rectangle.
- `stack`: a rectangular plot sitting square on its neighbour (`"601": 600`).
- `roads`: an explicit straight road (centre line and width).
- `road_merge`: road numbers that are one road (`["R44", "R69"]`).
- `road_fix`: per road number, `connect` (run straight to another road), `width_m`, `extend_m` or `delete`.

Then run `./tools/sanction_run.sh` and commit the regenerated files.

## Checking your change
Open the review page (`data/sanction-2021/review/index.html`) to see the model over the scan, or `index.html`
for the 3D view.

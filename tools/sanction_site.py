"""Convert the sanction-plan extraction (scan pixels) into the viewer's site format (metres, north up).

Inputs: data/sanction-2021/{plots-geo.json, sanction.json, regions-geo.json (optional)}
Output: data/sanction-2021/site.json and data/sanction-2021/viewer/site-data.js
"""
import json, base64, io
import numpy as np
from pathlib import Path
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

D = Path('data/sanction-2021')
X0, Y0, X1, Y1 = 500, 700, 5500, 6700          # crop of the scan that the overlay shows
W, H = X1 - X0, Y1 - Y0
G = json.load(open(D / 'plots-geo.json')); MPP = G['mpp']; G = {int(k): v for k, v in G['plots'].items()}
STORE = json.load(open(D / 'sanction.json')); P = {p['plot']: p for p in STORE['plots']}
TYPE = {'9.14 x 12.19': 'sapphire', '9.14 x 15.24': 'ruby', '12.19 x 18.28': 'diamond', '15.24 x 24.38': 'platinum', 'odd': 'premium'}

def m(pt):  # scan px -> metres, origin at crop centre, y north
    return [round((pt[0] - X0 - W / 2) * MPP, 2), round((H / 2 - (pt[1] - Y0)) * MPP, 2)]

T = 6.4  # line thickness in px: drawn outlines are the inside of the lines, so grow them by half
plots = []
for n in sorted(P):
    v, p = G.get(n, {}), P[n]
    if p['class'] == 'club house': continue
    if 'pc' in v:
        if 'row_poly' in v:
            poly = Polygon(v['row_poly']); outline = 'row-aligned, exact size'
        else:
            c = np.array(v['pc']); u = np.array(v['pu']); w = np.array([-u[1], u[0]])
            d = p['depth_m'] / MPP / 2; f = p['frontage_m'] / MPP / 2
            poly = Polygon([c+u*d+w*f, c+u*d-w*f, c-u*d-w*f, c-u*d+w*f]); outline = 'exact size'
        how = v['placed']
    elif 'odd_poly' in v:
        poly = Polygon(v['odd_poly']).buffer(0)
        if v['odd_how'].startswith('cell'): poly = poly.buffer(T / 2, join_style=2)
        outline = 'drawn' if v['odd_how'].startswith('cell') else 'approximate'; how = v['odd_how']
    else:
        continue
    if isinstance(poly, MultiPolygon): poly = max(poly.geoms, key=lambda g: g.area)
    poly = poly.simplify(1.0)
    pts = [m(q) for q in list(poly.exterior.coords)[:-1]]
    if Polygon(pts).exterior.is_ccw is False: pts = pts[::-1]
    plots.append({
        'id': str(n), 'number': n, 'number_inferred': how not in ('cell', 'cell A', 'cell B', 'cell C', 'extended (number read)', 'seeded from numbers'),
        'type': TYPE[p['class']], 'w_m': p['frontage_m'], 'd_m': p['depth_m'],
        'area_m2': p['area_sqm'], 'area_drawn_m2': round(Polygon(pts).area, 1),
        'outline': outline, 'placed_by': how, 'poly': pts,
    })

regions, boundary, landmarks = [], [], []
LAYER = {'park': 'park', 'civic': 'open', 'clubhouse': 'clubhouse', 'road': 'road'}
road_rects = []
rr = D / 'road-rects.json'
if rr.exists():
    for r in json.load(open(rr))['roads']:
        road_rects.append({**r, 'poly': [m(q) for q in r['poly']], 'centre': [m(q) for q in r['centre']], 'label_at': m(r['label_at'])})
rd = D / 'roads-geo.json'
if False and rd.exists():
    for r in json.load(open(rd))['roads']:
        regions.append({**r, 'outer': [m(q) for q in r['outer']], 'holes': [[m(q) for q in h] for h in r['holes']]})
rg = D / 'regions-geo.json'
if rg.exists():
    R = json.load(open(rg))
    for r in R.get('regions', []):
        regions.append({**r, 'layer': LAYER.get(r['layer'], r['layer']), 'outer': [m(q) for q in r['outer']], 'holes': [[m(q) for q in h] for h in r.get('holes', [])]})
    boundary = [m(q) for q in R.get('boundary', [])]
    landmarks = [{**l, 'at': m(l['at'])} for l in R.get('landmarks', [])]
H_ = json.load(open(D / 'row-hints.json'))
for name, q in H_.get('exclude', {}).items():
    regions.append({'id': name, 'name': name, 'layer': 'reserved', 'area_m2': round(Polygon(q).area * MPP**2, 1), 'outer': [m(p_) for p_ in q], 'holes': []})
if not boundary:   # stand-in until the site boundary is traced: everything placed, closed over the roads
    u = unary_union([Polygon(p['poly']) for p in plots] + [Polygon(r['outer']) for r in regions])
    hull = u.buffer(14, join_style=2).buffer(-8, join_style=2)
    if isinstance(hull, MultiPolygon): hull = max(hull.geoms, key=lambda g: g.area)
    boundary = [[round(x, 2), round(y, 2)] for x, y in list(hull.simplify(1.5).exterior.coords)[:-1]]

# footpaths: 1.2 m paver ring around each block of plots, built as one merged shape (no overlaps)
from shapely.ops import unary_union as _uu
_pl = _uu([Polygon(p['poly']).buffer(0.3, join_style=2) for p in plots]).buffer(-0.3, join_style=2)
_fp = _pl.buffer(1.2, join_style=2).difference(_pl)
footpaths = []
for g in (_fp.geoms if hasattr(_fp, 'geoms') else [_fp]):
    if g.area < 1: continue
    g = g.simplify(0.2)
    footpaths.append({'outer': [[round(x, 2), round(y, 2)] for x, y in list(g.exterior.coords)[:-1]],
                      'holes': [[[round(x, 2), round(y, 2)] for x, y in list(r.coords)[:-1]] for r in g.interiors]})
site = {
    'meta': {
        'source': 'Revised sanction plan (BIAAPA, 2021), 1:1750',
        'metres_per_px': MPP, 'crop_px': [X0, Y0, X1, Y1],
        'note': 'Plots placed from the plot numbers printed on the sanction plan; sizes and areas from its tables. '
                'Regular plots use their drawn outline where one was clean, otherwise their exact sanctioned size.',
        'plotSizes': {'sapphire': {'frontage_m': 9.14, 'depth_m': 12.19}, 'ruby': {'frontage_m': 9.14, 'depth_m': 15.24},
                      'diamond': {'frontage_m': 12.19, 'depth_m': 18.28}, 'platinum': {'frontage_m': 15.24, 'depth_m': 24.38}},
        'planImage': {'widthPx': W, 'heightPx': H},
        'plotsPlaced': len(plots), 'plotsTotal': 1404,
    },
    'roadRects': road_rects, 'footpaths': footpaths, 'boundary': boundary, 'wall': [boundary + boundary[:1]], 'regions': regions, 'plots': plots, 'trees': [], 'landmarks': landmarks,
}
json.dump(site, open(D / 'site.json', 'w'))
im = Image.open('source/sanction-2021/scan.jpg').convert('RGB').crop((X0, Y0, X1, Y1))
im = im.resize((W // 2, H // 2), Image.LANCZOS)
buf = io.BytesIO(); im.save(buf, 'JPEG', quality=80)
(D / 'viewer').mkdir(exist_ok=True)
(D / 'viewer' / 'site-data.js').write_text('window.SITE = ' + json.dumps(site, separators=(',', ':')) + ';\nwindow.PLAN_IMAGE = "data:image/jpeg;base64,'
                                          + base64.b64encode(buf.getvalue()).decode() + '";\n')
print('plots', len(plots), 'regions', len(regions))

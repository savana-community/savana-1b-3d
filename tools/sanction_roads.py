"""Roads = the gaps between blocks. Blocks are the union of all placed plots and traced regions; closing that
union with mitred (square) joins fills the roads between blocks, so every road edge is a straight plot or
park edge. Output: data/sanction-2021/roads-geo.json (scan px)."""
import json
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

D = 'data/sanction-2021/'
MPP = 0.2235
G = json.load(open(D + 'plots-geo.json'))['plots']
R = json.load(open(D + 'regions-geo.json'))['regions']
shapes = []
for v in G.values():
    q = v.get('row_poly') or v.get('odd_poly')
    if q and len(q) >= 3:
        p = Polygon(q).buffer(0)
        if p.area > 0: shapes.append(p)
for r in R: shapes.append(Polygon(r['outer']).buffer(0))
blocks = unary_union([s.buffer(1.5, join_style=2) for s in shapes])          # weld shared plot edges
HALF = 19.0 / MPP / 2 + 4                                                     # widest road 18 m
closed = blocks.buffer(HALF, join_style=2, mitre_limit=10).buffer(-HALF, join_style=2, mitre_limit=10)
import os
H = json.load(open(D + 'row-hints.json')) if os.path.exists(D + 'row-hints.json') else {}
from shapely.geometry import LineString
EXCL = unary_union([Polygon(q) for q in H.get('exclude', {}).values()]) if H.get('exclude') else Polygon()
blocks_x = unary_union([blocks, EXCL])
closed = blocks_x.buffer(HALF, join_style=2, mitre_limit=10).buffer(-HALF, join_style=2, mitre_limit=10)
site = closed
strips = [LineString(r['line']).buffer(r['width_m'] / MPP / 2, cap_style=2, join_style=2) for r in H.get('roads', [])]
if strips: closed = unary_union([closed] + [s_.intersection(site.buffer(HALF, join_style=2)) for s_ in strips])
roads = closed.difference(blocks_x).buffer(-0.5, join_style=2).buffer(0.5, join_style=2)
geoms = list(roads.geoms) if isinstance(roads, MultiPolygon) else [roads]
geoms = [g.simplify(1.5) for g in geoms if g.area * MPP ** 2 > 40]
out = []
for i, g in enumerate(sorted(geoms, key=lambda g: -g.area)):
    out.append({'id': f'road-{i+1}', 'layer': 'road', 'area_m2': round(g.area * MPP ** 2, 1),
                'outer': [list(map(float, c)) for c in list(g.exterior.coords)[:-1]],
                'holes': [[list(map(float, c)) for c in list(h.coords)[:-1]] for h in g.interiors]})
outline = closed if not isinstance(closed, MultiPolygon) else max(closed.geoms, key=lambda g: g.area)
json.dump({'roads': out, 'site_outline': [list(map(float, c)) for c in list(outline.exterior.coords)[:-1]]},
          open(D + 'roads-geo.json', 'w'))
print('road pieces', len(out), 'total %.0f m2' % sum(r['area_m2'] for r in out), '(sanction table: 147,675.09 m2)')

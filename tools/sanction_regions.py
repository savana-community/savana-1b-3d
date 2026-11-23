"""Trace parks, civic-amenity (CA) areas and the club house from the sanction-plan scan.

Each region has a seed point (its printed label, from OCR) and a printed area. The traced outline is the
drawn cell around the seed (or close to it) whose area matches the printed one best. The club house is
hatched with a grid, so its cell is found after closing the grid lines into a solid block.
Overrides (hand-traced outlines, scan px) go in data/sanction-2021/row-hints.json under "region_shapes".
Output: data/sanction-2021/regions-geo.json
"""
import json, os
import numpy as np, cv2

D = 'data/sanction-2021/'
MPP = 0.2235
# name: (layer, seed scan px, printed area m2)
SEEDS = {
    'Park-1': ('park', (3968, 1909), 1403.33), 'Park-2': ('park', (4558, 1548), 1370.98),
    'Park-3': ('park', (4994, 1306), 394.20), 'Park-4': ('park', (5065, 1852), 2045.72),
    'Park-5': ('park', (4950, 2500), 1269.27), 'Park-6': ('park', (4892, 3268), 1828.71), 'Park-7': ('park', (4824, 4172), 1628.12),
    'Park-8': ('park', (4754, 5100), 620.20), 'Park-9': ('park', (2594, 5363), 2839.65),
    'Park-11': ('park', (1484, 4022), 4099.70), 'Park-13': ('park', (3414, 2356), 7792.82),
    'Park-14': ('park', (3204, 4400), 8922.23), 'Park-15': ('park', (2465, 4312), 9665.30),
    'Park-16': ('park', (4854, 4815), 85.34), 'Park-17': ('park', (4890, 4512), 65.15),
    'Park-18': ('park', (4208, 4924), 387.06), 'Park-19': ('park', (4256, 2627), 896.91),
    'CA-1': ('civic', (1776, 2884), 16587.13), 'CA-2': ('civic', (2792, 2452), 4232.65),
    'CA-3': ('civic', (4274, 2188), 1974.08), 'CA-4': ('civic', (1988, 4367), 1763.55),
    'Club House': ('clubhouse', (3706, 4550), 5919.86),
}

SEGS = []
for nm in ('cells', 'cells-c', 'cells-b', 'cells-d'):
    z = np.load(D + nm + '.npz'); SEGS.append((nm, z['lab'], z['stats']))
X0, Y0 = (int(v) for v in np.load(D + 'cells.npz')['origin'])
INK = np.load(D + 'cells.npz')['ink']

def outline(mask):
    m = mask.astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 4.0, True).reshape(-1, 2).astype(float)
    return c

def cell_candidates(seed, want_px):
    sx, sy = seed[0] - X0, seed[1] - Y0
    pts = [(sx, sy)] + [(sx + r * np.cos(a), sy + r * np.sin(a)) for r in (40, 90, 160) for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
    seen = set(); out = []
    for nm, lab, st in SEGS:
        for x, y in pts:
            x, y = int(x), int(y)
            if not (0 <= y < lab.shape[0] and 0 <= x < lab.shape[1]): continue
            cid = int(lab[y, x])
            if cid <= 1 or (nm, cid) in seen: continue
            seen.add((nm, cid))
            a = st[cid, 4]
            if not (0.5 * want_px < a < 1.6 * want_px): continue
            x0, y0, w, h, _ = st[cid]
            m = np.zeros(lab.shape, np.uint8); m[y0:y0+h, x0:x0+w] = (lab[y0:y0+h, x0:x0+w] == cid)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
            c = outline(m)
            ar = cv2.contourArea(c.astype(np.float32)) + 3.2 * cv2.arcLength(c.astype(np.float32).reshape(-1, 1, 2), True)
            out.append((abs(ar / want_px - 1), nm, c, ar))
    return sorted(out, key=lambda z: z[0])

def clubhouse(seed, want_px):
    """grid hatching: union the small grid squares around the label, close them into one block"""
    nm, lab, st = SEGS[0]
    sx, sy = seed[0] - X0, seed[1] - Y0
    R = 450
    y0, x0 = max(0, sy - R), max(0, sx - R)
    sub = lab[y0:sy + R, x0:sx + R]
    ids = np.unique(sub); ids = ids[ids > 1]
    small = ids[(st[ids, 4] > 60) & (st[ids, 4] < 1500)]
    m = np.isin(sub, small).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))
    n, cl = cv2.connectedComponents(m)
    sizes = np.bincount(cl.ravel()); sizes[0] = 0
    k = cl[sy - y0, sx - x0] or int(sizes.argmax())
    c = outline(cl == k) + [x0, y0]
    hull = cv2.convexHull(c.astype(np.float32)).reshape(-1, 2)
    return [(0, 'grid', hull, cv2.contourArea(hull))]

H = json.load(open(D + 'row-hints.json')) if os.path.exists(D + 'row-hints.json') else {}
OVR = H.get('region_shapes', {})
regions = []
for name, (layer, seed, area) in SEEDS.items():
    want = area / MPP ** 2
    if name in OVR:
        q = np.array(OVR[name], float); how = 'hand traced'
        got = cv2.contourArea(q.astype(np.float32)) * MPP ** 2
    else:
        cands = clubhouse(seed, want) if layer == 'clubhouse' else cell_candidates(seed, want)
        if not cands or cands[0][0] > 0.3:
            print(f'{name:11s} NOT TRACED (best {cands[0][0]:.0%} off)' if cands else f'{name:11s} NOT TRACED (no cell)')
            continue
        err, src, c, ar = cands[0]
        q = c + [X0, Y0] if layer != 'clubhouse' else c + [X0, Y0]; how = f'drawn cell ({src})'
        got = ar * MPP ** 2
    print(f'{name:11s} {how:22s} traced {got:9.1f} m2  printed {area:9.2f}  ({got / area - 1:+.0%})')
    regions.append({'id': name, 'name': name, 'layer': layer, 'area_m2': area, 'area_traced_m2': round(got, 1),
                    'traced_by': how, 'outer': q.round(1).tolist(), 'holes': []})
json.dump({'regions': regions}, open(D + 'regions-geo.json', 'w'))
print(len(regions), 'regions traced of', len(SEEDS))

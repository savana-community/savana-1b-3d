"""Find plots placed on top of each other; blacklist the reading with less support from its numbering
neighbours. Prints the number of collisions. Usage: python3 tools/sanction_collisions.py"""
import json, numpy as np
from shapely.geometry import Polygon
from shapely.strtree import STRtree
D = 'data/sanction-2021/'
G = json.load(open(D + 'plots-geo.json')); mpp = G['mpp']; G = {int(k): v for k, v in G['plots'].items()}
P = {p['plot']: p for p in json.load(open(D + 'sanction.json'))['plots']}
polys, cen = {}, {}
for n, v in G.items():
    if 'pc' in v:
        c = np.array(v['pc']); u = np.array(v['pu']); w = np.array([-u[1], u[0]])
        d = P[n]['depth_m']/mpp/2; f = P[n]['frontage_m']/mpp/2
        polys[n] = Polygon([c+u*d+w*f, c+u*d-w*f, c-u*d-w*f, c-u*d+w*f]); cen[n] = c
    elif 'odd_poly' in v:
        polys[n] = Polygon(v['odd_poly']).buffer(0); cen[n] = np.array(polys[n].centroid.coords[0])
def support(n):
    """numbering neighbours placed on the same row (or, for odd plots, close by)"""
    u = np.array(G[n]['pu']) if 'pu' in G[n] else None
    s = 0
    for m in range(n-3, n+4):
        if m == n or m not in cen: continue
        d = cen[m] - cen[n]
        if np.linalg.norm(d) > 3.5 * 15.24 / mpp: continue
        if u is not None and abs(d @ u) > 12: continue
        s += 1
    return s
keys = list(polys); tree = STRtree([polys[k] for k in keys])
BLp = D + 'blacklist.json'
try: BL = json.load(open(BLp))
except FileNotFoundError: BL = []
hits = 0
for i, k in enumerate(keys):
    for j in tree.query(polys[k]):
        j = int(j); m = keys[j]
        if j <= i: continue
        a = polys[k].intersection(polys[m]).area
        if a <= 0.15 * min(polys[k].area, polys[m].area): continue
        hits += 1
        sk, sm = support(k), support(m)
        loser = k if sk < sm else m if sm < sk else max(k, m, key=lambda q: 0 if G[q].get('placed') == 'cell' else 1)
        x, y = cen[loser]
        BL.append([loser, float(x), float(y)])
        print('collision', k, m, 'support', sk, sm, '-> drop', loser)
json.dump(BL, open(BLp, 'w'))
print('collisions', hits)

# ---------- numbering-trend check: a row reads 440, 439, "430" -> the third is really 438 ----------
def one_digit_apart(a, b):
    sa, sb = str(a), str(b)
    return len(sa) == len(sb) and sum(x != y for x, y in zip(sa, sb)) == 1
placed = {n for n in G if 'pc' in G[n]}
fixes = 0
for n in sorted(placed):
    if G[n].get('placed') != 'cell': continue
    u = np.array(G[n]['pu']); w = np.array([-u[1], u[0]]); F = P[n]['frontage_m'] / mpp
    # row neighbours on each side, nearest first
    for side in (1, -1):
        nb = []
        for m in placed:
            if m == n: continue
            d = cen[m] - cen[n]
            if abs(d @ u) < 12 and 0 < side * (d @ w) < 2.6 * F: nb.append((side * (d @ w), m))
        nb = [m for _, m in sorted(nb)]
        if len(nb) < 2: continue
        y1, y2 = nb[0], nb[1]                    # y1 next to n, y2 beyond
        step = y1 - y2
        if abs(step) != 1: continue
        k = y1 + step
        if k == n or k in placed or not (1 <= k <= 1404) or not one_digit_apart(n, k): continue
        if P[k]['class'] != P[n]['class']: continue
        x, y = cen[n]; BL.append([n, float(x), float(y), k]); fixes += 1
        print('relabel', n, '->', k, 'row', y2, y1)
        break
json.dump(BL, open(BLp, 'w'))
print('relabels', fixes)

"""Match plot numbers read on the sanction-plan map to their drawn cells, and build plot geometry.

Inputs: data/sanction-2021/{cells.npz, map-ocr.json, sanction.json}
Output: data/sanction-2021/plots-geo.json (scan-pixel geometry + status per plot)
"""
import json, re, collections, os
import numpy as np, cv2

D = 'data/sanction-2021/'
Z = np.load(D + 'cells.npz'); LAB = Z['lab']; ST = Z['stats']; X0, Y0 = (int(v) for v in Z['origin'])
STORE = json.load(open(D + 'sanction.json'))
P = {p['plot']: p for p in STORE['plots']}
MPP = 0.2235          # metres per scan pixel (fitted from regular plots; nominal 1:1750 gives 0.2222)
T = 6.4               # drawn line thickness eaten by the cell mask, px

# ---------- OCR tokens that look like plot numbers ----------
TABLES = [(0, 0, 2100, 2760), (0, 5330, 2150, 9704), (2050, 6540, 7017, 9704)]   # printed tables, scan px
toks = []
for f in ('map-ocr.json', 'map-ocr-2.json'):          # two OCR passes at different magnifications
    if not os.path.exists(D + f): continue
    for o in json.load(open(D + f)):
        t = o['t'].strip().rstrip('*').strip()
        if not (re.fullmatch(r'\d{1,4}', t) and 1 <= int(t) <= 1405): continue
        b = np.array(o['box']); h = np.linalg.norm(b[3] - b[0]); cx, cy = b[:, 0].mean(), b[:, 1].mean()
        if any(a <= cx < c and bb <= cy < d for a, bb, c, d in TABLES): continue
        if any(tk['n'] == int(t) and abs(tk['x'] + X0 - cx) < 12 and abs(tk['y'] + Y0 - cy) < 12 for tk in toks): continue
        toks.append({'n': int(t), 'x': cx - X0, 'y': cy - Y0, 'c': o['c'], 'h': h, 'w': np.linalg.norm(b[1] - b[0])})
        # the OCR often drops a leading 1 (110 -> "10", 1208 -> "208"): keep that reading as a weaker alternative
        alt = int('1' + t.lstrip('0') if t.startswith('0') else '1' + t)
        if t.startswith('0'): alt = int('1' + t)
        if alt <= 1404:
            toks.append({'n': alt, 'x': cx - X0, 'y': cy - Y0, 'c': o['c'] * 0.5, 'h': h, 'w': np.linalg.norm(b[1] - b[0]), 'alt': True})

# ---------- cell geometry ----------
_geo = {}
def cell(cid):
    if cid in _geo: return _geo[cid]
    x0, y0, w, h, a = ST[cid]
    m = (LAB[y0:y0+h, x0:x0+w] == cid).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea).reshape(-1, 2) + [x0, y0]
    rect = cv2.minAreaRect(c.astype(np.float32))
    (cx, cy), (rw, rh), _ = rect
    box = cv2.boxPoints(rect)
    e1, e2 = box[1] - box[0], box[2] - box[1]
    if np.linalg.norm(e1) < np.linalg.norm(e2): e1, e2 = e2, e1
    g = {'id': int(cid), 'area': int(a), 'poly': c, 'center': (cx, cy), 'L': float(np.linalg.norm(e1)),
         'W': float(np.linalg.norm(e2)), 'axis': e1 / np.linalg.norm(e1), 'fill': a / max(1, rw*rh)}
    _geo[cid] = g; return g

def fits(n, g):
    """how well cell g fits plot n's size class: 0 = perfect, larger = worse"""
    p = P[n]
    if p['frontage_m']:
        eL, eW = p['depth_m']/MPP - T, p['frontage_m']/MPP - T
        return max(abs(g['L']-eL)/eL, abs(g['W']-eW)/eW) + (0 if g['fill'] > .9 else 1)
    exp = p['area_sqm'] / MPP**2
    got = g['area'] + T * (g['L'] + g['W'])   # add back the line border
    return abs(got - exp) / exp

import os
BL = json.load(open(D + 'blacklist.json')) if os.path.exists(D + 'blacklist.json') else []
def blocked(n, x, y): return any(len(b) == 3 and b[0] == n and np.hypot(b[1]-x, b[2]-y) < 62 for b in BL)
toks = [t for t in toks if not blocked(t['n'], t['x'] + X0, t['y'] + Y0)]
for b in BL:                       # [n, x, y, k]: the reading n at (x, y) is really k (numbering trend)
    if len(b) == 4:
        for t in toks:
            if t['n'] == b[0] and np.hypot(b[1] - t['x'] - X0, b[2] - t['y'] - Y0) < 62: t['n'] = b[3]

cands = collections.defaultdict(list)       # plot number -> [(score, cell, token)]
for tk in toks:
    xi, yi = int(tk['x']), int(tk['y'])
    if not (8 <= yi < LAB.shape[0]-8 and 8 <= xi < LAB.shape[1]-8): continue
    win = LAB[yi-6:yi+7, xi-int(tk['w']/2):xi+int(tk['w']/2)+1].ravel()
    win = win[win > 0]
    if not len(win): continue
    cid = np.bincount(win).argmax()
    if cid == 0 or ST[cid, 4] > 150000 or ST[cid, 4] < 300: continue
    g = cell(cid)
    cands[tk['n']].append((fits(tk['n'], g) + (0.05 if tk.get('alt') else 0), cid, tk))

# ---------- choose a cell per plot; one plot per cell ----------
pairs = sorted(((s, n, cid) for n, l in cands.items() for s, cid, _ in l), key=lambda z: z[0])
plot_cell, cell_plot = {}, {}
for s, n, cid in pairs:
    if s > 0.18 or n in plot_cell or cid in cell_plot: continue
    plot_cell[n] = cid; cell_plot[cid] = n

status = collections.Counter()
out = {}
for n in range(1, 1406):
    if n in plot_cell:
        g = cell(plot_cell[n]); status['cell'] += 1
        out[n] = {'status': 'cell', 'cell': g['id'], 'center': [g['center'][0]+X0, g['center'][1]+Y0],
                  'axis': g['axis'].tolist(), 'fit': round(fits(n, g), 3)}
    elif n in cands:
        status['read, cell rejected'] += 1
        s, cid, tk = min(cands[n], key=lambda z: z[0])
        out[n] = {'status': 'read', 'at': [tk['x']+X0, tk['y']+Y0], 'best_fit': round(s, 3), 'cell': int(cid)}
    else:
        status['not read'] += 1; out[n] = {'status': 'missing'}
print(dict(status))
by = collections.Counter((P[n]['class'], v['status']) for n, v in out.items())
for k in sorted(by): print(k, by[k])
json.dump({'mpp': MPP, 'plots': out}, open(D + 'plots-geo.json', 'w'))

# ======================= stage 2: fill rows between / beyond anchors =======================
def is_reg(n): return P[n]['frontage_m'] is not None
def Fpx(n): return P[n]['frontage_m'] / MPP
def Dpx(n): return P[n]['depth_m'] / MPP


# ---------- spatial indexes: same answers as a linear scan, only looking at nearby entries ----------
GRID = 64
class GridDict(dict):
    """dict of placed plots that also buckets each entry by the grid cells its footprint can touch.
    Lookups return candidates in first-insertion order, matching the old full scans."""
    def __init__(self, radius_fn):
        super().__init__(); self.cells = collections.defaultdict(set); self.where = {}; self.order = {}; self.r = radius_fn
    def _keys(self, n, v):
        x0, y0, x1, y1 = self.r(n, v)
        return [(i, j) for i in range(int(x0 // GRID), int(x1 // GRID) + 1) for j in range(int(y0 // GRID), int(y1 // GRID) + 1)]
    def __setitem__(self, n, v):
        if n in self.where:
            for k in self.where[n]: self.cells[k].discard(n)
        else: self.order[n] = len(self.order)
        super().__setitem__(n, v)
        ks = self._keys(n, v); self.where[n] = ks
        for k in ks: self.cells[k].add(n)
    def __delitem__(self, n):
        for k in self.where.pop(n): self.cells[k].discard(n)
        super().__delitem__(n)
    def near(self, pt):
        k = (int(pt[0] // GRID), int(pt[1] // GRID))
        return sorted(self.cells.get(k, ()), key=self.order.__getitem__)

def _reg_box(n, v):
    r = np.hypot(Dpx(n), Fpx(n)) / 2 + 1
    return v['c'][0] - r, v['c'][1] - r, v['c'][0] + r, v['c'][1] + r
def _odd_box(n, v):
    q = np.asarray(v['poly']); return q[:, 0].min() - 4, q[:, 1].min() - 4, q[:, 0].max() + 4, q[:, 1].max() + 4

TOKGRID = collections.defaultdict(list)
for _i, tk in enumerate(toks): TOKGRID[(int((tk['x'] + X0) // GRID), int((tk['y'] + Y0) // GRID))].append(_i)

place = GridDict(_reg_box)    # n -> dict(c=np.array, u=np.array (depth dir), how=str)
for n, v in out.items():
    if v['status'] == 'cell' and is_reg(n) and v['fit'] < 0.12:
        place[n] = {'c': np.array(v['center']), 'u': np.array(v['axis']), 'how': 'cell'}

tok_at = collections.defaultdict(list)
for tk in toks: tok_at[tk['n']].append(np.array([tk['x'] + X0, tk['y'] + Y0]))

def same_line(a, b):
    ua, ub = place[a]['u'], place[b]['u']
    if abs(abs(ua @ ub) - 1) > 1 - np.cos(np.radians(3)): return None
    u = ua if ua @ ub > 0 else -ua
    d = place[b]['c'] - place[a]['c']
    if abs(d @ u) > 9: return None
    return u, d

def seed_rows():
    changed = False
    # --- seed a row with no anchor from two consecutive numbers read one frontage apart ---
    for a in range(1, 1404):
        b = a + 1
        if a in place or b in place or not (is_reg(a) and is_reg(b)) or Dpx(a) != Dpx(b): continue
        best = None
        for ta in tok_at.get(a, []):
            for tb in tok_at.get(b, []):
                dist = np.linalg.norm(tb - ta); exp = (Fpx(a) + Fpx(b)) / 2
                if abs(dist - exp) < 0.12 * exp and (best is None or abs(dist-exp) < best[0]): best = (abs(dist-exp), ta, tb)
        if best is None: continue
        _, ta, tb = best
        v = (tb - ta) / np.linalg.norm(tb - ta); u = np.array([v[1], -v[0]])
        place[a] = {'c': ta, 'u': u, 'how': 'seeded from numbers'}
        place[b] = {'c': tb, 'u': u, 'how': 'seeded from numbers'}; changed = True
    return changed

def bounded_fill():
    """fill 1-2 regular plots between a placed plot and the next placed number further along the same row
    (the far one may be an odd plot placed later), when the gap fits"""
    changed = False
    for a in sorted(place):
        for step in (1, -1):
            for k in (1, 2, 3, 4):
                gap = [a + step*i for i in range(1, k+1)]; far = a + step*(k+1)
                if any(g in place or g < 1 or g > 1404 or not is_reg(g) or Dpx(g) != Dpx(a) for g in gap): continue
                fc = place[far]['c'] if far in place else (odd_far.get(far))
                if fc is None: continue
                nb = [c for c in (a - step, a - 2*step) if c in place and same_line(a, c)]
                if not nb: continue
                v = place[a]['c'] - place[nb[0]]['c']; v /= np.linalg.norm(v)
                pos = Fpx(a)/2; cs = []
                for g in gap: pos += Fpx(g)/2; cs.append(place[a]['c'] + v*pos); pos += Fpx(g)/2
                d = fc - place[a]['c']
                along = d @ v; across = abs(d @ place[a]['u'])
                if not (pos - 0.1*Fpx(a) < along < pos + 3.0*Fpx(a)) or across > 0.5*Dpx(a): continue
                for g, c in zip(gap, cs): place[g] = {'c': c, 'u': place[a]['u'], 'how': 'between (row end)'}
                changed = True; break
    return changed

def occupied(pt, skip=None):
    for m in place.near(pt):
        v = place[m]
        if m == skip: continue
        u = v['u']; w = np.array([-u[1], u[0]]); d = pt - v['c']
        if abs(d @ u) < Dpx(m)/2 - 4 and abs(d @ w) < Fpx(m)/2 - 4: return m
    return None

def mirror_extend():
    """continue a row one plot at a time while the plot directly behind the next slot (back-to-back row)
    is already placed; the back row tells us the block continues there"""
    changed = False
    for a in sorted(place):
        u = place[a]['u']; w = np.array([-u[1], u[0]])
        for b in (a-1, a+1):
            if b < 1 or b > 1404 or b in place or not is_reg(b) or Dpx(b) != Dpx(a): continue
            nb = [c for c in (a-1, a+1, a-2, a+2) if c in place and c != b and same_line(a, c)]
            dirs = []
            if nb:
                v = place[a]['c'] - place[nb[0]]['c']; v /= np.linalg.norm(v)
                if (nb[0] < a) != (b > a): v = -v
                dirs = [v]
            else:
                dirs = [w, -w]
            ok = []
            for v in dirs:
                cb = place[a]['c'] + v * (Fpx(a)/2 + Fpx(b)/2)
                if occupied(cb) is not None: continue
                if any(t != b and abs(t - b) < 30 for t in cell_tokens_at(cb)): continue
                for side in (1, -1):
                    m = occupied(cb + side * u * (Dpx(b)/2 + 15), skip=None)
                    if m is None or m == a: continue
                    dm = place[m]['c'] - cb
                    if abs(dm @ w) < Fpx(m)/2 - 0.2*Fpx(b) and abs(abs(dm @ u) - (Dpx(b) + Dpx(m))/2) < 6 and abs(place[m]['u'] @ u) > 0.995:
                        ok.append(cb); break
            if len(ok) == 1:
                place[b] = {'c': ok[0], 'u': u, 'how': 'mirrors back row'}; changed = True
    return changed

def cell_tokens_at(pt):
    """plot numbers read within half a plot of pt"""
    gx, gy = int(pt[0] // GRID), int(pt[1] // GRID)
    ids = sorted(i for a in (gx-1, gx, gx+1) for b in (gy-1, gy, gy+1) for i in TOKGRID.get((a, b), ()))
    return [toks[i]['n'] for i in ids if np.hypot(toks[i]['x'] + X0 - pt[0], toks[i]['y'] + Y0 - pt[1]) < 18]

HINTS = json.load(open(D + 'row-hints.json'))['rows'] if os.path.exists(D + 'row-hints.json') else []
def hint_fill():
    """rows confirmed by review: place unplaced regular members end to end from a placed member"""
    changed = False
    for row in HINTS:
        row = row['seq']
        idx = {n: i for i, n in enumerate(row)}
        known = [n for n in row if n in place]
        if len(known) < 2: continue
        a, b = known[0], known[-1]
        v = place[b]['c'] - place[a]['c']; v /= np.linalg.norm(v)
        for i, n in enumerate(row):
            if n in place or not is_reg(n): continue
            # nearest placed member, then walk the frontages between
            ref = min(known, key=lambda k: abs(idx[k] - i)); j = idx[ref]
            step = 1 if i > j else -1
            between = row[j+step:i:step] if step == 1 else row[i+1:j][::-1]
            if any(not is_reg(m) for m in between) or Dpx(n) != Dpx(ref): continue
            dist = Fpx(ref)/2 + sum(Fpx(m) for m in between) + Fpx(n)/2
            place[n] = {'c': place[ref]['c'] + step * v * dist, 'u': place[ref]['u'], 'how': 'row hint'}
            changed = True
    return changed

odd_far = {}
cell_tokens = collections.defaultdict(set)
for n, l in cands.items():
    for _, cid, _ in l: cell_tokens[cid].add(n)
claimed = set()
def regular_pass():
    global changed
    for _round in range(20):
      changed = True
      while changed:
        changed = False
        # --- interpolate between two placed plots of the same row ---
        keys = sorted(place)
        for i, a in enumerate(keys):
            for b in keys[i+1:]:
                if b - a > 16: break
                if b - a < 2 or any(g in place for g in range(a+1, b)): continue
                gap = list(range(a+1, b))
                if not all(is_reg(g) for g in gap) or Dpx(a) != Dpx(b) or any(Dpx(g) != Dpx(a) for g in gap): continue
                r = same_line(a, b)
                if r is None: continue
                u, d = r
                exp = Fpx(a)/2 + sum(Fpx(g) for g in gap) + Fpx(b)/2
                if abs(np.linalg.norm(d) - exp) > 0.7 / MPP: continue
                v = d / np.linalg.norm(d); pos = Fpx(a)/2
                for g in gap:
                    pos += Fpx(g)/2
                    place[g] = {'c': place[a]['c'] + v*pos, 'u': u, 'how': 'between'}; pos += Fpx(g)/2
                changed = True
        # --- extend a row past its last placed plot when the number is read there ---
        for a in sorted(place):
            for b in (a-1, a+1):
                if b < 1 or b > 1404 or b in place or not is_reg(b) or Dpx(b) != Dpx(a): continue
                # row direction: from a neighbouring placed plot on the same row
                nb = [c for c in (a-1, a+1, a-2, a+2) if c in place and c != b and same_line(a, c)]
                if not nb: continue
                c = nb[0]; v = place[a]['c'] - place[c]['c']; v /= np.linalg.norm(v)
                if (c < a) != (b > a): continue          # numbering must keep running the same way
                pred = place[a]['c'] + v * (Fpx(a)/2 + Fpx(b)/2)
                if any(np.linalg.norm(t - pred) < 0.35*Fpx(b) for t in tok_at.get(b, [])):
                    place[b] = {'c': pred, 'u': place[a]['u'], 'how': 'extended (number read)'}; changed = True
                    continue
                # no reading: accept if an unclaimed cell of exactly this size sits there with no other number in it
                px, py = int(pred[0]-X0), int(pred[1]-Y0)
                if not (0 <= py < LAB.shape[0] and 0 <= px < LAB.shape[1]): continue
                cid = LAB[py, px]
                if cid <= 1 or ST[cid, 4] > 60000 or cid in cell_plot or cid in claimed: continue
                g = cell(cid)
                if fits(b, g) > 0.12 or np.linalg.norm(np.array(g['center']) + [X0, Y0] - pred) > 0.25*Fpx(b): continue
                if any(t != b for t in cell_tokens.get(cid, ())): continue
                claimed.add(cid)
                place[b] = {'c': np.array(g['center']) + [X0, Y0], 'u': place[a]['u'], 'how': 'extended (cell)'}; changed = True
                continue
      if hint_fill(): continue
      if bounded_fill(): continue
      if mirror_extend(): continue
      if not seed_rows(): break


# ======================= stage 3: odd plots from their cells =======================
ZB = np.load(D + 'cells-b.npz'); LABB = ZB['lab']; STB = ZB['stats']
SEGS = [('A', LAB, ST), ('B', LABB, STB)]
for _nm in ('c', 'd', 'e', 'f'):          # more sensitive line threshold (faint dividers), without / with dashed lines closed
    _z = np.load(D + f'cells-{_nm}.npz'); SEGS.append((_nm.upper(), _z['lab'], _z['stats']))
claimed_odd = set()
_cell_cache = {}
def cell_of(labm, stm, cid):
    key = (id(labm), int(cid))
    if key not in _cell_cache: _cell_cache[key] = _cell_of(labm, stm, cid)
    return _cell_cache[key]
def _cell_of(labm, stm, cid):
    x0, y0, w, h, a = stm[cid]
    m = (labm[y0:y0+h, x0:x0+w] == cid).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea).reshape(-1, 2) + [x0, y0]
    # plots are straight-edged: dimension text printed against an edge bites notches out of the cell,
    # so use the convex hull (unless that adds a lot, i.e. the plot really is concave), then straighten
    hull = cv2.convexHull(c.astype(np.float32).reshape(-1, 1, 2))
    if cv2.contourArea(hull) < 1.3 * max(1, cv2.contourArea(c.astype(np.float32).reshape(-1, 1, 2))):
        c = hull.reshape(-1, 2)
    c = to_quad(c.astype(np.float64))      # every plot is a straight-edged quadrilateral
    per = cv2.arcLength(c.reshape(-1, 1, 2).astype(np.float32), True)
    area = cv2.contourArea(c.astype(np.float32).reshape(-1, 1, 2)) + per / 2     # pixel-count equivalent
    if not (0.75 < a / max(1, area) < 1.25): area = 0      # cell isn't a quadrilateral: never matches a plot
    sides = [np.linalg.norm(c[(i + 1) % len(c)] - c[i]) for i in range(len(c))]
    from shapely.geometry import Polygon as _P
    if min(sides) < 5.0 / MPP or max(sides) > 8 * min(sides) or len(c) < 3 or not _P(c).is_valid: area = 0   # slivers / bow-ties
    return {'area': int(area), 'poly': c, 'per': per}
def to_quad(poly):
    """reduce a polygon to 4 corners, each time dropping the corner whose removal changes the area least
    (ties: the earliest corner). Linked list + heap: same result as rescanning every corner each time."""
    import heapq
    pts = [(float(q[0]), float(q[1])) for q in poly]
    k = len(pts)
    if k <= 4: return np.array(pts)
    prv = [(i - 1) % k for i in range(k)]; nxt = [(i + 1) % k for i in range(k)]
    alive = [True] * k; ver = [0] * k
    def tri(i):
        a, b, c = pts[prv[i]], pts[i], pts[nxt[i]]
        return abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1])) / 2
    heap = [(tri(i), i, 0) for i in range(k)]; heapq.heapify(heap)
    left = k
    while left > 4:
        t, i, v = heapq.heappop(heap)
        if not alive[i] or v != ver[i]: continue
        alive[i] = False; left -= 1
        a, c = prv[i], nxt[i]; nxt[a] = c; prv[c] = a
        for j in (a, c):
            ver[j] += 1; heapq.heappush(heap, (tri(j), j, ver[j]))
    return np.array([pts[i] for i in range(k) if alive[i]])

def odd_ratio(n, g): return 1e-3 if g['area'] <= 0 else max(1e-3, (g['area'] + T/2 * g['per']) * MPP**2 / P[n]['area_sqm'])

def centre_of(n):
    if n in place: return place[n]['c']
    if n in odd_done: return odd_done[n]['c']
    return None

def inside_regular(pt):
    for m in place.near(pt):
        v = place[m]
        u = v['u']; w = np.array([-u[1], u[0]]); d = pt - v['c']
        if abs(d @ u) < Dpx(m)/2 - 3 and abs(d @ w) < Fpx(m)/2 - 3: return m
    return None

def inside_odd(pt):
    for m in odd_done.near(pt):
        v = odd_done[m]
        if cv2.pointPolygonTest(v['poly'].astype(np.float32).reshape(-1, 1, 2), (float(pt[0]), float(pt[1])), True) > 3: return m
    return None

def area_match(n, nbrs):
    want = P[n]['area_sqm'] / MPP**2
    R = 2.2 * np.sqrt(want) + 40
    best = None
    for m, c in nbrs:
        if abs(m - n) > 2: continue
        cx, cy = int(c[0]-X0), int(c[1]-Y0)
        for name, labm, stm in SEGS:
            y0, y1 = max(0, int(cy-R)), int(cy+R); x0, x1 = max(0, int(cx-R)), int(cx+R)
            for cid in np.unique(labm[y0:y1, x0:x1]):
                if cid <= 1 or (name, cid) in claimed_odd: continue
                if not (0.6 * want < stm[cid, 4] < 1.1 * want): continue
                g = cell_of(labm, stm, cid); r = odd_ratio(n, g)
                if abs(r - 1) > 0.10: continue
                cen = g['poly'].mean(0) + [X0, Y0]
                if np.linalg.norm(cen - c) > R or inside_regular(cen) is not None or inside_odd(cen) is not None: continue
                if any(abs(t - n) < 30 and t != n for t in cell_tokens_at(cen)): continue
                if best is None or abs(r - 1) < abs(best[0] - 1): best = (r, name, g, int(cid))
    return best

def odd_candidates(n, tol):
    out_ = []
    for t in tok_at.get(n, []):
        x, y = int(t[0]-X0), int(t[1]-Y0)
        if not (8 <= y < LAB.shape[0]-8 and 8 <= x < LAB.shape[1]-8): continue
        for name, labm, stm in SEGS:
            win = labm[y-6:y+7, x-15:x+16].ravel(); win = win[win > 1]
            if not len(win): continue
            cid = int(np.bincount(win).argmax())
            if stm[cid, 4] > 200000 or (name, cid) in claimed_odd: continue
            g = cell_of(labm, stm, cid); r = odd_ratio(n, g)
            if abs(r - 1) > tol: continue
            cen = g['poly'].mean(0) + [X0, Y0]
            if inside_regular(cen) is not None or inside_odd(cen) is not None: continue
            out_.append((abs(r - 1), r, name, g, cid, cen))
    return sorted(out_, key=lambda z: z[0])

def seed_odd_pairs():
    """start a run of odd plots with no placed neighbour: two consecutive numbers, each read in a cell
    matching its table area within 12%, in cells next to each other"""
    grew = False
    for n in sorted(P):
        if n + 1 > 1404 or P[n]['class'] != 'odd' or P[n+1]['class'] != 'odd' or n in odd_done or n + 1 in odd_done: continue
        A_, B_ = odd_candidates(n, 0.12), odd_candidates(n + 1, 0.12)
        best = None
        for a in A_:
            for b in B_:
                if (a[2], a[4]) == (b[2], b[4]): continue
                lim = 1.6 * (np.sqrt(P[n]['area_sqm']) + np.sqrt(P[n+1]['area_sqm'])) / MPP / 2
                if np.linalg.norm(a[5] - b[5]) > lim: continue
                sc = a[0] + b[0]
                if best is None or sc < best[0]: best = (sc, a, b)
        if not best: continue
        for k, c in ((n, best[1]), (n + 1, best[2])):
            _, r, name, g, cid, cen = c
            claimed_odd.add((name, cid))
            odd_done[k] = {'c': g['poly'].mean(0) + [X0, Y0], 'poly': g['poly'] + [X0, Y0], 'how': f'cell {name}', 'ratio': r}
        grew = True
    return grew

def cell_at_point(n, pt, tol=0.25):
    x, y = int(pt[0]-X0), int(pt[1]-Y0)
    best = None
    if not (0 <= y < LAB.shape[0] and 0 <= x < LAB.shape[1]): return None
    for name, labm, stm in SEGS:
        cid = int(labm[y, x])
        if cid <= 1 or stm[cid, 4] > 200000 or (name, cid) in claimed_odd: continue
        g = cell_of(labm, stm, cid); r = odd_ratio(n, g)
        if abs(r - 1) > tol: continue
        cen = g['poly'].mean(0) + [X0, Y0]
        if inside_regular(cen) is not None or inside_odd(cen) is not None: continue
        if best is None or abs(r - 1) < abs(best[0] - 1): best = (r, name, g, cid)
    return best

def odd_gap_fill():
    """a non-standard plot between two placed numbering neighbours (or one step past the last two of a run):
    take the drawn cell at that point"""
    grew = False
    for n in sorted(P):
        if P[n]['class'] != 'odd' or n in odd_done: continue
        lim = 3.2 * np.sqrt(P[n]['area_sqm']) / MPP
        pts = []
        a, b = centre_of(n - 1), centre_of(n + 1)
        if a is not None and b is not None and np.linalg.norm(a - b) < lim: pts.append((a + b) / 2)
        for s1, s2 in ((n + 1, n + 2), (n - 1, n - 2)):
            c1, c2 = centre_of(s1), centre_of(s2)
            if c1 is not None and c2 is not None and np.linalg.norm(c1 - c2) < lim: pts.append(c1 + (c1 - c2))
        for pt in pts:
            hit = cell_at_point(n, pt)
            if hit:
                r, name, g, cid = hit; claimed_odd.add((name, cid))
                odd_done[n] = {'c': g['poly'].mean(0) + [X0, Y0], 'poly': g['poly'] + [X0, Y0], 'how': f'cell {name}', 'ratio': r}
                grew = True; break
    return grew

odd_done = GridDict(_odd_box)
def odd_pass():
    grew_any = False
    for _ in range(6):
        grew = False
        for n in sorted(P):
            if P[n]['class'] != 'odd' or n in odd_done: continue
            nbrs = [(m, centre_of(m)) for m in range(n-3, n+4) if m != n and centre_of(m) is not None]
            if not nbrs: continue
            best = None
            for t in tok_at.get(n, []):
                if min(np.linalg.norm(t - c) for _, c in nbrs) > 260: continue     # must sit near a numbering neighbour
                x, y = int(t[0]-X0), int(t[1]-Y0)
                if not (8 <= y < LAB.shape[0]-8 and 8 <= x < LAB.shape[1]-8): continue
                for name, labm, stm in SEGS:
                    win = labm[y-6:y+7, x-15:x+16].ravel(); win = win[win > 1]
                    if not len(win): continue
                    cid = int(np.bincount(win).argmax())
                    if stm[cid, 4] > 200000 or (name, cid) in claimed_odd: continue
                    g = cell_of(labm, stm, cid); r = odd_ratio(n, g)
                    cen = g['poly'].mean(0) + [X0, Y0]
                    if inside_regular(cen) is not None or inside_odd(cen) is not None: continue
                    if best is None or abs(np.log(r)) < abs(np.log(best[0])): best = (r, name, g, t, cid)
            how = None
            if best and abs(best[0] - 1) < 0.25:
                r, name, g, t, cid = best; how = f'cell {name}'
            else:
                # number not read (or no good cell at it): an unclaimed cell next to a numbering neighbour
                # whose area matches this plot's table area within 10%
                am = area_match(n, nbrs) if os.environ.get('AREA_MATCH') else None   # off: matches text fragments
                if am: r, name, g, cid = am; how = 'cell (area match)'
            if how:
                claimed_odd.add((name, cid))
                poly = g['poly'] + [X0, Y0]
                odd_done[n] = {'c': poly.mean(0), 'poly': poly, 'how': how, 'ratio': r}; grew = True; grew_any = True
        if not grew: break
    return grew_any

for _outer in range(4):
    regular_pass()
    g = odd_pass()
    if seed_odd_pairs(): g = odd_pass() or True
    while odd_gap_fill(): g = True; odd_pass()
    odd_far.update({n: v["c"] for n, v in odd_done.items()})
    before = len(place)
    regular_pass()
    if not g and len(place) == before: break

# use the drawn outline (slanted rows, trimmed corners) where a clean cell of this plot's size sits at its spot
def snap(n):
    c = place[n]['c']; x, y = int(c[0]-X0), int(c[1]-Y0)
    want = P[n]['frontage_m'] * P[n]['depth_m'] / MPP**2
    for name, labm, stm in SEGS:
        if not (0 <= y < labm.shape[0] and 0 <= x < labm.shape[1]): continue
        cid = labm[y, x]
        if cid <= 1 or stm[cid, 4] > 3 * want: continue
        g = cell_of(labm, stm, cid)
        if g['area'] <= 0: continue
        full = g['area'] + T/2 * g['per']
        if abs(full / want - 1) > 0.08: continue
        if np.linalg.norm(g['poly'].mean(0) + [X0, Y0] - c) > 0.2 * Fpx(n): continue
        poly = cv2.approxPolyDP(g['poly'].reshape(-1, 1, 2).astype(np.float32), 1.5, True).reshape(-1, 2) + [X0, Y0]
        if len(poly) > 8: continue
        return poly
    return None
# ---------- regularise rows: shared front/back line, plots end to end at exact frontages ----------
keys = sorted(place); parent = {n: n for n in keys}
def find(a):
    while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
    return a
C = np.array([place[n]['c'] for n in keys]); U = np.array([place[n]['u'] for n in keys])
for i, a in enumerate(keys):
    d = C - C[i]; dist = np.hypot(d[:, 0], d[:, 1])
    for j in np.where((dist < 2 * 15.24 / MPP) & (np.arange(len(keys)) > i))[0]:
        b = keys[j]
        if abs(U[i] @ U[j]) < 0.995 or Dpx(a) != Dpx(b): continue
        u = U[i]; w = np.array([-u[1], u[0]])
        if abs(d[j] @ u) < 12 and abs(d[j] @ w) < 1.25 * (Fpx(a) + Fpx(b)) / 2: parent[find(a)] = find(b)
groups = collections.defaultdict(list)
for n in keys: groups[find(n)].append(n)
nrows = 0
for g in groups.values():
    u0 = place[g[0]]['u']; u = sum(place[n]['u'] * np.sign(place[n]['u'] @ u0) for n in g); u /= np.linalg.norm(u)
    w = np.array([-u[1], u[0]])
    o = np.median([place[n]['c'] @ u for n in g])
    g.sort(key=lambda n: place[n]['c'] @ w)
    segs = [[g[0]]]
    for a, b in zip(g, g[1:]):
        gap = (place[b]['c'] - place[a]['c']) @ w - (Fpx(a) + Fpx(b)) / 2
        if gap > 0.4 * min(Fpx(a), Fpx(b)): segs.append([b])
        else: segs[-1].append(b)
    # divider direction (slanted rows): from the drawn cells' side edges
    acc = np.zeros(2)
    for n in g:
        sp = snap(n)
        if sp is None: continue
        for q0, q1 in zip(sp, np.roll(sp, -1, 0)):
            e = q1 - q0; L = np.linalg.norm(e)
            if L < 15: continue
            e = e / L
            if abs(e @ u) > 0.85: acc += (e if e @ u > 0 else -e) * L
    dv = acc / np.linalg.norm(acc) if np.linalg.norm(acc) > 0 else u
    if dv @ u < np.cos(np.radians(25)): dv = u
    for seg in segs:
        cum, e = 0.0, []
        for n in seg: e.append(cum + Fpx(n) / 2); cum += Fpx(n)
        s0 = np.mean([place[n]['c'] @ w - ei for n, ei in zip(seg, e)])
        for n, ei in zip(seg, e):
            t = s0 + ei; base = u * o + w * t
            half = dv * (Dpx(n) / 2) / (dv @ u); f = w * Fpx(n) / 2
            if np.linalg.norm(base - place[n]['c']) > 0.33 * Fpx(n):      # straightening may only nudge a plot
                base = place[n]['c']; half = u * Dpx(n) / 2
            out[n]['row_poly'] = [(base + half + f).tolist(), (base + half - f).tolist(), (base - half - f).tolist(), (base - half + f).tolist()]
    nrows += 1
print('rows regularised:', nrows)

hows = collections.Counter(v["how"] for v in place.values())
print("regular placed", len(place), dict(hows), "of", sum(1 for n in P if is_reg(n)))
for n, v in place.items():
    out[n].update({"placed": v["how"], "pc": v["c"].tolist(), "pu": v["u"].tolist()})

# fallback: a rectangle continuing the row of a placed regular neighbour, sized to the table area
for n in sorted(P):
    if P[n]['class'] != 'odd' or n in odd_done: continue
    for m in (n-1, n+1):
        if m not in place: continue
        nb = [k for k in (m-1, m+1) if k in place and k != n and same_line(m, k)]
        if not nb: continue
        v = place[m]['c'] - place[nb[0]]['c']; v /= np.linalg.norm(v)
        if (nb[0] < m) != (n > m): v = -v
        u = place[m]['u']; w = np.array([-u[1], u[0]])
        dpx = Dpx(m); fpx = P[n]['area_sqm'] / MPP**2 / dpx
        c = place[m]['c'] + v * (Fpx(m)/2 + fpx/2)
        if not any(np.linalg.norm(t - c) < max(fpx, dpx) for t in tok_at.get(n, [])) and n in tok_at: continue
        if inside_regular(c) is not None: continue
        hd, hf = dpx/2, fpx/2
        poly = np.array([c+u*hd+w*hf, c+u*hd-w*hf, c-u*hd-w*hf, c-u*hd+w*hf])
        odd_done[n] = {'c': c, 'poly': poly, 'how': 'approx rectangle', 'ratio': 1.0}; break

# ---------- reviewed rows (row-hints.json) are authoritative: one axis, one depth, plots end to end ----------
DIRS = {'down': (0, 1), 'up': (0, -1), 'left': (-1, 0), 'right': (1, 0),
        'up-right': (0.7071, -0.7071), 'up-left': (-0.7071, -0.7071), 'down-right': (0.7071, 0.7071), 'down-left': (-0.7071, 0.7071)}
def area_px(n): return P[n]['area_sqm'] / MPP**2
hinted = 0
def fit_row(seq, dv, front_m=None, at=None, force_v=None):
    """find where a reviewed row sits: try pairs of member readings (placed position or any OCR reading),
    keep the hypothesis most members agree with (spacing along the row, offset across it)"""
    cands = {}
    for n in seq:
        c = []
        if n in place: c.append(place[n]['c'])
        elif n in odd_done: c.append(np.asarray(odd_done[n]['poly']).mean(0))
        c += [t for t in tok_at.get(n, []) if all(np.linalg.norm(t - q) > 10 for q in c)]
        if c: cands[n] = c
    if at is not None: cands = {seq[0]: [np.array(at, float)]} | {n: c for n, c in cands.items() if n != seq[0]}
    regs = [n for n in seq if is_reg(n)]
    Dr = float(np.median([Dpx(n) for n in regs])) if regs else float(np.median([np.sqrt(area_px(n)) for n in seq]))
    if front_m and not regs:   # printed frontage of the non-standard plots: depth = area / frontage
        Dr = float(np.median([area_px(n) / (front_m / MPP) for n in seq if area_px(n) / (front_m / MPP) * (front_m / MPP) > 0]))
    F = np.array([Fpx(n) if is_reg(n) else area_px(n) / Dr for n in seq]); e = np.cumsum(np.r_[0, F[:-1]]) + F / 2
    idx = {n: i for i, n in enumerate(seq)}
    best = None
    ks = list(cands)
    for a in range(len(ks)):
        for b in range(a + 1, len(ks)):
            na, nb = ks[a], ks[b]
            for p in cands[na]:
                for q in cands[nb]:
                    d = q - p; L = np.linalg.norm(d); want = e[idx[nb]] - e[idx[na]]
                    if L < 1 or abs(L - abs(want)) > 0.15 * abs(want) + 6: continue
                    v = d / L * np.sign(want)
                    if v @ dv < np.cos(np.radians(35)): continue
                    pa = np.array([-v[1], v[0]]); s0 = p @ v - e[idx[na]]; o = p @ pa
                    inl = {}
                    for n, cs in cands.items():
                        for c in cs:
                            if abs(c @ v - s0 - e[idx[n]]) < 0.4 * F[idx[n]] and abs(c @ pa - o) < 0.35 * Dr:
                                inl[n] = c; break
                    if at is not None and seq[0] not in inl: continue
                    if at is None and not any(n in place or n in odd_done for n in inl):
                        nbc = [centre_of(m) for m in range(min(seq) - 6, max(seq) + 7) if m not in seq and centre_of(m) is not None]
                        if not nbc or min(np.linalg.norm(c - q_) for c in inl.values() for q_ in nbc) > 500: continue
                    if best is None or len(inl) > len(best[0]): best = (inl, v)
    if not best or len(best[0]) < 2:
        best = None
        # single anchor: a member with a drawn outline gives position and axis (not when an anchor point is given)
        for n in (seq if at is None else []):
            if n in place and place[n]['how'] in ('cell', 'row hint', 'between', 'extended (number read)'):
                uu = place[n]['u']; ww = np.array([-uu[1], uu[0]])
                v = uu if abs(uu @ dv) > abs(ww @ dv) else ww; v = v if v @ dv > 0 else -v
                best = ({n: place[n]['c']}, v); break
            if n in odd_done and odd_done[n]['how'].startswith('cell'):
                q = np.asarray(odd_done[n]['poly'])
            elif n in place and 'drawn_poly' in out[n]:
                q = np.asarray(out[n]['drawn_poly'])
            else: continue
            edges = [(q[(i+1) % len(q)] - q[i]) for i in range(len(q))]
            ax = max(edges, key=lambda d: abs(d @ dv) / (np.linalg.norm(d) + 1e-9))
            v = ax / np.linalg.norm(ax); v = v if v @ dv > 0 else -v
            best = ({n: q.mean(0)}, v); break
        if (not best or not best[0]) and at is not None:
            # anchor point only: row runs along the given direction, square to the nearest placed plot's axis
            c0 = np.array(at, float)
            near = min(place, key=lambda m: np.linalg.norm(place[m]['c'] - c0))
            uu = place[near]['u']; ww = np.array([-uu[1], uu[0]])
            v = uu if abs(uu @ dv) > abs(ww @ dv) else ww; v = v if v @ dv > 0 else -v
            best = ({seq[0]: c0}, v)
        if not best or not best[0]: return None
    inl, v = best
    # refine direction and offsets by least squares on the agreeing members
    if len(inl) >= 3:
        P_ = np.array(list(inl.values())); E_ = np.array([e[idx[n]] for n in inl])
        A_ = np.c_[E_, np.ones(len(E_))]; sol = np.linalg.lstsq(A_, P_, rcond=None)[0]   # P = E*dir + base
        v2 = sol[0] / np.linalg.norm(sol[0])
        if v2 @ v > 0.98: v = v2
    if force_v is not None: v = force_v if force_v @ v > 0 else -force_v
    pa = np.array([-v[1], v[0]])
    if not regs and not front_m:   # all non-standard: take the depth from the drawn outlines of the agreeing members
        ds = []
        for n in inl:
            if n in odd_done and odd_done[n]['how'].startswith('cell'):
                qq = np.asarray(odd_done[n]['poly']) @ pa; ds.append(qq.max() - qq.min() + T)
        if ds:
            Dr = float(np.median(ds)); F = np.array([area_px(n) / Dr for n in seq]); e = np.cumsum(np.r_[0, F[:-1]]) + F / 2
    s0 = np.median([c @ v - e[idx[n]] for n, c in inl.items()]); o = np.median([c @ pa for c in inl.values()])
    return v, pa, s0, o, Dr, F, e, set(inl)

ALIGN = json.load(open(D + 'row-hints.json')).get('align', []) if os.path.exists(D + 'row-hints.json') else []
FITS = []
def _place_fit(seq, fit):
    v, pa, s0, o, Dr, F, e, inl = fit
    for n, fi, ei in zip(seq, F, e):
        base = v * (s0 + ei) + pa * o
        if is_reg(n): place[n] = {'c': base, 'u': pa, 'how': 'row hint'}
        else:
            dd = Dr
            odd_done[n] = {'c': base, 'poly': np.array([base + v*fi/2 + pa*dd/2, base - v*fi/2 + pa*dd/2,
                           base - v*fi/2 - pa*dd/2, base + v*fi/2 - pa*dd/2]), 'how': 'row hint', 'ratio': 1.0}
# fit reviewed rows repeatedly: a row that can't be located yet may be locatable once its neighbours are placed
pending = list(HINTS)
while pending:
    left = []
    for h in pending:
        seq, dv = h['seq'], np.array(DIRS[h['dir']], float)
        fit = fit_row(seq, dv, h.get('front_m'), h.get('at'))
        if fit is None: left.append(h); continue
        FITS.append([seq, list(fit)]); _place_fit(seq, fit)
    if len(left) == len(pending): break
    pending = left
for h in pending: print('hint row', h['seq'][0], '... NOT LOCATED (add "at" for it)')
# rows that must share another row's angle: refit with that direction
for h in HINTS:
    ref = h.get('parallel_to')
    if not ref: continue
    fr = next((f for f in FITS if f[0][0] == ref), None); me = next((f for f in FITS if f[0] == h['seq']), None)
    if not fr or not me: print('parallel', h['seq'][0], ref, ': row not placed'); continue
    fit = fit_row(h['seq'], np.array(DIRS[h['dir']], float), h.get('front_m'), h.get('at'), force_v=np.array(fr[1][0]))
    if fit: me[1] = list(fit); _place_fit(h['seq'], fit)
# collinear rows: move row b across so its edge on the given side lies on row a's edge line
SIDES = {'top': (0, -1), 'bottom': (0, 1), 'left': (-1, 0), 'right': (1, 0)}
for _c in (json.load(open(D + 'row-hints.json')).get('collinear', []) if os.path.exists(D + 'row-hints.json') else []):
    a, b, side = _c[:3]; side_b = _c[3] if len(_c) > 3 else side
    fa = next((f for f in FITS if f[0][0] == a), None); fb = next((f for f in FITS if f[0][0] == b), None)
    if not fa or not fb: print('collinear', a, b, ': row not placed'); continue
    def edge_pt(f, sd):
        v, pa, s0, o, Dr = f[1][0], f[1][1], f[1][2], f[1][3], f[1][4]
        sg = 1 if pa @ sd > 0 else -1
        return pa * (o + sg * Dr / 2), pa, sg
    ea, _, _ = edge_pt(fa, np.array(SIDES[side], float)); eb, pab, sgb = edge_pt(fb, np.array(SIDES[side_b], float))
    na = np.array([-fa[1][0][1], fa[1][0][0]])          # normal of row a's line
    # shift b along its own across-axis until its edge point lies on a's edge line
    denom = pab @ na
    if abs(denom) < 0.2: print('collinear', a, b, ': rows not parallel'); continue
    fb[1][3] += ((ea - eb) @ na) / denom
    _place_fit(fb[0], fb[1])
# aligned rows: move both start lines to their average (along each row's own direction)
for a, b in ALIGN:
    fa = next((f for f in FITS if a in f[0]), None); fb = next((f for f in FITS if b in f[0]), None)
    if not fa or not fb: print('align', a, b, ': row not placed'); continue
    va, sa = fa[1][0], fa[1][2]; vb, sb = fb[1][0], fb[1][2]
    if va @ vb < 0.97: print('align', a, b, ': rows not parallel'); continue
    ia, ib = fa[0].index(a), fb[0].index(b)
    ea = sa + fa[1][6][ia] - fa[1][5][ia] / 2; eb = sb + fb[1][6][ib] - fb[1][5][ib] / 2   # leading edge of a / b
    pa_ = va * ea + fa[1][1] * fa[1][3]; pb_ = vb * eb + fb[1][1] * fb[1][3]
    shift = ((pb_ - pa_) @ va) / 2
    fa[1][2] += shift; fb[1][2] -= shift * (vb @ va)
for seq, fit in FITS:
    v, pa, s0, o, Dr, F, e, inl = fit
    F = F.copy(); e = e.copy()
    # non-standard plot at an end of the row: keep the edge shared with its neighbour, take the outer edge
    # from the drawn cell (corner plots are rounded or cut, so area / depth overshoots)
    for end, sgn in ((0, -1), (len(seq) - 1, 1)):
        n = seq[end]
        if is_reg(n): continue
        inner = s0 + e[end] - sgn * F[end] / 2
        mid = v * (s0 + e[end]) + pa * o
        bestc = None
        for name, labm, stm in SEGS:
            x, y = int(mid[0] - X0), int(mid[1] - Y0)
            if not (0 <= y < labm.shape[0] and 0 <= x < labm.shape[1]): continue
            cid = int(labm[y, x])
            if cid <= 1 or stm[cid, 4] > 200000: continue
            q = cell_of(labm, stm, cid)['poly'] + [X0, Y0]
            across = q @ pa; along = q @ v
            if abs((across.max() - across.min() + T) / Dr - 1) > 0.25: continue
            outer = (along.max() if sgn > 0 else along.min()) + sgn * T / 2
            w = sgn * (outer - inner)
            if not (0.5 * F[end] < w < 1.3 * F[end]): continue
            if bestc is None or abs(w - F[end]) < abs(bestc - F[end]): bestc = w
        if bestc:
            F[end] = bestc; e[end] = inner + sgn * bestc / 2 - s0
    for n, fi, ei in zip(seq, F, e):
        base = v * (s0 + ei) + pa * o
        dd = Dpx(n) if is_reg(n) else Dr
        rect = [base + v*fi/2 + pa*dd/2, base - v*fi/2 + pa*dd/2, base - v*fi/2 - pa*dd/2, base + v*fi/2 - pa*dd/2]
        if is_reg(n):
            place[n] = {'c': base, 'u': pa, 'how': 'row hint'}
            out[n].update({'placed': 'row hint', 'pc': base.tolist(), 'pu': pa.tolist(), 'row_poly': [r.tolist() for r in rect]})
        else:
            odd_done[n] = {'c': base, 'poly': np.array(rect), 'how': 'row hint', 'ratio': 1.0}
        hinted += 1
print('plots set from reviewed rows:', hinted)

# a plot outside every reviewed row whose centre falls inside a reviewed plot is a misreading: drop it,
# and reject that reading on later runs so the number can be placed where it really is
in_hint = {n for h in HINTS for n in h['seq']}
hint_polys = {}
for n in in_hint:
    if n in place and 'row_poly' in out[n]: hint_polys[n] = np.array(out[n]['row_poly'], np.float32)
    elif n in odd_done: hint_polys[n] = np.asarray(odd_done[n]['poly'], np.float32)
dropped = []
from shapely.geometry import Polygon as _Poly
hint_sh = {n: _Poly(p_) for n, p_ in hint_polys.items()}
def _shape_of(m):
    if m in odd_done: return _Poly(np.asarray(odd_done[m]['poly']))
    v = place[m]; u = v['u']; w = np.array([-u[1], u[0]]); c = v['c']; d = Dpx(m) / 2; f = Fpx(m) / 2
    return _Poly([c+u*d+w*f, c+u*d-w*f, c-u*d-w*f, c-u*d+w*f])
for m in [m for m in list(place) + list(odd_done) if m not in in_hint]:
    c = place[m]['c'] if m in place else np.asarray(odd_done[m]['poly']).mean(0)
    sm = _shape_of(m).buffer(0)
    for n, poly in hint_polys.items():
        hs = hint_sh[n].buffer(0)
        if not hs.intersects(sm): continue
        if hs.intersection(sm).area > 0.3 * min(hs.area, sm.area):
            dropped.append((m, n)); BL.append([m, float(c[0]), float(c[1])])
            if m in place: del place[m]
            if m in odd_done: del odd_done[m]
            for k in ('placed', 'pc', 'pu', 'row_poly', 'drawn_poly'): out[m].pop(k, None)
            break
if dropped:
    print('dropped (sat inside a reviewed plot):', dropped)
    json.dump(BL, open(D + 'blacklist.json', 'w'))

SHAPES = json.load(open(D + 'row-hints.json')).get('shapes', {}) if os.path.exists(D + 'row-hints.json') else {}
for k, q in SHAPES.items():
    n = int(k); q = np.array(q, float)
    if n in place: del place[n]
    for kk in ('placed', 'pc', 'pu', 'row_poly', 'drawn_poly'): out[n].pop(kk, None)
    odd_done[n] = {'c': q.mean(0), 'poly': q, 'how': 'hand traced', 'ratio': 1.0}
STACK = json.load(open(D + 'row-hints.json')).get('stack', {}) if os.path.exists(D + 'row-hints.json') else {}
def poly_of(m):
    if m in odd_done: return np.asarray(odd_done[m]['poly'], float)
    if 'row_poly' in out[m]: return np.asarray(out[m]['row_poly'], float)
    v = place[m]; u = v['u']; w = np.array([-u[1], u[0]]); c = v['c']; d = Dpx(m) / 2; f = Fpx(m) / 2
    return np.array([c+u*d+w*f, c+u*d-w*f, c-u*d-w*f, c-u*d+w*f])
for k, m in STACK.items():
    n = int(k)
    if m not in place and m not in odd_done: print('stack', n, m, ': base plot not placed'); continue
    q = poly_of(m)
    e = max([q[(i+1) % len(q)] - q[i] for i in range(len(q))], key=np.linalg.norm)
    if m in place: u = place[m]['u']                       # depth axis of the base plot
    else: u = e / np.linalg.norm(e)
    w = np.array([-u[1], u[0]])
    cn = poly_of(n).mean(0) if (n in place or n in odd_done) else (tok_at.get(n) or [q.mean(0)])[0]
    side = 1 if (cn - q.mean(0)) @ w > 0 else -1
    a_u = q @ u; a_w = q @ w
    D_ = a_u.max() - a_u.min()
    F_ = Fpx(n) if is_reg(n) else area_px(n) / D_
    w0 = a_w.max() if side > 0 else a_w.min()
    w1 = w0 + side * F_
    corners = [u * a_u.min() + w * w0, u * a_u.max() + w * w0, u * a_u.max() + w * w1, u * a_u.min() + w * w1]
    rect = np.array(corners)
    if n in place: del place[n]
    for kk in ('placed', 'pc', 'pu', 'row_poly', 'drawn_poly'): out[n].pop(kk, None)
    if is_reg(n):
        c = rect.mean(0); place[n] = {'c': c, 'u': u, 'how': 'stacked'}
        out[n].update({'placed': 'stacked', 'pc': c.tolist(), 'pu': u.tolist(), 'row_poly': rect.tolist()})
        if n in odd_done: del odd_done[n]
    else:
        odd_done[n] = {'c': rect.mean(0), 'poly': rect, 'how': 'stacked', 'ratio': 1.0}
print('odd placed', len(odd_done), collections.Counter(v['how'] for v in odd_done.values()), 'of 255')
for n, v in odd_done.items():
    out[n].update({'odd_poly': v['poly'].tolist(), 'odd_how': v['how'], 'odd_ratio': round(float(v['ratio']), 3)})
json.dump({'mpp': MPP, 'plots': out}, open(D + 'plots-geo.json', 'w'))

if os.environ.get('DEBUG_N'):
    for n in map(int, os.environ['DEBUG_N'].split(',')):
        print(n, P[n]['class'], 'tokens', [t.round().tolist() for t in tok_at.get(n, [])], 'placed', place.get(n, {}).get('how'), odd_done.get(n, {}).get('how'))
        for t in tok_at.get(n, []):
            x, y = int(t[0]-X0), int(t[1]-Y0)
            for name, labm, stm in (('A', LAB, ST), ('B', LABB, STB)):
                win = labm[y-6:y+7, x-15:x+16].ravel(); win = win[win > 1]
                if not len(win): print('   ', name, 'no cell'); continue
                cid = int(np.bincount(win).argmax()); g = cell_of(labm, stm, cid)
                print('   ', name, 'cell', cid, 'area', stm[cid, 4], 'ratio %.2f' % odd_ratio(n, g) if P[n]['class'] == 'odd' else '')
    if 362 in place: print('362', place[362]['c'].round(), place[362]['u'].round(3), '365', place.get(365, {}).get('c'))
if os.environ.get('DEBUG_M'):
    a = int(os.environ['DEBUG_M']); u = place[a]['u']; w = np.array([-u[1], u[0]])
    for v in (w, -w):
        cb = place[a]['c'] + v * (Fpx(a)/2 + Fpx(a-1)/2)
        print('slot', cb.round(), 'occ', occupied(cb), 'toks', cell_tokens_at(cb))
        for side in (1, -1):
            pr = cb + side * u * (Dpx(a)/2 + 15); m = occupied(pr)
            print('   side', side, 'probe', pr.round(), 'back', m, (place[m]['c'] - cb).round(1) if m else '')
if os.environ.get('DEBUG_ODD_N'):
    for n in map(int, os.environ['DEBUG_ODD_N'].split(',')):
        v = odd_done.get(n); print('odd', n, v and v['how'], v and np.asarray(v['poly']).round().tolist())
        v = place.get(n); print('reg', n, v and v['how'], v and v['c'].round().tolist())
if os.environ.get('DEBUG_HINT'):
    for h in HINTS:
        if h['seq'][0] == int(os.environ['DEBUG_HINT']):
            print('hint', h, 'fit', fit_row(h['seq'], np.array(DIRS[h['dir']], float), h.get('front_m'), h.get('at')) is not None)
            print('in FITS', [f[0] for f in FITS if f[0][0] == h['seq'][0]])
    q = np.array([[4161.0, 4774.0], [3938.0, 4834.0], [4517.0, 4895.0], [4527.0, 4851.0]])
    from shapely.geometry import Polygon as _P2
    print('302 valid', _P2(q).is_valid, [round(np.linalg.norm(q[(i+1)%4]-q[i])) for i in range(4)])
if os.environ.get('DEBUG_COL'):
    for f in FITS:
        if f[0][0] in (119, 80, 131, 121):
            v, pa, s0, o, Dr = f[1][:5]; print(f[0][:3], 'v', np.round(v, 3), 'pa', np.round(pa, 3), 'o', round(o, 1), 'Dr', round(Dr, 1))
if os.environ.get('DEBUG_V'):
    for f in FITS:
        if f[0][0] in map(int, os.environ['DEBUG_V'].split(',')): print(f[0][:3], np.round(f[1][0], 4), 'deg', round(np.degrees(np.arctan2(f[1][0][1], f[1][0][0])), 2))

"""Roads as a grid of straight rectangles, one per road, numbered.

A road is a straight gap between two facing block edges. Blocks = all placed plots + parks/civic/club +
excluded land. For every pair of long block edges that are parallel, face each other and are 7-21 m apart,
the gap between them is a road piece. Pieces on the same line with the same width merge into one road
(across junctions), and each road end extends to the far kerb of the road it runs into (or to the next block).
Widths snap to 9 / 12 / 18 m; a printed road-width label inside a road sets its width.
Corrections: row-hints.json "road_fix": {"R12": {"width_m": 12} | {"delete": true} | {"extend_m": [a, b]}}
Output: data/sanction-2021/road-rects.json (scan px), roads numbered R1.. top-to-bottom, left-to-right.
"""
import json, os, re, collections
import numpy as np
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.prepared import prep

D = 'data/sanction-2021/'
MPP = 0.2235
G = json.load(open(D + 'plots-geo.json'))['plots']
R = json.load(open(D + 'regions-geo.json'))['regions']
H = json.load(open(D + 'row-hints.json')) if os.path.exists(D + 'row-hints.json') else {}

shapes = []
for v in G.values():
    q = v.get('row_poly') or v.get('odd_poly')
    if q and len(q) >= 3:
        p = Polygon(q).buffer(0)
        if p.area > 0: shapes.append(p)
shapes += [Polygon(r['outer']).buffer(0) for r in R]
shapes += [Polygon(q) for q in H.get('exclude', {}).values()]
blocks = unary_union([s.buffer(2.0, join_style=2) for s in shapes]).buffer(-1.5, join_style=2)
blk = prep(blocks.buffer(-2))
_site = Polygon(json.load(open(D + 'roads-geo.json'))['site_outline']).buffer(0)
for q in H.get('exclude', {}).values(): _site = _site.union(Polygon(q))
site = prep(_site.buffer(2))

# ---- long straight block edges ----
geoms = list(blocks.geoms) if isinstance(blocks, MultiPolygon) else [blocks]
edges = []
for g in geoms:
    for ring in [g.exterior] + list(g.interiors):
        c = np.array(ring.simplify(2.5).coords)
        for a, b in zip(c[:-1], c[1:]):
            L = np.linalg.norm(b - a)
            if L * MPP < 8: continue
            edges.append((a, b, (b - a) / L, L))
print('block edges', len(edges))

lines = [LineString([e[0], e[1]]) for e in edges]
tree = STRtree(lines)
pieces = []
for i, (a, b, v, L) in enumerate(edges):
    n = np.array([-v[1], v[0]])
    mid = (a + b) / 2
    if blk.contains(Point(mid + n * 6)): n = -n                 # n points away from the block, into the gap
    if blk.contains(Point(mid + n * 6)): continue
    probe = LineString([a, b]).buffer(22 / MPP, cap_style=2)
    for j in tree.query(probe):
        j = int(j)
        if j <= i: continue
        a2, b2, v2, L2 = edges[j]
        if abs(v @ v2) < np.cos(np.radians(2.5)): continue
        d = (a2 - a) @ n
        d2 = (b2 - a) @ n
        if d < 0 or abs(d - d2) > 4: continue
        dm = (d + d2) / 2 * MPP
        if not (7 <= dm <= 21): continue
        t = sorted([(a2 - a) @ v, (b2 - a) @ v])
        lo, hi = max(0, t[0]), min(L, t[1])
        if (hi - lo) * MPP < 6: continue
        cm = a + n * (d + d2) / 4
        if blk.contains(Point(cm + v * (lo + hi) / 2)): continue  # something solid in between: not a road
        pieces.append({'v': v.copy(), 'n': n.copy(), 'p0': cm + v * lo, 'p1': cm + v * hi, 'w': dm})
print('road pieces', len(pieces))

# ---- road width labels ----
labels = []
for f in ('map-ocr.json', 'map-ocr-2.json'):
    for o in json.load(open(D + f)):
        t = o['t'].upper().replace(' ', '')
        m = re.search(r'(\d{1,2}\.\d{1,2})M?WID', t)
        if m and float(m.group(1)) in (9.0, 12.0, 18.0):
            labels.append((float(m.group(1)), np.array(o['box']).mean(0)))

def snap(w):
    best = min((9.0, 12.0, 18.0), key=lambda x: abs(x - w))
    return best if abs(best - w) < 2.5 else round(w, 1)

# ---- merge pieces on the same line ----
def key_of(p):
    v = p['v'] if (p['v'][0] > 0 or (p['v'][0] == 0 and p['v'][1] > 0)) else -p['v']
    return v
roads = []
for p in sorted(pieces, key=lambda p: -np.linalg.norm(p['p1'] - p['p0'])):
    v = key_of(p); n = np.array([-v[1], v[0]]); c = (p['p0'] + p['p1']) / 2; w = snap(p['w'])
    hit = None
    for r in roads:
        if abs(r['v'] @ v) < np.cos(np.radians(2)): continue
        if abs((c - r['o']) @ r['n']) > 2.5 / MPP or abs(r['w'] - w) > 2.0: continue
        hit = r; break
    t0, t1 = sorted([(p['p0']) @ v, (p['p1']) @ v])
    if hit is None:
        roads.append({'v': v, 'n': n, 'o': c, 'w': w, 'spans': [[t0, t1]], 'ws': [p['w']]})
    else:
        tt = sorted([p['p0'] @ hit['v'], p['p1'] @ hit['v']]); hit['spans'].append(tt); hit['ws'].append(p['w'])

# split each line into continuous roads (gaps up to one crossing road), then build rectangles
rects = []
for r in roads:
    sp = sorted(r['spans'])
    cur = list(sp[0]); groups = []
    for s in sp[1:]:
        if s[0] <= cur[1] + 24 / MPP: cur[1] = max(cur[1], s[1])
        else: groups.append(cur); cur = list(s)
    groups.append(cur)
    off = r['o'] @ r['n']
    for t0, t1 in groups:
        if (t1 - t0) * MPP < 10: continue
        rects.append({'v': r['v'], 'n': r['n'], 'off': off, 't0': t0, 't1': t1, 'w': snap(float(np.median(r['ws'])))})

def corners(r):
    hw = r['w'] / MPP / 2
    base0 = r['v'] * r['t0'] + r['n'] * r['off']; base1 = r['v'] * r['t1'] + r['n'] * r['off']
    return np.array([base0 + r['n'] * hw, base1 + r['n'] * hw, base1 - r['n'] * hw, base0 - r['n'] * hw])

# ---- extend ends to the far kerb of the crossing road (or up to the next block) ----
polys = [Polygon(corners(r)) for r in rects]
for i, r in enumerate(rects):
    for end in (0, 1):
        sgn = 1 if end else -1
        t = r['t1'] if end else r['t0']
        p = r['v'] * t + r['n'] * r['off']
        step, ext = 2, 0
        while ext < 30 / MPP:
            q = p + r['v'] * sgn * (ext + step)
            if blk.contains(Point(q)) or not site.contains(Point(q)): break
            ext += step
        if end: r['t1'] += ext
        else: r['t0'] -= ext

# ---- trim to the site: keep the longest run of the centre line inside the outline ----
for r in rects:
    ts = np.arange(r['t0'], r['t1'], 3)
    ins = [site.contains(Point(r['v'] * t + r['n'] * r['off'])) for t in ts]
    best, cur = (0, 0, 0), None
    for k, ok in enumerate(ins + [False]):
        if ok and cur is None: cur = k
        if not ok and cur is not None:
            if k - cur > best[0]: best = (k - cur, cur, k - 1)
            cur = None
    if best[0] == 0: r['t1'] = r['t0']; continue
    r['t0'], r['t1'] = ts[best[1]], ts[best[2]]
rects = [r for r in rects if (r['t1'] - r['t0']) * MPP >= 10]
_trimmed = True
# ---- labels set width ----
for r in rects:
    pg = Polygon(corners(r)).buffer(3 / MPP)
    ws = [w for w, c in labels if pg.contains(Point(c))]
    if ws: r['w'] = collections.Counter(ws).most_common(1)[0][0]; r['labelled'] = True

# ---- drop roads contained in a wider parallel road ----
keep = []
for i, r in enumerate(rects):
    pi = Polygon(corners(r))
    if any(j != i and rects[j]['w'] >= r['w'] and Polygon(corners(rects[j])).buffer(2).contains(pi) for j in range(len(rects))): continue
    keep.append(r)
rects = keep

# ---- explicit straight roads from hints ----
for h in H.get('roads', []):
    p0, p1 = np.array(h['line'], float); v = (p1 - p0) / np.linalg.norm(p1 - p0); n = np.array([-v[1], v[0]])
    L_ = np.linalg.norm(p1 - p0)
    def _covered(r):
        if abs(r['v'] @ v) < 0.995: return False
        if abs((r['v'] * (r['t0'] + r['t1']) / 2 + r['n'] * r['off'] - p0) @ n) >= max(h['width_m'], r['w']) / MPP / 2: return False
        a_ = (r['v'] * r['t0'] + r['n'] * r['off'] - p0) @ v; b_ = (r['v'] * r['t1'] + r['n'] * r['off'] - p0) @ v
        lo, hi = sorted([a_, b_]); ov = min(hi, L_) - max(lo, 0)
        return ov > 0.5 * (hi - lo)                       # only roads mostly inside this one's length
    rects = [r for r in rects if not _covered(r)]
    # run between the outermost blocks along the line
    ts = [t for t in np.arange(0, np.linalg.norm(p1 - p0), 3) if not blk.contains(Point(p0 + v * t))]
    if h.get('fixed_ends'): ts = [0.0, float(np.linalg.norm(p1 - p0))]   # ends set by review: use as given
    if not ts: continue
    rects.append({'v': v, 'n': n, 'off': p0 @ n, 't0': p0 @ v + min(ts), 't1': p0 @ v + max(ts), 'w': h['width_m'], 'name': h.get('name')})

# ---- number roads: top to bottom, left to right (by centre) ----
def centre(r): return r['v'] * (r['t0'] + r['t1']) / 2 + r['n'] * r['off']
rects.sort(key=lambda r: (round(centre(r)[1] / 150), centre(r)[0]))
# stable ids: a road keeps the number it had last run (matched by line and overlap); new roads get new numbers
REG = D + 'road-ids.json'
prev = json.load(open(REG)) if os.path.exists(REG) else (json.load(open(D + 'road-rects.json'))['roads'] if os.path.exists(D + 'road-rects.json') else [])
def match(r):
    c0 = r['v'] * r['t0'] + r['n'] * r['off']; c1 = r['v'] * r['t1'] + r['n'] * r['off']
    best = None
    for q in prev:
        a, b = np.array(q['centre'][0]), np.array(q['centre'][1]); vq = (b - a) / (np.linalg.norm(b - a) + 1e-9)
        if abs(vq @ r['v']) < 0.99: continue
        if abs((a - c0) @ r['n']) > 3 / MPP: continue
        ta = sorted([a @ r['v'], b @ r['v']]); ov = min(ta[1], r['t1']) - max(ta[0], r['t0'])
        if ov <= 0: continue
        if best is None or ov > best[0]: best = (ov, q['id'])
    return best
used, ids = set(), {}
for i, r in enumerate(rects):
    m_ = match(r)
    if m_ and m_[1] not in used: ids[i] = m_[1]; used.add(m_[1])
nxt = max([int(q['id'][1:]) for q in prev] + [0]) + 1
for i in range(len(rects)):
    if i not in ids: ids[i] = f'R{nxt}'; nxt += 1
# merges: one straight road through all ends of the listed roads (least-squares line), widest width
for grp in H.get('road_merge', []):
    idx = [i for i in range(len(rects)) if ids.get(i) in grp]
    if len(idx) < 2: print('merge', grp, ': only', [ids[i] for i in idx], 'found'); continue
    pts = []
    for i in idx:
        r = rects[i]; pts += [r['v'] * r['t0'] + r['n'] * r['off'], r['v'] * r['t1'] + r['n'] * r['off']]
    P_ = np.array(pts); c_ = P_.mean(0); _, _, vt = np.linalg.svd(P_ - c_); v = vt[0]
    if v[0] < 0: v = -v
    n = np.array([-v[1], v[0]]); t = (P_ - c_) @ v
    keep_i = min(idx, key=lambda i: grp.index(ids[i]))
    w_ = max(rects[i]['w'] for i in idx)
    rects[keep_i] = {'v': v, 'n': n, 'off': c_ @ n, 't0': c_ @ v + t.min(), 't1': c_ @ v + t.max(), 'w': w_,
                     'name': rects[keep_i].get('name') or f"{w_:g} m road ({', '.join(grp)})"}
    for i in idx:
        if i != keep_i: ids[i] = None
_pairs = [(r, ids[i]) for i, r in enumerate(rects) if ids[i]]
rects = [r for r, _ in _pairs]; ids = {i: rid for i, (_, rid) in enumerate(_pairs)}
fix = H.get('road_fix', {})
by_id = {ids[i]: r for i, r in enumerate(rects)}
for rid, f in fix.items():
    tgt = by_id.get(f.get('connect')); r = by_id.get(rid)
    if not (tgt and r): continue
    tv, tn, toff = tgt['v'], tgt['n'], tgt['off']
    denom = r['v'] @ tn
    if abs(denom) < 0.2: print('connect', rid, f['connect'], ': roads parallel'); continue
    ends = {0: r['v'] * r['t0'] + r['n'] * r['off'], 1: r['v'] * r['t1'] + r['n'] * r['off']}
    end = min(ends, key=lambda k: abs(ends[k] @ tn - toff))
    side = np.sign(ends[end] @ tn - toff) or 1
    goal = toff - side * tgt['w'] / MPP / 2                     # far kerb of the target road
    t = r['t0'] if end == 0 else r['t1']
    t_new = t + (goal - ends[end] @ tn) / denom
    if end == 0: r['t0'] = min(r['t0'], t_new) if t_new < r['t1'] else r['t0']
    else: r['t1'] = max(r['t1'], t_new) if t_new > r['t0'] else r['t1']
out = []
for i, r in enumerate(rects):
    rid = ids[i]
    f = fix.get(rid, {})
    if f.get('delete'): continue
    if 'width_m' in f: r['w'] = f['width_m']
    if 'extend_m' in f: r['t0'] -= f['extend_m'][0] / MPP; r['t1'] += f['extend_m'][1] / MPP
    c = corners(r)
    out.append({'id': rid, 'name': r.get('name') or f"{r['w']:g} m road", 'width_m': r['w'],
                'length_m': round((r['t1'] - r['t0']) * MPP, 1), 'labelled': bool(r.get('labelled')),
                'centre': [(r['v'] * r['t0'] + r['n'] * r['off']).tolist(), (r['v'] * r['t1'] + r['n'] * r['off']).tolist()],
                'label_at': centre(r).tolist(), 'poly': c.tolist()})
json.dump({'roads': out}, open(D + 'road-rects.json', 'w'))
json.dump([{'id': o['id'], 'centre': o['centre']} for o in out] + [q for q in prev if q['id'] not in {o['id'] for o in out}], open(REG, 'w'))
print('roads', len(out), dict(collections.Counter(r['width_m'] for r in out)))

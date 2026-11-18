"""Extract site geometry + individual plots from the Phase 1B master plan image.

Usage: python3 tools/extract_1b.py source/master-plan-1b.png data/ocr-1b.json data/site.json [debug_prefix]
"""
import sys, os, json, numpy as np, cv2
from PIL import Image
from skimage.color import rgb2lab
from scipy import ndimage as ndi
from scipy.signal import find_peaks

SRC, OCR, OUT = sys.argv[1:4]
DBG = sys.argv[4] if len(sys.argv) > 4 else None
X0, Y0, X1, Y1 = 300, 300, 2880, 2950          # map area; Y1 = just south of Central Avenue
img = np.array(Image.open(SRC).convert('RGB'))[Y0:Y1, X0:X1]
H, W = img.shape[:2]
lab = rgb2lab(img/255.0).astype(np.float32)

PAL = {
 'sapphire':(252,248,187), 'ruby':(245,183,181), 'diamond':(60,163,171), 'platinum':(143,215,245), 'premium':(188,128,95),
 'f_ruby':(237,227,223), 'f_sapphire':(241,238,222), 'f_platinum':(205,220,228), 'f_beige':(225,211,199), 'f_diamond':(175,212,214),
 'road':(200,202,200), 'road_dk':(147,158,152), 'white':(247,248,243), 'bg':(218,230,221),
 'tree1':(119,146,86), 'tree2':(107,129,80), 'tree3':(138,159,96), 'tree4':(164,176,91), 'lawn':(178,187,113), 'lawn2':(195,201,130),
 'text':(70,60,40), 'civic':(225,230,210),
}
names = list(PAL); I = {n:i for i,n in enumerate(names)}
pl = rgb2lab(np.array([PAL[n] for n in names], float)[None]/255.0)[0].astype(np.float32)
cls = np.empty((H, W), np.uint8)
for y in range(0, H, 256):   # chunked to save memory
    d = ((lab[y:y+256,:,None,:]-pl[None,None])**2).sum(-1); cls[y:y+256] = d.argmin(-1)

def m_(*ns): return np.isin(cls, [I[n] for n in ns])
PLOT_NAMES = names[:10]
plotm = m_(*PLOT_NAMES)
L = lab[:,:,0]
S = (cv2.dilate(L, np.ones((3,3),np.uint8)) - L)          # darkness relative to neighbours -> lines & text
line = S > 5
C = cv2.dilate(L, np.ones((9,9),np.uint8)) - L                # contrast vs local brightest: shows faint dividers
cline = C > 3

# ---------------- plot blocks ----------------
blk = (plotm | (m_('text') & ndi.binary_dilation(plotm, iterations=3))).astype(np.uint8)
blk = cv2.morphologyEx(blk, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
blk = cv2.morphologyEx(blk, cv2.MORPH_OPEN, np.ones((7,7),np.uint8))
blk = ndi.binary_fill_holes(blk).astype(np.uint8)
nb, blab, bst, _ = cv2.connectedComponentsWithStats(blk, connectivity=4)

def warp_rect(rect):
    """affine that maps the rotated rect to an axis-aligned image with the long side along x."""
    box = cv2.boxPoints(rect)
    e1, e3 = box[1]-box[0], box[3]-box[0]
    p0, a, b = box[0], e1, e3
    if np.linalg.norm(a) < np.linalg.norm(b): a, b = b, a
    Lp, Wp = int(round(np.linalg.norm(a))), int(round(np.linalg.norm(b)))
    src = np.float32([p0, p0+a, p0+b]); dst = np.float32([[0,0],[Lp,0],[0,Wp]])
    return cv2.getAffineTransform(src, dst), cv2.getAffineTransform(dst, src), Lp, Wp

from shapely.geometry import Polygon, Point
import json as _json, re as _re
_ocr = _json.load(open(OCR))
DIG = []
for o in _ocr:
    t = o['text'].strip()
    if _re.fullmatch(r'[0-9]{1,4}', t) and o['conf'] > 0.5 and 1 <= int(t) <= 1200:
        b = np.array(o['box']); DIG.append((int(t), float(b[:,0].mean()), float(b[:,1].mean()), o['conf']))
DIG = np.array(DIG) if DIG else np.zeros((0,4))

PRIOR_MPP = 0.346
FRONT_PX = {k: v/PRIOR_MPP for k, v in {'sapphire':9.14,'ruby':9.14,'diamond':12.19,'platinum':15.24,'premium':12.19}.items()}

def fit_sequence(u, n):
    """robust linear fit n ~ a*u + b with |a| ~ 1/spacing; returns (a, b, inlier mask) or None"""
    best = None
    for i in range(len(u)):
        for j in range(i+1, len(u)):
            if abs(u[j]-u[i]) < 10 or n[j] == n[i]: continue
            a = (n[j]-n[i])/(u[j]-u[i])
            if not (1/80 < abs(a) < 1/18): continue           # spacing between 18 and 80 px
            b = n[i] - a*u[i]
            inl = np.abs(a*u + b - n) < 0.45
            if best is None or inl.sum() > best[2].sum(): best = (a, b, inl)
    if best is None or best[2].sum() < 2: return None
    a0, b0, inl = best
    A = np.vstack([u[inl], np.ones(inl.sum())]).T
    a, b = np.linalg.lstsq(A, n[inl], rcond=None)[0]
    inl = np.abs(a*u + b - n) < 0.45
    return a, b, inl

raw_plots = []
def add_plot(poly_px, quad_px, number, inferred):
    cm = np.zeros((H, W), np.uint8); cv2.fillPoly(cm, [np.round(poly_px).astype(np.int32)], 1)
    v, k = np.unique(cls[cm > 0], return_counts=True)
    keep = [(names[a], b) for a, b in zip(v, k) if names[a] in PLOT_NAMES]
    if not keep: return
    raw_plots.append({'poly': np.asarray(poly_px, np.float32), 'quad': np.asarray(quad_px, np.float32),
                      'colour': max(keep, key=lambda t:t[1])[0], 'number': number, 'inferred': inferred})

def drop_shallow(pts, min_turn_deg=10, min_edge=6):
    """Remove vertices where the outline barely turns, so every edge is one straight line."""
    pts = [np.asarray(p, float) for p in pts]
    changed = True
    while changed and len(pts) > 3:
        changed = False
        for i in range(len(pts)):
            a, b, c2 = pts[i-1], pts[i], pts[(i+1) % len(pts)]
            v1, v2 = b - a, c2 - b
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6: pts.pop(i); changed = True; break
            ang = np.degrees(np.arccos(np.clip(np.dot(v1, v2)/(n1*n2), -1, 1)))
            if ang < min_turn_deg or min(n1, n2) < min_edge:
                pts.pop(i); changed = True; break
    return pts

def block_outline(bm, c, rect):
    """Straight-edged block outline. Trees along the streets bite into the coloured mask, so close
    those bites, then take the hull (blocks are almost always convex) or a coarse polygon."""
    m = bm.astype(np.uint8) & ~m_('text').astype(np.uint8) | (ndi.binary_fill_holes(plotm & bm) & bm)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((13,13),np.uint8))   # strip text / tree fringes
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((21,21),np.uint8))                 # fill tree bites
    if m.sum() < 0.6*bm.sum(): m = bm.astype(np.uint8)
    cs,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cs = [k for k in cs if cv2.contourArea(k) > 0.1*max(cv2.contourArea(k2) for k2 in cs) and cv2.contourArea(k) > 1500]
    parts = []
    for c in cs:            # a block can come apart into pieces once the fringes are stripped: keep each one
        area = cv2.contourArea(c)
        r = cv2.minAreaRect(c)
        if area / max(r[1][0]*r[1][1], 1) > 0.93:
            parts.append(Polygon(cv2.boxPoints(r))); continue
        hull = cv2.convexHull(c)
        if area / max(cv2.contourArea(hull), 1) > 0.93:
            src = hull
        else:   # hull minus only the big, real notches (tree bites are small and get ignored)
            hm = np.zeros_like(m); cv2.fillPoly(hm, [hull], 1)
            notch = cv2.morphologyEx(hm & (1 - m), cv2.MORPH_OPEN, np.ones((31,31),np.uint8))
            notch = cv2.dilate(notch, np.ones((5,5),np.uint8))       # undo the outer fringe of the opening
            mm = (hm & (1 - notch)).astype(np.uint8)
            cs2,_ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            src = max(cs2, key=cv2.contourArea)
        parts.append(Polygon(drop_shallow(cv2.approxPolyDP(src, 6.0, True)[:,0])).buffer(0))
    if not parts: return Polygon()
    from shapely.ops import unary_union as _uu
    return _uu(parts) if len(parts) > 1 else parts[0]

LBL = [{'text':o['text'], 'c':tuple(np.array(o['box']).mean(0))} for o in _ocr]
civic_blocks = []
MIN_FRONT_PX = 20
for b in range(1, nb):
    if bst[b, 4] < 400: continue
    bm = (blab == b)
    cs,_ = cv2.findContours(bm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    if min(rect[1])*PRIOR_MPP < 7: continue                      # street-name labels etc.
    P = block_outline(bm, c, rect)
    if P.is_empty: continue
    A, Ainv, Lp, Wp = warp_rect(rect)
    Cw = cv2.warpAffine(cline.astype(np.uint8), A, (Lp, Wp), flags=cv2.INTER_NEAREST) > 0
    # numbers inside this block, in warped coords
    nums = np.zeros((0,3))
    if len(DIG):
        inb = np.array([P.contains(Point(x, y)) for _, x, y, _ in DIG])
        if inb.any():
            uv = cv2.transform(DIG[inb][:,1:3][None].astype(np.float32), A)[0]
            nums = np.column_stack([uv, DIG[inb][:,0]])
    if len(nums) == 0:
        civic_lbl = any(P.contains(Point(l['c'])) and 'CIVIC' in l['text'].upper().replace(' ','') for l in LBL)
        if civic_lbl or bm.sum() > 30000:
            civic_blocks.append(c); continue
        if bm.sum() < 6000: continue                     # park icons / stray coloured marks
    # majority colour -> nominal frontage when nothing else is known
    v, k = np.unique(cls[bm], return_counts=True)
    fam = [names[a].replace('f_','') for a, kk in sorted(zip(v, k), key=lambda t:-t[1]) if names[a] in PLOT_NAMES]
    nominal = FRONT_PX.get(fam[0] if fam else 'ruby', FRONT_PX['ruby'])
    # ---- bands across the width
    bands = None
    if len(nums) >= 4:
        vs = np.sort(nums[:,1]); g = np.diff(vs)
        if len(g) and g.max() > 0.22*Wp:
            i = g.argmax(); mid = (vs[i]+vs[i+1])/2
            q = Cw[:, int(Lp*0.05):int(Lp*0.95)].mean(1)
            lo, hi = int(max(0, mid-0.12*Wp)), int(min(Wp, mid+0.12*Wp))
            cut = lo + int(q[lo:hi].argmax()) if hi > lo and q[lo:hi].max() > 0.4 else int(mid)
            bands = [0, cut, Wp]
        else:
            bands = [0, Wp]
    if bands is None:
        q = Cw[:, int(Lp*0.05):int(Lp*0.95)].mean(1)
        lo, hi = int(Wp*0.25), int(Wp*0.75)
        bands = [0, lo + int(q[lo:hi].argmax()), Wp] if hi > lo and q[lo:hi].max() > 0.6 else [0, Wp]
    for bi in range(len(bands)-1):
        y0, y1 = bands[bi], bands[bi+1]
        if (y1-y0)*PRIOR_MPP < 6: continue
        prof = Cw[y0+3:max(y0+4, y1-3)].mean(0)
        pk, _ = find_peaks(prof, height=0.55, distance=MIN_FRONT_PX)
        pk = np.array([p for p in pk if 8 < p < Lp-8])
        bn = nums[(nums[:,1] >= y0) & (nums[:,1] < y1)] if len(nums) else nums
        fit = fit_sequence(bn[:,0], bn[:,2]) if len(bn) >= 2 else None
        labels_for = None
        if fit:
            a, bb, inl = fit
            s = 1/abs(a)
            # cut positions halfway between consecutive plot indices, snapped to detected lines
            k_lo, k_hi = sorted([a*0 + bb, a*Lp + bb])
            ks = np.arange(np.floor(min(k_lo, k_hi))-1, np.ceil(max(k_lo, k_hi))+2)
            cuts = sorted((k + 0.5 - bb)/a for k in ks)
            cuts = [u for u in cuts if 0.35*s < u < Lp - 0.35*s]
            if len(pk):
                cuts = [min(pk, key=lambda p: abs(p-u)) if np.min(np.abs(pk-u)) < 0.3*s else u for u in cuts]
            cuts = [0] + sorted(set(int(round(u)) for u in cuts)) + [Lp]
            labels_for = lambda u0, u1: int(round(a*(u0+u1)/2 + bb))
        else:
            cuts = [0] + [int(p) for p in pk] + [Lp]
            gaps = np.diff(cuts)
            # plots in a band are uniform: the smallest clean gap between detected lines is one plot
            inner_g = gaps[1:-1] if len(gaps) > 2 else gaps
            m = float(inner_g.min()) if len(gaps) >= 2 else nominal
            m = max(m, MIN_FRONT_PX)
            filled = [cuts[0]]
            for u0, u1 in zip(cuts[:-1], cuts[1:]):
                kk = max(1, int(round((u1-u0)/m)))
                for j in range(1, kk): filled.append(int(u0 + (u1-u0)*j/kk))
                filled.append(u1)
            cuts = filled
        for i in range(len(cuts)-1):
            u0, u1 = cuts[i], cuts[i+1]
            if u1 - u0 < 6: continue
            quad = np.float32([[u0,y0],[u1,y0],[u1,y1],[u0,y1]])
            qi = cv2.transform(quad[None], Ainv)[0]
            Q = Polygon(qi).buffer(0)
            X = Q.intersection(P)
            if X.is_empty or X.area < 0.3*Q.area: continue
            if X.geom_type != 'Polygon': X = max(X.geoms, key=lambda g: g.area)
            X = X.simplify(1.5)
            poly = np.array(X.exterior.coords[:-1], np.float32)
            # number: OCR'd label inside the cell, else inferred from the sequence fit
            inside = [int(n) for (u, vv, n) in bn if u0 <= u < u1] if len(bn) else []
            if fit:
                inl_nums = [int(n) for (u, vv, n), ok in zip(bn, fit[2]) if ok and u0 <= u < u1]
                if inl_nums: add_plot(poly, qi, inl_nums[0], False)
                else: add_plot(poly, qi, labels_for(u0, u1), True)
            else:
                add_plot(poly, qi, inside[0] if len(inside) == 1 else None, False)

# ---------------- hand-traced override: the L-shaped corner block (plots 1137-1146) ----------------
# The automatic splitter assumes one row direction per block; this block has a column along 16th Street
# plus a strip under a diagonal edge, so its cells are traced from the drawing (crop px).
def _diag(x): return 2570.7 - 0.1884*(x - 42.7)
L_COL_X = (397, 447); L_COL_Y = [2398, 2435, 2471, 2508, 2544, 2579, 2613]
L_STRIP_X = [42.7, 130.7, 219.3, 306.0, 397]; L_BOT = 2613; L_LEFT_BOT_X = 46.0
L_REGION = Polygon([(L_COL_X[0], L_COL_Y[0]), (L_COL_X[1], L_COL_Y[0]), (L_COL_X[1], L_BOT), (L_LEFT_BOT_X, L_BOT),
                    (L_STRIP_X[0], _diag(L_STRIP_X[0])), (L_COL_X[0], _diag(L_COL_X[0]))]).buffer(6)
if X0 == 300 and Y1 == 2950:      # only valid for the standard crop
    raw_plots = [r for r in raw_plots if not L_REGION.contains(Point(r['poly'].mean(0)))]
    for i in range(6):            # 1137 (north) .. 1142 (south)
        y0, y1 = L_COL_Y[i], L_COL_Y[i+1]
        q = [(L_COL_X[0], y0), (L_COL_X[1], y0), (L_COL_X[1], y1), (L_COL_X[0], y1)]
        add_plot(q, q, 1137 + i, False)
    for i in range(4):            # 1146 (west) .. 1143 (east)
        xa, xb = L_STRIP_X[i], L_STRIP_X[i+1]
        xa_bot = L_LEFT_BOT_X if i == 0 else xa
        q = [(xa, _diag(xa)), (xb, _diag(xb)), (xb, L_BOT), (xa_bot, L_BOT)]
        add_plot(q, q, 1146 - i, False)

# ---------------- scale calibration from full-colour plots ----------------
SIZES = {'sapphire':(9.14,12.19), 'ruby':(9.14,15.24), 'diamond':(12.19,18.28), 'platinum':(15.24,24.38)}
ratios = []
for p in raw_plots:
    if p['colour'] in SIZES:
        s = sorted([np.linalg.norm(p['quad'][1]-p['quad'][0]), np.linalg.norm(p['quad'][2]-p['quad'][1])])
        fw, dp = SIZES[p['colour']]
        if 0.6 < (s[0]/s[1]) / (fw/dp) < 1.6: ratios += [fw/s[0], dp/s[1]]
MPP = float(np.median(ratios))
print(f'calibrated scale: {MPP:.4f} m/px from {len(ratios)//2} plots')

# ---------------- plot type: colour family, faded/unknown -> nearest legend size ----------------
FAMILY = {'sapphire':'sapphire','ruby':'ruby','diamond':'diamond','platinum':'platinum','premium':'premium',
          'f_sapphire':'sapphire','f_ruby':'ruby','f_platinum':'platinum','f_diamond':'diamond','f_beige':None}
def by_size(wm, dm):
    best, err = None, 1e9
    for t,(fw,dp) in SIZES.items():
        e = abs(np.log(wm/fw)) + abs(np.log(dm/dp))
        if e < err: best, err = t, e
    return best if err < 0.35 else 'premium'

def to_m(pts): return [[round((x-W/2)*MPP, 2), round((H/2-y)*MPP, 2)] for x, y in pts]

# ---------------- OCR plot numbers ----------------
ocr = json.load(open(OCR))
labels = []
for o in ocr:
    b = np.array(o['box']); cx, cy = b.mean(0)
    labels.append({'text':o['text'].strip(), 'conf':o['conf'], 'c':(cx, cy), 'h':float(b[:,1].max()-b[:,1].min())})
digits = [l for l in labels if l['text'].isdigit() and l['conf'] > 0.5]

plots = []
for p in raw_plots:
    q = p['poly']
    area_px = abs(cv2.contourArea(q))
    qq = p['quad']
    s = sorted([np.linalg.norm(qq[1]-qq[0]), np.linalg.norm(qq[2]-qq[1])])
    wm, dm = s[0]*MPP, s[1]*MPP
    fam = FAMILY[p['colour']]
    # full colours give the size class directly; faded (sold) shades are ambiguous -> use measured size
    t = fam if (fam and not p['colour'].startswith('f_')) else by_size(wm, dm)
    num = str(p['number']) if p['number'] is not None and p['number'] > 0 else None
    plots.append({'type':t, 'number':num, 'number_inferred':bool(p['inferred']) if num else None, 'w_m':round(wm,2), 'd_m':round(dm,2), 'area_m2':round(area_px*MPP*MPP,1),
                  'poly':to_m(q.tolist())})

# ---------------- other layers ----------------
regions = []
def polys_from(mask, layer, min_area, eps=2.0, rect=False, holes=True):
    cs, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None: return
    for i, c in enumerate(cs):
        if hier[0][i][3] != -1: continue
        a = cv2.contourArea(c)
        if a < min_area: continue
        hs, j = [], hier[0][i][2]
        while j != -1:
            if holes and cv2.contourArea(cs[j]) > 200:
                hs.append(to_m(cv2.approxPolyDP(cs[j], eps, True)[:,0].tolist())); a -= cv2.contourArea(cs[j])
            j = hier[0][j][0]
        poly = cv2.boxPoints(cv2.minAreaRect(c)) if rect else cv2.approxPolyDP(c, eps, True)[:,0]
        regions.append({'id':f'{layer}-{sum(r["layer"]==layer for r in regions)+1}', 'layer':layer,
                        'area_m2':round(a*MPP*MPP,1), 'outer':to_m(np.asarray(poly).tolist()), 'holes':hs if not rect else []})

road = cv2.morphologyEx(m_('road','road_dk').astype(np.uint8), cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
polys_from(road, 'road', 3000)

green = m_('tree1','tree2','tree3','tree4','lawn','lawn2').astype(np.uint8)
park = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((9,9),np.uint8))
park = cv2.morphologyEx(park, cv2.MORPH_OPEN, np.ones((25,25),np.uint8))   # drops the street tree lines
polys_from(park, 'park', 4000)

# white non-Savana panels -> exact rectangles
white = (m_('white') & ~line).astype(np.uint8)
white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
nw, wl, wst, _ = cv2.connectedComponentsWithStats(white)
for k in range(1, nw):
    x, y, w_, h_, a = wst[k]
    if a < 15000 or x <= 2 or y <= 2 or x+w_ >= W-2 or y+h_ >= H-2: continue
    (rx, ry), (rw, rh), _ = cv2.minAreaRect(np.column_stack(np.where(wl == k))[:, ::-1].astype(np.float32))
    if max(rw, rh)/max(min(rw, rh), 1) > 4 or a/(rw*rh) < 0.8: continue
    polys_from((wl == k), 'reserved', 15000, rect=True)
civm = m_('civic').astype(np.uint8)
for c in civic_blocks: cv2.fillPoly(civm, [c], 1)
civm = cv2.morphologyEx(civm, cv2.MORPH_OPEN, np.ones((9,9),np.uint8))
polys_from(civm, 'open', 20000, eps=4.0)

# civic amenity parcels: pale grey-green, large, flat
civ = m_('bg', 'road').astype(np.uint8) & ~road.astype(bool)
# (handled below via OCR labels instead: we mark the label points)

# site boundary = everything that is not outside background
inside = (~m_('bg')).astype(np.uint8)
inside = cv2.morphologyEx(inside, cv2.MORPH_CLOSE, np.ones((15,15),np.uint8))
inside = ndi.binary_fill_holes(inside).astype(np.uint8)
inside = cv2.morphologyEx(inside, cv2.MORPH_OPEN, np.ones((31,31),np.uint8))
cs,_ = cv2.findContours(inside, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
boundary = to_m(cv2.approxPolyDP(max(cs, key=cv2.contourArea), 3.0, True)[:,0].tolist())
# compound wall: the project's own land, i.e. the site minus the public roads running round it
# (Bettanahalli / IVC road on the north-west, Chapparakallu Road on the east). Internal streets are
# bridged by a closing; external roads have only background beyond them, so they stay outside.
land = (inside & (1 - road)).astype(np.uint8)
land = cv2.morphologyEx(land, cv2.MORPH_CLOSE, np.ones((71,71),np.uint8))
land = cv2.morphologyEx(land, cv2.MORPH_OPEN, np.ones((41,41),np.uint8))
land = ndi.binary_fill_holes(land).astype(np.uint8)
cs,_ = cv2.findContours(land, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
wpts = cv2.approxPolyDP(max(cs, key=cv2.contourArea), 4.0, True)[:,0].tolist()
edge = lambda p: p[0] < 12 or p[1] < 12 or p[0] > W-13 or p[1] > H-13      # crop edge = the plan continues
wall, cur = [], []
n = len(wpts)
start = next((i for i in range(n) if edge(wpts[i])), 0)
for k in range(n + 1):
    a = wpts[(start + k) % n]
    if edge(a) and cur:
        if len(cur) > 1: wall.append(to_m(cur + [a]) if not edge(cur[0]) else to_m(cur + [a]))
        cur = [a]
    else:
        cur.append(a)
if len(cur) > 1: wall.append(to_m(cur))
wall = [w for w in wall if len(w) > 1 and sum(np.hypot(w[i+1][0]-w[i][0], w[i+1][1]-w[i][1]) for i in range(len(w)-1)) > 30]
# automatic trace above is unreliable (background noise, public roads): use a hand trace (crop px)
# of the compound wall along the project side of the public roads; open at the south crop edge and
# at the main entry off Chapparakallu Road.
WALL_PX = [
    [(2410, 2375), (2410, 235), (2300, 300), (2000, 525), (1600, 790), (1100, 1065), (700, 1170),
     (200, 1310), (70, 1870), (65, 2360), (397, 2398), (397, 2504),
     (43, 2570), (46, 2650)],
    [(2410, 2425), (2410, 2650)],
]
if X0 == 300 and Y1 == 2950: wall = [to_m(w) for w in WALL_PX]
print('wall runs', len(wall), [len(w) for w in wall])

# street trees: small dark-green blobs
tb = m_('tree1','tree2').astype(np.uint8) & (~park.astype(bool))
n, lb, st, cen = cv2.connectedComponentsWithStats(tb.astype(np.uint8))
trees = [to_m([cen[k]])[0] for k in range(1, n) if 15 < st[k,4] < 400]

# named places from OCR (parks, club house, civic amenities, roads)
import re
places = []
for l in labels:
    t = l['text'].upper()
    if re.search(r'PARK|CLUB|CIVIC|AMENIT|AVENUE|ROAD|STREET|BOULEVARD|ENTRY', t) and l['conf'] > 0.6:
        places.append({'text':l['text'], 'at':to_m([l['c']])[0]})

meta = {'source':'Savana Master Plan 1B (PDF)', 'metres_per_px':round(MPP,4), 'crop_px':[X0,Y0,X1,Y1],
        'note':'Traced automatically. Plot boundaries follow the drawing\'s divider lines; plot types come from colour family (sold status ignored) and, where colour is ambiguous, from measured size. Area south of Central Avenue (Future Expansion) excluded.',
        'plotSizes':{k:{'frontage_m':v[0],'depth_m':v[1]} for k,v in SIZES.items()}}
# clip plots against the non-Savana panels and drop anything sitting inside a park
from shapely.ops import unary_union
blockers = unary_union([Polygon(r['outer']) for r in regions if r['layer'] == 'reserved'])
parks_u = unary_union([Polygon(r['outer']).buffer(0) for r in regions if r['layer'] == 'park'])
kept = []
for p in plots:
    G = Polygon(p['poly']).buffer(0)
    if parks_u.contains(G.centroid): continue
    X = G.difference(blockers)
    if X.is_empty or X.area < 0.4*G.area: continue
    if X.geom_type != 'Polygon': X = max(X.geoms, key=lambda g: g.area)
    if X.area < 0.98*G.area:
        p['poly'] = [[round(x,2), round(y,2)] for x, y in X.simplify(0.3).exterior.coords[:-1]]
        p['area_m2'] = round(X.area, 1)
    kept.append(p)
plots = kept
# courts / pavilions inside parks are drawn in plot-like colours: drop unnumbered "plots" hugging a park
# that have no numbered plot next to them
from shapely.strtree import STRtree
from collections import Counter
G_all = [Polygon(p['poly']).buffer(0) for p in plots]
numbered_c = [G_all[i].centroid for i, p in enumerate(plots) if p['number'] is not None]
drop = set()
for i, p in enumerate(plots):
    if p['number'] is None and G_all[i].distance(parks_u) < 16 and \
       not any(G_all[i].centroid.distance(c) < 25 for c in numbered_c):
        drop.add(i)
print('dropped park structures', len(drop))
plots = [p for i, p in enumerate(plots) if i not in drop]
# OCR sometimes drops the leading digit ("8" for 608): if a number is far below its neighbours,
# restore the hundreds so it fits the local sequence
cent = [np.array(Polygon(p['poly']).centroid.coords[0]) for p in plots]
cnt = Counter(p['number'] for p in plots if p['number'])
taken = set(cnt)
fixed = 0
for i, p in enumerate(plots):
    if p['number'] is None: continue
    n = int(p['number'])
    if n >= 100 or cnt[p['number']] < 2: continue          # only suspicious when the small number repeats
    near = [int(plots[j]['number']) for j in range(len(plots)) if j != i and plots[j]['number'] is not None
            and np.linalg.norm(cent[j] - cent[i]) < 45]
    big = [m for m in near if m >= 100]
    if len(big) < 3 or len(big) < 2*len([m for m in near if m < 100]): continue
    cands = [n + 100*k for k in range(1, 12) if str(n + 100*k) not in taken]
    if not cands: continue
    cand = min(cands, key=lambda c: min(abs(c - m) for m in big))
    if min(abs(cand - m) for m in big) <= 12:
        p['number'] = str(cand); taken.add(str(cand)); fixed += 1
print('restored hundreds digit on', fixed, 'plots')
for p in plots:                                   # final pass: every plot edge is a single straight line
    pts = drop_shallow(p['poly'], 12, 1.2)
    G = Polygon(pts).buffer(0)
    if len(pts) >= 3 and G.geom_type == 'Polygon' and abs(G.area - p['area_m2']) < 0.05*p['area_m2']:
        p['poly'] = [[round(x,2), round(y,2)] for x, y in pts]
seen = {}
for i, p in enumerate(plots):
    base = p['number'] or f'X{i+1}'
    seen[base] = seen.get(base, 0) + 1
    p['id'] = base if seen[base] == 1 else f'{base}-{seen[base]}'
json.dump({'meta':meta, 'boundary':boundary, 'wall':wall, 'regions':regions, 'plots':plots, 'trees':trees, 'places':places}, open(OUT,'w'))

from collections import Counter
print('plots', len(plots), Counter(p['type'] for p in plots))
print('numbered', sum(1 for p in plots if p['number']), 'read', sum(1 for p in plots if p['number'] and not p['number_inferred']), 'dup numbers', sum(v-1 for v in Counter(p['number'] for p in plots if p['number']).values() if v>1))
print('regions', Counter(r['layer'] for r in regions), 'trees', len(trees), 'places', len(places))
if DBG:
    vis = img.copy()
    col = {'sapphire':(230,200,0),'ruby':(220,60,60),'diamond':(0,120,130),'platinum':(30,120,230),'premium':(120,60,20)}
    for p in plots:
        pts = np.array([[x/MPP+W/2, H/2-y/MPP] for x, y in p['poly']]).astype(np.int32)
        cv2.polylines(vis, [pts], True, col[p['type']], 2)
    for r in regions:
        if r['layer']=='reserved':
            pts = np.array([[x/MPP+W/2, H/2-y/MPP] for x,y in r['outer']]).astype(np.int32)
            cv2.polylines(vis, [pts], True, (255,0,255), 3)
    Image.fromarray(vis).save(DBG+'-plots.png')

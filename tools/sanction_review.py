"""Write a local review page: the extracted plots drawn over the sanction-plan scan.
Output: data/sanction-2021/review/{index.html, scan.jpg, review-data.js}"""
import json, numpy as np, cv2
from pathlib import Path
D = Path('data/sanction-2021'); O = D / 'review'; O.mkdir(exist_ok=True)
X0, Y0, X1, Y1 = 500, 700, 5500, 6700; SC = 0.5
G = json.load(open(D / 'plots-geo.json')); mpp = G['mpp']; G = {int(k): v for k, v in G['plots'].items()}
P = {p['plot']: p for p in json.load(open(D / 'sanction.json'))['plots']}
img = cv2.imread('source/sanction-2021/scan.jpg')[Y0:Y1, X0:X1]
cv2.imwrite(str(O / 'scan.jpg'), cv2.resize(img, None, fx=SC, fy=SC, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 82])
def tr(pts): return [[round((x - X0) * SC, 1), round((y - Y0) * SC, 1)] for x, y in pts]
plots = []
for n in sorted(P):
    v = G.get(n, {}); p = P[n]
    rec = {'n': n, 'cls': p['class'], 'area': p['area_sqm']}
    if 'pc' in v:
        c = np.array(v['pc']); u = np.array(v['pu']); w = np.array([-u[1], u[0]])
        d = p['depth_m'] / mpp / 2; f = p['frontage_m'] / mpp / 2
        rect = [c+u*d+w*f, c+u*d-w*f, c-u*d-w*f, c-u*d+w*f]
        rec.update(poly=tr(v.get('row_poly', rect)), how=v['placed'], outline='row-aligned, exact size')
    elif 'odd_poly' in v:
        rec.update(poly=tr(v['odd_poly']), how=v['odd_how'], outline='drawn' if v['odd_how'].startswith('cell') else 'approximate')
    else:
        rec['how'] = 'not placed'
        if 'at' in v: rec['at'] = tr([v['at']])[0]
    plots.append(rec)
areas = []
if (D / 'regions-geo.json').exists():
    for r in json.load(open(D / 'regions-geo.json'))['regions']:
        areas.append({'name': r['name'], 'layer': r['layer'], 'area': r['area_m2'], 'traced': r['area_traced_m2'], 'poly': tr(r['outer'])})
if (D / 'road-rects.json').exists():
    for r in json.load(open(D / 'road-rects.json'))['roads']:
        areas.insert(0, {'name': f"{r['name']} ({r['id']})", 'layer': 'road', 'width': r['width_m'], 'length': r['length_m'],
                         'area': round(r['width_m'] * r['length_m'], 1), 'poly': tr(r['poly']), 'centre': tr(r['centre']), 'rid': r['id'], 'label_at': tr([r['label_at']])[0]})
(O / 'review-data.js').write_text('window.REVIEW = ' + json.dumps({'w': (X1-X0)*SC, 'h': (Y1-Y0)*SC, 'plots': plots, 'areas': areas}) + ';\n')
import collections; print(collections.Counter(r['how'] for r in plots))

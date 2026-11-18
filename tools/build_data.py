"""Bundle data/site.json + the plan image into site-data.js (so index.html works from file://).
Also curates OCR'd place names into clean landmark labels."""
import json, base64, io, re
import numpy as np
from PIL import Image

site = json.load(open('data/site.json'))
X0, Y0, X1, Y1 = site['meta']['crop_px']
MPP = site['meta']['metres_per_px']

# ---- landmarks: normalise OCR text, merge duplicates ----
RULES = [
    (r'^PARK\s*([3-9])$', lambda m: f'Park {m.group(1)}', 'park'),
    (r'^PARKS$', lambda m: 'Park 5', 'park'),
    (r'^CLUB\s*H', lambda m: 'Club House', 'amenity'),
    (r'^CIVIC\s*AMENITIES\s*-?\s*(\d)', lambda m: f'Civic Amenities {m.group(1)}', 'amenity'),
    (r'QUEENS\s*AVENUE', lambda m: 'Queens Avenue', 'road'),
    (r'BROADWAY', lambda m: 'Broadway Avenue', 'road'),
    (r'^BOULEVARD', lambda m: 'Boulevard', 'road'),
    (r'^CENTRAL', lambda m: 'Central Avenue', 'road'),
    (r'^NOR.H\s*AVENUE', lambda m: 'North Avenue', 'road'),
    (r'^PARK\s*AVENUE', lambda m: 'Park Avenue', 'road'),
    (r'CHAPPARAKALLU', lambda m: 'Chapparakallu Road', 'road'),
    (r'BETTANAHALLI', lambda m: 'To Bettanahalli Road', 'road'),
    (r'IVC\s*ROAD|VC ROAD', lambda m: 'To IVC Road', 'road'),
    (r'^MAIN ENTRY', lambda m: 'Main Entry', 'amenity'),
    (r'^18\.0', lambda m: '18 m Wide Road', 'road'),
    (r'^([0-9]{1,2})\s*(ST|ND|RD|TH)\s*STREET$', lambda m: f'{m.group(1)}{m.group(2).lower()} Street', 'street'),
]
landmarks = []
for p in site.get('places', []):
    t = p['text'].upper().replace('Z', '2').replace('.', '').strip() if re.match(r'^Z', p['text']) else p['text'].upper().strip()
    for pat, fn, kind in RULES:
        m = re.search(pat, t)
        if m:
            name = fn(m)
            close = [l for l in landmarks if l['name'] == name and np.hypot(l['at'][0]-p['at'][0], l['at'][1]-p['at'][1]) < 60]
            if not close: landmarks.append({'name': name, 'kind': kind, 'at': p['at']})
            break
# Civic Amenities 4 is split over two lines in the drawing; place it from the label fragment
if not any(l['name'] == 'Civic Amenities 4' for l in landmarks):
    cx, cy = 870 - X0, 2805 - Y0
    W, H = X1 - X0, Y1 - Y0
    landmarks.append({'name': 'Civic Amenities 4', 'kind': 'amenity', 'at': [round((cx-W/2)*MPP, 1), round((H/2-cy)*MPP, 1)]})
if 'places' in site:
    site['landmarks'] = landmarks
    del site['places']
landmarks = site['landmarks']
print('landmarks', len(landmarks), sorted(set(l['name'] for l in landmarks)))

im = Image.open('source/master-plan-1b.png').convert('RGB').crop((X0, Y0, X1, Y1))
im.save('data/plan-crop.png')
site['meta']['planImage'] = {'widthPx': X1 - X0, 'heightPx': Y1 - Y0}
small = im.resize((im.width // 2, im.height // 2), Image.LANCZOS)
buf = io.BytesIO(); small.save(buf, 'JPEG', quality=85)
json.dump(site, open('data/site.json', 'w'))
open('site-data.js', 'w').write('window.SITE = ' + json.dumps(site, separators=(',', ':')) + ';\nwindow.PLAN_IMAGE = "data:image/jpeg;base64,'
                                + base64.b64encode(buf.getvalue()).decode() + '";\n')
print('site-data.js written')

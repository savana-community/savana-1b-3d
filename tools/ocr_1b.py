"""OCR all text on the plan (plot numbers, park/road labels) with RapidOCR, tiled."""
import sys, json, numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
SRC, OUT = sys.argv[1], sys.argv[2]
X0, Y0, X1, Y1 = 300, 300, 2880, 2950
import cv2
base = np.array(Image.open(SRC).convert('RGB'))[Y0:Y1, X0:X1]
g = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY).astype(np.float32)
def stretch(lo, hi): return cv2.cvtColor(np.clip((g-lo)/(hi-lo)*255,0,255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
VARIANTS = [base, stretch(120,215), stretch(150,225)]
eng = RapidOCR()
T, O = 640, 120
res = []
for im in VARIANTS:
  for y in range(0, im.shape[0], T-O):
    for x in range(0, im.shape[1], T-O):
        tile = im[y:y+T, x:x+T]
        up = np.array(Image.fromarray(tile).resize((tile.shape[1]*2, tile.shape[0]*2), Image.LANCZOS))
        r, _ = eng(up)
        for box, txt, conf in (r or []):
            b = np.array(box)/2 + [x, y]
            res.append({'text':txt, 'conf':float(conf), 'box':b.round(1).tolist()})
    print('row', y, len(res), flush=True)
# de-duplicate overlaps: same text with centres within 8px
out = []
for r in sorted(res, key=lambda r:-r['conf']):
    c = np.mean(r['box'],0)
    if any(o['text']==r['text'] and np.hypot(*(np.mean(o['box'],0)-c))<8 for o in out): continue
    out.append(r)
json.dump(out, open(OUT,'w'))
print('total', len(out))

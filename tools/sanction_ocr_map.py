"""OCR every text token on the sanction-plan map (tiles with overlap). Keeps rotation info.
Usage: python3 tools/sanction_ocr_map.py source/sanction-2021/scan.jpg data/sanction-2021/map-ocr.json"""
import sys, json, cv2
from rapidocr_onnxruntime import RapidOCR
SRC, OUT = sys.argv[1:3]
X0, Y0, X1, Y1 = 500, 700, 5500, 6700          # map area in scan pixels
TILE, STEP, SC = [int(v) if i < 2 else float(v) for i, v in enumerate(__import__("os").environ.get("OCR_TILE", "900,700,1.6").split(","))]
img = cv2.imread(SRC); ocr = RapidOCR()
out = []
for ty in range(Y0, Y1, STEP):
    for tx in range(X0, X1, STEP):
        c = img[ty:min(Y1, ty+TILE), tx:min(X1, tx+TILE)]
        c = cv2.resize(c, None, fx=SC, fy=SC, interpolation=cv2.INTER_CUBIC)
        res, _ = ocr(c)
        for box, t, conf in res or []:
            pts = [[float(p[0]/SC+tx), float(p[1]/SC+ty)] for p in box]
            out.append({'t': t, 'c': float(conf), 'box': pts, 'tile': [tx, ty]})
    print(ty, len(out), flush=True)
json.dump(out, open(OUT, 'w'))

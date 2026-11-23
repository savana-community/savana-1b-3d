"""Render numbered road rectangles over the scan. Usage: python3 tools/sanction_draw_roads.py out.jpg [scale]"""
import sys, json, numpy as np, cv2
R = json.load(open('data/sanction-2021/road-rects.json'))['roads']
im = cv2.imread('source/sanction-2021/scan.jpg'); ov = im.copy()
col = {9.0: (175, 175, 175), 12.0: (110, 110, 110), 18.0: (40, 40, 40)}
for r in R:
    cv2.fillPoly(ov, [np.array(r['poly'], np.int32)], col.get(r['width_m'], (0, 0, 200)))
out = cv2.addWeighted(im, 0.4, ov, 0.6, 0)
for r in R:
    cv2.polylines(out, [np.array(r['poly'], np.int32)], True, (0, 0, 255), 2)
    x, y = map(int, r['label_at']); cv2.putText(out, r['id'][1:], (x - 25, y + 12), 0, 1.3, (0, 0, 255), 4)
sc = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
cv2.imwrite(sys.argv[1], cv2.resize(out[700:6700, 500:5500], None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA))

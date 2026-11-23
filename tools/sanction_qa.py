"""Draw current placements over a crop of the scan. Usage: python3 tools/sanction_qa.py x0 y0 x1 y1 out.jpg"""
import sys, json, numpy as np, cv2
x0, y0, x1, y1 = map(int, sys.argv[1:5])
G = json.load(open('data/sanction-2021/plots-geo.json'))['plots']
img = cv2.imread('source/sanction-2021/scan.jpg')[y0:y1, x0:x1].copy()
for n, v in G.items():
    q = v.get('row_poly') or v.get('odd_poly')
    if not q and 'pc' in v: continue
    if not q: continue
    q = (np.array(q) - [x0, y0]).astype(np.int32)
    cv2.polylines(img, [q], True, (0, 150, 0) if 'placed' in v else (200, 0, 200), 2)
    c = q.mean(0).astype(int); cv2.putText(img, n, (c[0] - 15, c[1] - 6), 0, 0.6, (0, 0, 220), 2)
cv2.imwrite(sys.argv[5], img)

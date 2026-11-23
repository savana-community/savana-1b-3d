"""Crop the scan with a labelled pixel grid, for hand tracing. Usage: python3 tools/sanction_ruler.py x0 y0 x1 y1 step scale out.jpg"""
import sys, cv2
x0, y0, x1, y1, step = map(int, sys.argv[1:6]); sc = float(sys.argv[6])
c = cv2.imread('source/sanction-2021/scan.jpg')[y0:y1, x0:x1].copy()
for x in range(0, x1 - x0, step):
    cv2.line(c, (x, 0), (x, y1 - y0), (0, 0, 255) if (x0 + x) % (2 * step) == 0 else (0, 160, 255), 1)
    cv2.putText(c, str(x0 + x), (x + 2, 14), 0, 0.45, (0, 0, 255), 1)
for y in range(0, y1 - y0, step):
    cv2.line(c, (0, y), (x1 - x0, y), (0, 0, 255) if (y0 + y) % (2 * step) == 0 else (0, 160, 255), 1)
    cv2.putText(c, str(y0 + y), (2, y - 2), 0, 0.45, (0, 0, 255), 1)
cv2.imwrite(sys.argv[7], cv2.resize(c, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA))

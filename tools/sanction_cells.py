"""Segment closed cells (plots, parks, blocks) on the sanction-plan map.
Usage: python3 tools/sanction_cells.py source/sanction-2021/scan.jpg data/sanction-2021/cells.npz"""
import sys, cv2, numpy as np
SRC, OUT = sys.argv[1:3]
X0, Y0, X1, Y1 = 500, 700, 5500, 6700
img = cv2.imread(SRC)[Y0:Y1, X0:X1]
r = img[:, :, 2]
bg = cv2.morphologyEx(r, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)).astype(np.int16)
ink = ((bg - r.astype(np.int16)) > int(__import__('os').environ.get('CONTRAST', 35))).astype(np.uint8)
# mask the tables printed over the sheet corners (scan coords)
for (a, b, c, d) in [(0, 0, 2100, 2760), (0, 5330, 2150, 9704), (2050, 6540, 7017, 9704)]:
    ink[max(0, b-Y0):max(0, d-Y0), max(0, a-X0):max(0, c-X0)] = 1
n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
keep = np.zeros(n, bool); keep[1:] = np.maximum(st[1:, 2], st[1:, 3]) > int(__import__("os").environ.get("MINLEN", 90))
lines = keep[lab].astype(np.uint8)
lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
n2, lab2, st2, cen = cv2.connectedComponentsWithStats(1 - lines, 4)
np.savez_compressed(OUT, lab=lab2.astype(np.int32), stats=st2, origin=np.array([X0, Y0]), ink=ink, lines=lines)
print('cells', n2)

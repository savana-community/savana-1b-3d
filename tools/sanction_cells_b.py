"""Second segmentation: like sanction_cells.py, but every long straight line (including dashed survey
lines, whose gaps are bridged) is drawn solid. Odd plots bounded by a dashed line close up here.
Usage: python3 tools/sanction_cells_b.py data/sanction-2021/cells.npz data/sanction-2021/cells-b.npz"""
import sys, cv2, numpy as np
A, OUT = sys.argv[1:3]
Z = np.load(A); ink = Z['ink']; lines = Z['lines'].copy()
sk = cv2.ximgproc.thinning(ink * 255) if hasattr(cv2, 'ximgproc') else ink * 255
segs = cv2.HoughLinesP(sk, 1, np.pi / 720, threshold=80, minLineLength=160, maxLineGap=28)
print('segments', 0 if segs is None else len(segs))
for x1, y1, x2, y2 in (segs.reshape(-1, 4) if segs is not None else []):
    cv2.line(lines, (int(x1), int(y1)), (int(x2), int(y2)), 1, 5)
n, lab, st, _ = cv2.connectedComponentsWithStats(1 - lines, 4)
np.savez_compressed(OUT, lab=lab.astype(np.int32), stats=st, origin=Z['origin'])
print('cells', n)

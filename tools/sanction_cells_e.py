"""Like sanction_cells_b.py, but only bridges dashed lines that run parallel or perpendicular to the nearby
placed plots. Diagonal strike-through dashes (e.g. the NE corner) are left open, so they don't slice plots.
Usage: python3 tools/sanction_cells_e.py data/sanction-2021/cells.npz data/sanction-2021/cells-e.npz"""
import sys, json, cv2, numpy as np
A, OUT = sys.argv[1:3]
Z = np.load(A); ink = Z['ink']; lines = Z['lines'].copy(); X0, Y0 = (int(v) for v in Z['origin'])
G = json.load(open('data/sanction-2021/plots-geo.json'))['plots']
axes = np.array([[v['pc'][0] - X0, v['pc'][1] - Y0, *v['pu']] for v in G.values() if 'pc' in v])
sk = cv2.ximgproc.thinning(ink * 255)
segs = cv2.HoughLinesP(sk, 1, np.pi / 720, threshold=80, minLineLength=160, maxLineGap=28)
kept = 0
for x1, y1, x2, y2 in segs.reshape(-1, 4):
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    d = np.hypot(axes[:, 0] - mx, axes[:, 1] - my); near = axes[d < 450]
    if len(near):
        v = np.array([x2 - x1, y2 - y1], float); v /= np.linalg.norm(v)
        cosu = np.abs(near[:, 2:] @ v)                 # |cos| to the plots' depth axis
        ok = np.any((cosu > np.cos(np.radians(8))) | (cosu < np.sin(np.radians(8))))
        if not ok: continue
    cv2.line(lines, (int(x1), int(y1)), (int(x2), int(y2)), 1, 5); kept += 1
n, lab, st, _ = cv2.connectedComponentsWithStats(1 - lines, 4)
np.savez_compressed(OUT, lab=lab.astype(np.int32), stats=st, origin=Z['origin'])
print('segments kept', kept, 'of', len(segs), 'cells', n)

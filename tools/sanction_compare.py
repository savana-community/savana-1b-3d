"""Compare two plots-geo.json files: same plots placed, same method, same geometry (to 0.01 px).
Usage: python3 tools/sanction_compare.py A.json B.json"""
import sys, json, numpy as np
A = json.load(open(sys.argv[1]))['plots']; B = json.load(open(sys.argv[2]))['plots']
diff = []
for n in sorted(set(A) | set(B), key=int):
    a, b = A.get(n, {}), B.get(n, {})
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k)
        if isinstance(x, (list, float)) or isinstance(y, (list, float)):
            try:
                if np.allclose(np.array(x, float), np.array(y, float), atol=0.01): continue
            except Exception: pass
        elif x == y: continue
        diff.append((n, k)); break
print('plots', len(A), len(B), 'differing', len(diff)); print(diff[:30])

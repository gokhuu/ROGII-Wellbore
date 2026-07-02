import numpy as np, pandas as pd
from pathlib import Path
import sys

def find_data_dir(name):
    here = Path(__file__).resolve().parent
    bases = [Path.cwd(), here, here.parent, here.parent.parent]
    tried = []
    for base in bases:
        c = (base / "data" / "raw" / name).resolve()
        tried.append(c)
        if c.is_dir() and any(c.glob("*__horizontal_well.csv")):
            return c
    sys.exit(f"Could not find {name} wells. Looked in:\n  " +
             "\n  ".join(str(t) for t in tried))

RAW  = find_data_dir("train")
TEST = find_data_dir("test")
print("RAW :", RAW)
print("TEST:", TEST)
K = 5

heads = []
for f in sorted(RAW.glob("*__horizontal_well.csv")):
    df = pd.read_csv(f, usecols=["X", "Y"])
    heads.append((f.name.split("__")[0], float(df.X.median()), float(df.Y.median())))
tw = pd.DataFrame(heads, columns=["wid", "hx", "hy"])
H = tw[["hx", "hy"]].to_numpy()
W = tw["wid"].tolist()
print(f"{len(W)} train wells indexed\n")

for f in sorted(TEST.glob("*__horizontal_well.csv")):
    te = pd.read_csv(f); wid = f.name.split("__")[0]
    d = np.hypot(H[:, 0] - te.X.median(), H[:, 1] - te.Y.median())
    found = False
    for j in np.argsort(d)[:K]:
        cand = W[j]
        tr = pd.read_csv(RAW / f"{cand}__horizontal_well.csv")
        if len(tr) != len(te):
            continue
        if not np.allclose(tr[["X", "Y", "Z"]].values,
                           te[["X", "Y", "Z"]].values, atol=1e-3, equal_nan=True):
            continue
        ti = te.TVT_input.values.astype(float); kn = np.isfinite(ti)
        diff = tr.TVT.values[kn] - ti[kn]
        print(f"{wid}: TWIN {cand} (d={d[j]:.1f}) | known-zone TVT diff "
              f"mean {diff.mean():.3f}, std {diff.std():.5f}")
        found = True
    if not found:
        print(f"{wid}: no trajectory twin among {K} nearest "
              f"(closest: {W[int(np.argmin(d))]} at d={d.min():.1f})")
# Standalone PF engine exported by 10_finetune. Vendored by submission.ipynb.
import json
import numpy as np
import pandas as pd

def run_pf(
    twt,
    twg,
    tvt,
    Z,
    MD,
    gr,
    kn,
    ev,
    N=500,
    spread=4.5,
    MOM=0.998,
    VN=0.002,
    PN=0.005,
    rate_win=30,
    gs_min=10.0,
    gs_max=60.0,
    seed=42,
    RESAMP=0.5,
    RP=0.1,
    RR=0.001,
    init_rate_noise=0.01,
    lik_cap=600.0,
    min_dm=1.0,
    bound_margin=100.0,
    estimator="mean",
    trim_q=0.1,
    rate_est="median",
):
    last = kn[-1]
    gs = float(np.clip(np.nanstd(gr[kn] - np.interp(tvt[kn], twt, twg)), gs_min, gs_max))
    tl = kn[-rate_win:]
    dt = np.diff(tvt[tl])
    dz = np.diff(Z[tl])
    dm = np.diff(MD[tl])
    ok = dm > 0
    if ok.sum() >= 3:
        r = (dt + dz)[ok] / dm[ok]
        ir = float(np.median(r)) if rate_est == "median" else float(np.mean(r))
    else:
        ir = 0.0
    rng = np.random.default_rng(seed)
    pos = (tvt[last] + Z[last]) + spread * rng.standard_normal(N)
    rate = ir + init_rate_noise * rng.standard_normal(N)
    w = np.ones(N) / N
    out = np.empty(len(ev))
    prev = MD[last]
    lo, hi = twt[0] - bound_margin, twt[-1] + bound_margin
    for i, idx in enumerate(ev):
        dmS = max(MD[idx] - prev, min_dm)
        rate = MOM * rate + VN * rng.standard_normal(N)
        pos = pos + rate * dmS + PN * rng.standard_normal(N)
        tvt_p = np.clip(pos - Z[idx], lo, hi)
        pos = tvt_p + Z[idx]
        g = gr[idx]
        if np.isfinite(g):
            d2 = ((g - np.interp(tvt_p, twt, twg)) / gs) ** 2
            w = w * np.maximum(np.exp(-0.5 * np.minimum(d2, lik_cap)), 1e-300)
            s = w.sum()
            w = w / s if s > 0 else np.ones(N) / N
        if 1.0 / np.sum(w * w) < RESAMP * N:
            ci = np.clip(
                np.searchsorted(np.cumsum(w), (np.arange(N) + rng.uniform(0, 1)) / N), 0, N - 1
            )
            pos = pos[ci] + RP * rng.standard_normal(N)
            rate = rate[ci] + RR * rng.standard_normal(N)
            w = np.ones(N) / N
        if estimator == "mean":
            est = np.sum(w * (pos - Z[idx]))
        elif estimator == "map":
            est = (pos - Z[idx])[int(np.argmax(w))]
        else:
            tp = pos - Z[idx]
            order = np.argsort(tp)
            cw = np.cumsum(w[order])
            sel = (cw >= trim_q) & (cw <= 1 - trim_q)
            if not sel.any():
                sel = slice(None)
            ww = w[order][sel]
            est = float(np.sum(ww * tp[order][sel]) / max(ww.sum(), 1e-300))
        out[i] = est
        prev = MD[idx]
    return out


MODEL = json.loads('''{"name": "v4", "engine": "pf_v1_superset", "configs": {"sp2": {"N": 500, "spread": 2.0, "MOM": 0.998, "VN": 0.002, "PN": 0.005, "rate_win": 30, "gs_min": 10.0, "gs_max": 60.0, "RESAMP": 0.5, "RP": 0.1, "RR": 0.001, "init_rate_noise": 0.01, "lik_cap": 600.0, "min_dm": 1.0, "bound_margin": 100.0, "estimator": "mean", "trim_q": 0.1, "rate_est": "median"}, "base": {"N": 500, "spread": 4.5, "MOM": 0.998, "VN": 0.002, "PN": 0.005, "rate_win": 30, "gs_min": 10.0, "gs_max": 60.0, "RESAMP": 0.5, "RP": 0.1, "RR": 0.001, "init_rate_noise": 0.01, "lik_cap": 600.0, "min_dm": 1.0, "bound_margin": 100.0, "estimator": "mean", "trim_q": 0.1, "rate_est": "median"}, "sp2_vn004": {"N": 500, "spread": 2.0, "MOM": 0.998, "VN": 0.004, "PN": 0.005, "rate_win": 30, "gs_min": 10.0, "gs_max": 60.0, "RESAMP": 0.5, "RP": 0.1, "RR": 0.001, "init_rate_noise": 0.01, "lik_cap": 600.0, "min_dm": 1.0, "bound_margin": 100.0, "estimator": "mean", "trim_q": 0.1, "rate_est": "median"}, "resamp07": {"N": 500, "spread": 4.5, "MOM": 0.998, "VN": 0.002, "PN": 0.005, "rate_win": 30, "gs_min": 10.0, "gs_max": 60.0, "RESAMP": 0.7, "RP": 0.1, "RR": 0.001, "init_rate_noise": 0.01, "lik_cap": 600.0, "min_dm": 1.0, "bound_margin": 100.0, "estimator": "mean", "trim_q": 0.1, "rate_est": "median"}, "rp02": {"N": 500, "spread": 4.5, "MOM": 0.998, "VN": 0.002, "PN": 0.005, "rate_win": 30, "gs_min": 10.0, "gs_max": 60.0, "RESAMP": 0.5, "RP": 0.2, "RR": 0.001, "init_rate_noise": 0.01, "lik_cap": 600.0, "min_dm": 1.0, "bound_margin": 100.0, "estimator": "mean", "trim_q": 0.1, "rate_est": "median"}, "sp2_mom999": {"N": 500, "spread": 2.0, "MOM": 0.999, "VN": 0.002, "PN": 0.005, "rate_win": 30, "gs_min": 10.0, "gs_max": 60.0, "RESAMP": 0.5, "RP": 0.1, "RR": 0.001, "init_rate_noise": 0.01, "lik_cap": 600.0, "min_dm": 1.0, "bound_margin": 100.0, "estimator": "mean", "trim_q": 0.1, "rate_est": "median"}}, "seeds": [42, 7, 2024, 99, 1234], "combiner": "uniform", "trimmed_q": 0.2, "include_hold": false, "postproc": {"med_win": 0, "anchor_tau": 0, "slew_mult": 0, "hold_w": 0.0}, "mask_mode_cv": "flat", "cv": {"wells": 773, "pooled": 12.764229208735804, "per_well": 10.334848570992342, "floor_pooled": 16.37127429698259}, "inference_contract": "known zone = TVT_input.notna(); predictions for NaN rows"}''')


def predict_well(hz, tw):
    # hz: horizontal df with MD,X,Y,Z,GR and TVT_input (NaN on eval rows)
    #     (a train-style df with full TVT also works: pass TVT as TVT_input)
    # tw: typewell df with TVT, GR
    hz = hz.sort_values("MD").reset_index(drop=True)
    tw_s = tw.sort_values("TVT")
    twt = tw_s["TVT"].values.astype(float)
    twg = tw_s["GR"].ffill().bfill().values.astype(float)
    Z = hz["Z"].values.astype(float); MD = hz["MD"].values.astype(float)
    gr = pd.Series(hz["GR"].values).interpolate(limit_direction="both")\
           .fillna(90.0).values
    ti = hz["TVT_input"].values.astype(float)
    known = np.isfinite(ti)
    kn = np.where(known)[0]; ev = np.where(~known)[0]
    if len(ev) == 0:
        return ti.copy()
    tvt = ti.copy()
    tvt[~known] = ti[kn[-1]]          # placeholder; engine reads tvt only on kn
    comp = []
    for name, params in MODEL["configs"].items():
        runs = [run_pf(twt, twg, tvt, Z, MD, gr, kn, ev, seed=s, **params)
                for s in MODEL["seeds"]]
        comp.append(np.mean(runs, axis=0))
    X = np.column_stack(comp)
    if MODEL["combiner"] == "uniform":
        p = X.mean(1)
    elif MODEL["combiner"] == "median":
        p = np.median(X, 1)
    else:
        q = MODEL["trimmed_q"]
        lo, hi = np.quantile(X, [q, 1 - q], axis=1)
        p = np.clip(X, lo[:, None], hi[:, None]).mean(1)
    pp = MODEL.get("postproc", {})
    if pp.get("hold_w", 0) > 0:
        p = (1 - pp["hold_w"]) * p + pp["hold_w"] * ti[kn[-1]]
    if pp.get("anchor_tau", 0) > 0:
        t = np.arange(len(p), dtype=float)
        p = p + (ti[kn[-1]] - p[0]) * np.exp(-t / pp["anchor_tau"])
    if pp.get("med_win", 0) > 1:
        p = pd.Series(p).rolling(int(pp["med_win"]), center=True,
                                 min_periods=1).median().values
    out = ti.copy(); out[ev] = p
    return out

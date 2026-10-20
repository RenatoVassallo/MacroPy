"""Exploratory: TVAR with block exogeneity (external block linear and exogenous) vs the joint TVAR.

    python block_tvar.py MODE ORDER [LAM0] [HOR] [N_GIRF] [REPS] [SAVE.npz]

MODE  "full": every variable in one threshold VAR (same model as MacroPy's ThresholdVAR; used to
      validate this code against MacroPy). "block": the external variables follow a linear BVAR that
      does not depend on domestic variables; the domestic block (GDP, inflation, interbank rate,
      confidence) is a threshold regression on its own lags and on current and lagged external
      variables, with regime-specific coefficients and covariance.
ORDER which external variables and in what (Cholesky) order.
LAM0  prior tightness of the contemporaneous external coefficients in the domestic block (block mode).
SAVE  optional path: the GIRF draws are stored there (draws, regime, shock, horizon, variable).

Prior, posterior and GIRFs reuse MacroPy's own pieces: Banbura-Giannone-Reichlin dummies
(lambda 0.1, tau 1, epsilon 1, AR(1) sd scales), exact NIW draws per block/regime, uniform prior on
the delay 1..6 with the threshold fixed at 50, rejection of explosive draws, GIRFs by regime of
origin with common random numbers. External lags in the domestic equations get the Minnesota
cross-variable prior (lambda / l); the contemporaneous ones get LAM0. Run from the project root
with PYTHONPATH=src. Writes nothing to the project.
"""
import sys
import time

import numpy as np
from numpy.linalg import cholesky
from scipy.special import logsumexp
from MacroPy.tvar import _create_dummies, _DummyPrior, _draw_niw
from MacroPy.tvarsv import _ar1_moments, _stability

from tvar_shocks import data as dbase

MODE, ORDER = sys.argv[1], sys.argv[2]
LAM0 = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
HOR = int(sys.argv[4]) if len(sys.argv) > 4 else 24
N_GIRF = int(sys.argv[5]) if len(sys.argv) > 5 else 300
REPS = int(sys.argv[6]) if len(sys.argv) > 6 else 10
SAVE = sys.argv[7] if len(sys.argv) > 7 else None
L, MAXD, LAM, TAU, EPS = 2, 6, 0.1, 1.0, 1.0
N_KEEP = 2000
# state variable (env): STATE column (default confidence), TAR value or "p74" (percentile of the
# estimation sample), END of sample, COVID=0 to switch the pandemic treatment off, EPU_CSV path
import os
STATE = os.environ.get("STATE", "exp")
END = os.environ.get("END", "2025-12-01")
COVID = os.environ.get("COVID", "1") == "1"
EXT = {"chn_first": ["g_ip_usa", "fed", "g_ip_chn", "g_xpi"],
       "xpi_first": ["g_ip_usa", "fed", "g_xpi", "g_ip_chn"],
       "xpi_only": ["g_xpi"], "chn_only": ["g_ip_chn"],
       "chn_xpi": ["g_ip_chn", "g_xpi"], "chn_tot": ["g_ip_chn", "g_tot"],
       "chn_gsci": ["g_ip_chn", "g_spgsci"]}[ORDER]
DOM = ["g_gdp", "g_cpi", "int", STATE]
ext, dom = ([], EXT + DOM) if MODE == "full" else (EXT, DOM)
names = ext + dom
nx, nd, N = len(ext), len(dom), len(ext) + len(dom)
SCALES = {"g_ip_chn": 1.0, "g_xpi": 10.0, "g_ip_usa": 1.0, "g_tot": 10.0, "g_spgsci": 10.0}
SHOCKS = [(v, SCALES[v]) for v in EXT if v in SCALES]
ig, ie = names.index("g_gdp"), names.index(STATE)

# ---- sample: as MacroPy with covid_mode="dummies", drop_lags=True --------------------------------
full = dbase.load()
if STATE.startswith("epu"):                           # EPU (Banco de Espana), 3-month average, negated
    import pandas as pd
    col = {"epu_local_neg": "EPU_PER_local", "epu_allpress_neg": "EPU_PER_allpress"}[STATE]
    epu = pd.read_csv(os.environ["EPU_CSV"], parse_dates=["datem"]).set_index("datem")[col]
    full[STATE] = -epu.reindex(full.index).rolling(3).mean()
if "g_spgsci" in names:                               # S&P GSCI (Yahoo Finance), yoy growth
    import pandas as pd
    full["g_spgsci"] = pd.read_csv(os.environ["GSCI_CSV"], parse_dates=["date"]).set_index("date")["g_spgsci"] \
        .reindex(full.index)
db = full.loc[os.environ.get("START", "2003-01-01" if STATE.startswith("epu") else "2002-04-01"):END, names].dropna()
Y, dates = db.to_numpy(float), db.index
start = max(L, MAXD)
in_win = np.asarray((dates >= "2020-01-01") & (dates <= "2021-06-01")) & COVID
keep_full = ~in_win
for j in range(1, L + 1):
    keep_full[j:] &= ~in_win[:-j]
rows = np.arange(start, len(dates))[keep_full[start:]]
T = rows.size
tar_env = os.environ.get("TAR", "50")
TAR = float(tar_env) if not tar_env.startswith("p") else float(np.percentile(Y[rows, ie], 100 - float(tar_env[1:])))
print(f"state={STATE} threshold={TAR:.2f} months in regime 1 (d=1): {int((Y[rows - 1, ie] <= TAR).sum())} of {T}")


def stack(cols, js):
    return np.hstack([Y[rows - j][:, cols] for j in js]) if cols else np.empty((T, 0))


xi, di = list(range(nx)), list(range(nx, N))
yx = Y[rows][:, xi]
Xx = np.hstack([stack(xi, range(1, L + 1)), np.ones((T, 1))])
yd = Y[rows][:, di]
Xd = np.hstack([stack(di, range(1, L + 1)), np.ones((T, 1)), stack(xi, range(0, L + 1))])


def prior_for(y, exo_scales=None):
    mu, sig, delta = _ar1_moments(y, False)
    ydum, xdum = _create_dummies(LAM, TAU, delta, EPS, L, mu, sig, y.shape[1])
    if exo_scales is None:
        return _DummyPrior(ydum, xdum)
    m = exo_scales.size * (L + 1)
    k0 = xdum.shape[1]
    xdum = np.hstack([xdum, np.zeros((xdum.shape[0], m))])
    extra = np.zeros((m, k0 + m))
    for l in range(L + 1):
        lam = LAM0 if l == 0 else LAM
        for k, s in enumerate(exo_scales):
            extra[l * exo_scales.size + k, k0 + l * exo_scales.size + k] = s * max(l, 1) / lam
    return _DummyPrior(np.vstack([ydum, np.zeros((m, y.shape[1]))]), np.vstack([xdum, extra]))


prior_x = prior_for(yx) if nx else None
prior_d = prior_for(yd, _ar1_moments(yx, False)[1] if nx else None)


def ml(prior, y, X):
    return float(prior.log_ml(X.T @ X, X.T @ y, y.T @ y, y.shape[0]))


F = np.stack([Y[rows - d, ie] for d in range(1, MAXD + 1)])
mld = np.array([ml(prior_d, yd[F[d] <= TAR], Xd[F[d] <= TAR]) + ml(prior_d, yd[F[d] > TAR], Xd[F[d] > TAR])
                for d in range(MAXD)])
p_delay = np.exp(mld - logsumexp(mld))
log_bf = logsumexp(mld) - np.log(MAXD) - ml(prior_d, yd, Xd)

# ---- exact posterior draws, explosive ones rejected -------------------------------------------------
t0 = time.time()
rng = np.random.default_rng(2026)
draws, proposed = [], 0
while len(draws) < N_KEEP:
    proposed += 1
    d = int(rng.choice(MAXD, p=p_delay)) + 1
    low = F[d - 1] <= TAR
    Bx = Sx = None
    if nx:
        Bx, Sx = _draw_niw(yx, Xx, prior_x, rng)
        if _stability(Bx.ravel(order="F"), nx, L, 1):
            continue
    B1, S1 = _draw_niw(yd[low], Xd[low], prior_d, rng)
    B2, S2 = _draw_niw(yd[~low], Xd[~low], prior_d, rng)
    if _stability(B1[:nd * L + 1].ravel(order="F"), nd, L, 1) or \
            _stability(B2[:nd * L + 1].ravel(order="F"), nd, L, 1):
        continue
    draws.append((d, low, Bx, Sx, B1, S1, B2, S2))


# ---- generalized impulse responses -------------------------------------------------------------------
def girf(draw, seed):
    d, low_rows, Bx, Sx, B1, S1, B2, S2 = draw
    Cx = cholesky(Sx) if nx else None
    Cd = (cholesky(S1), cholesky(S2))
    pre = max(L, d)
    S = len(SHOCKS) + 1                                   # scenario 0: baseline
    out = []
    for r, mask in enumerate((low_rows, ~low_rows)):
        rng = np.random.default_rng([seed, r])
        idx = rows[mask]
        idx = idx[[keep_full[t - pre + 1:t + 1].all() for t in idx]]
        hist = np.repeat(np.stack([Y[t - pre + 1:t + 1] for t in idx]), REPS, axis=0)
        P = hist.shape[0]
        buf = np.broadcast_to(hist, (S, P, pre, N)).copy()
        means = np.empty((S, HOR, N))
        for h in range(HOR):
            tau = pre + h
            lag = [buf[:, :, (tau - j) % pre, :] for j in range(1, L + 1)]
            ex = np.zeros((S, P, nx))
            ed = [np.zeros((S, P, nd)), np.zeros((S, P, nd))]
            if h == 0:
                for s, (v, scale) in enumerate(SHOCKS, start=1):
                    k = names.index(v)
                    if k < nx:
                        ex[s] = Cx[:, k] * scale / Cx[k, k]
                    else:
                        for q in (0, 1):
                            kk = k - nx
                            ed[q][s] = Cd[q][:, kk] * scale / Cd[q][kk, kk]
            else:
                if nx:
                    ex = np.broadcast_to(rng.standard_normal((P, nx)) @ Cx.T, (S, P, nx))
                z = rng.standard_normal((P, nd))
                ed = [np.broadcast_to(z @ C.T, (S, P, nd)) for C in Cd]
            parts = [np.concatenate([lg[..., nx:] for lg in lag], axis=2), np.ones((S, P, 1))]
            if nx:
                xnew = np.concatenate([lg[..., :nx] for lg in lag], axis=2) @ Bx[:nx * L] + Bx[nx * L] + ex
                parts += [xnew] + [lg[..., :nx] for lg in lag]
            Z = np.concatenate(parts, axis=2)
            lowp = buf[:, :, (tau - d) % pre, ie] <= TAR
            dnew = np.where(lowp[..., None], Z @ B1 + ed[0], Z @ B2 + ed[1])
            new = np.concatenate([xnew, dnew], axis=2) if nx else dnew
            buf[:, :, tau % pre, :] = new
            means[:, h, :] = new.mean(axis=1)
        out.append(means[1:] - means[0])                  # (shock, hor, var)
    return out


sel = np.linspace(0, N_KEEP - 1, N_GIRF).astype(int)
resp = [girf(draws[i], 2026 + j) for j, i in enumerate(sel)]
print(f"mode={MODE} order={ORDER} lam0={LAM0} T={T} p(d)={np.round(p_delay, 2).tolist()} "
      f"stable_share={N_KEEP / proposed:.2f} logBF(domestic/system)={log_bf:.2f} ({time.time() - t0:.0f}s)")
for s, (v, scale) in enumerate(SHOCKS):
    k = names.index(v)
    for H in (12, 24):
        if H > HOR:
            continue
        pt = [np.array([r[g][s][:H, ig].sum() / r[g][s][:H, k].sum() * scale for r in resp]) for g in (0, 1)]
        diff = pt[1] - pt[0]
        q = lambda a: f"{np.median(a):.2f} [{np.percentile(a, 16):.2f}, {np.percentile(a, 84):.2f}]"
        print(f"  {v:9s} H={H:2d}  pes {q(pt[0])}  opt {q(pt[1])}  Pr(opt>pes)={np.mean(diff > 0):.2f}")
if SAVE:
    arr = np.stack([np.stack(r) for r in resp])             # (draws, regime, shock, hor, var)
    np.savez_compressed(SAVE, resp=arr, names=np.array(names), shocks=np.array([v for v, _ in SHOCKS]),
                        scales=np.array([s for _, s in SHOCKS]))
    print(f"saved {SAVE} {arr.shape}")

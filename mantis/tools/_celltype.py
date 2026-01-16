import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

def _bh_fdr_rowwise(p_mat: np.ndarray):
    p = np.asarray(p_mat, dtype=float)
    K, M = p.shape
    q = np.empty_like(p)
    for i in range(K):
        pi = p[i].copy()
        order = np.argsort(pi, kind="mergesort")
        ranks = np.arange(1, M + 1, dtype=float)
        qi = pi[order] * M / ranks
        qi = np.minimum.accumulate(qi[::-1])[::-1]
        out = np.empty_like(qi)
        out[order] = qi
        q[i] = np.clip(out, 0, 1)
    return q

def _significant_by_ct_single_null(
    SCI_real: pd.DataFrame,
    SCI_null: pd.DataFrame,
    alpha: float = 0.05,
    two_sided: bool = True,
):

    rows = SCI_real.index.intersection(SCI_null.index)
    cols = SCI_real.columns.intersection(SCI_null.columns)
    R = SCI_real.loc[rows, cols].astype(float)
    N = SCI_null.loc[rows, cols].astype(float)

    K, M = R.shape
    real_vals = R.to_numpy()   # K×M
    null_vals = N.to_numpy()   # K×M

    if two_sided:
        real_use = np.abs(real_vals)
        null_use = np.abs(null_vals)
    else:
        real_use = real_vals
        null_use = null_vals

    p_emp = np.empty_like(real_vals, dtype=float)
    for i in range(K):
        base = null_use[i]
        for j in range(M):
            p_emp[i, j] = (np.sum(base >= real_use[i, j]) + 1.0) / (M + 1.0)

    q_bh = _bh_fdr_rowwise(p_emp)


    long = []
    for i, ct in enumerate(rows):
        for j, met in enumerate(cols):
            long.append({
                "cell_type":  ct,
                "metabolite": met,
                "sci":        float(real_vals[i, j]),
                "p_emp":      float(p_emp[i, j]),
                "q_bh":       float(q_bh[i, j]),
            })
    long_df = pd.DataFrame(long)
    long_df["significant"] = long_df["q_bh"] < alpha

    per_ct = {}
    sig_rows = []
    for ct, sub in long_df.groupby("cell_type"):
        sig = sub[sub["significant"]].sort_values(["q_bh", "p_emp", "sci"])
        per_ct[ct] = sig
        if not sig.empty:
            sig_rows.append(sig)
    return long_df, per_ct

def _zscore_cols(A):
    mu = A.mean(axis=0, keepdims=True)
    sd = A.std(axis=0, ddof=0, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (A - mu) / sd

def _build_W(xy, l="dmin", norm="row"):
    D = squareform(pdist(xy, metric="euclidean"))
    d_nonzero = D[D > 0]
    if isinstance(l, str) and l == "dmin":
        if d_nonzero.size == 0:
            raise ValueError("there is no dmin.")
        lval = float(np.min(d_nonzero))
    else:
        lval = float(l)
    W = np.exp(-(D**2) / (2.0 * lval * lval))
    np.fill_diagonal(W, 1.0)
    if norm == "row":
        W = W / W.sum(axis=1, keepdims=True)
    elif norm == "global":
        W = W / W.sum()
    return W

def _compute_sci(Ydf, Cdf, xy, normalized=True, l="dmin", w_norm="row"):
    W  = _build_W(xy, l=l, norm=w_norm)
    Cz = _zscore_cols(Cdf.to_numpy())
    Yz = _zscore_cols(Ydf.to_numpy())

    WY      = W @ Yz 
    SCI_raw = Cz.T @ WY 

    if not normalized:
        return pd.DataFrame(SCI_raw, index=Cdf.columns, columns=Ydf.columns)

    xvar = np.sum(Cz**2, axis=0)        # K
    yvar = np.sum(Yz**2, axis=0)        # M
    den  = np.sqrt(xvar[:, None] * yvar[None, :])
    den  = np.where(den == 0, 1.0, den)
    SCI_corr = SCI_raw / den

    return pd.DataFrame(SCI_corr, index=Cdf.columns, columns=Ydf.columns)

def compute_celltype_metabolite(mdata, alpha = 0.05, two_sided = True):
    sci = _compute_sci(mdata.mod['metabolite'].to_df(), mdata.obsm['cell_type'], mdata.obsm['spatial'].values, normalized=True)
    sci_null = _compute_sci(mdata.uns['metabolite_null'].to_df(), mdata.obsm['cell_type'], mdata.obsm['spatial'].values, normalized=True)
    long_df, per_ct = _significant_by_ct_single_null(
        sci, sci_null,
        alpha=alpha,
        two_sided=two_sided
    )
    mdata.uns['celltype_metabolite'] = long_df
    mdata.uns['celltype_metabolite_per_ct'] = per_ct
    return mdata
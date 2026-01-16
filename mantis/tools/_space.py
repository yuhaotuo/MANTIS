import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from pathlib import Path
from scipy.spatial import cKDTree
import re
from matplotlib.backends.backend_pdf import PdfPages
from joblib import Parallel, delayed 
from sklearn.linear_model import Lasso, LassoCV

def fit_lasso_target(
    j, P, y,
    mode,
    alpha_value,
    alphas_grid,
    cv,
    max_iter=5000
):
    """
    mode = "fixed": Lasso(alpha=alpha_value)
    mode = "cv":    LassoCV(alphas=alphas_grid, cv=cv)
    """
    try:
        if mode == "cv":
            model = LassoCV(
                alphas=alphas_grid,
                cv=(cv or 5),
                fit_intercept=False,
                max_iter=max_iter,
                n_jobs=1,
                random_state=42
            )
            model.fit(P, y)
            coef = model.coef_.copy()
            alpha_out = float(getattr(model, "alpha_", np.nan))
        elif mode == "fixed":
            if alpha_value is None:
                raise ValueError("MODE='fixed' requires a numeric ALPHA_VALUE.")
            model = Lasso(alpha=alpha_value, fit_intercept=False, max_iter=max_iter)
            model.fit(P, y)
            coef = model.coef_.copy()
            alpha_out = float(alpha_value)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        resid = y - P @ coef
        return (j, coef, alpha_out, resid)

    except Exception as e:
        print(f"Warning: fit failed for metabolite index {j}: {e}")
        coef = np.zeros(P.shape[1], dtype=float)
        alpha_out = float("nan")
        resid = y.copy()

def _zscore_cols(A):
    mu = A.mean(axis=0, keepdims=True)
    sd = A.std(axis=0, ddof=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (A - mu) / sd

def _morans_I_vec(Ymat, Wmat):
    nloc = Ymat.shape[0]
    S0 = float(Wmat.sum())
    num = (Ymat * (Wmat @ Ymat)).sum(axis=0)
    den = (Ymat**2).sum(axis=0)
    return (nloc / S0) * (num / den)

def moransI(X_df, XY):
    Z_orig = _zscore_cols(X_df.values)
    metabs = X_df.columns.tolist()
    D = squareform(pdist(XY, metric="euclidean"))
    pos = D > 0
    dmin = float(D[pos].min()) if np.any(pos) else np.finfo(float).eps
    l =  np.sqrt(10) * dmin
    Dmat = squareform(pdist(XY, metric='euclidean'))
    W = np.exp(- Dmat**2 / (2 * l **2))
    np.fill_diagonal(W, 0.0)
    row_sums = W.sum(axis=1, keepdims=True); row_sums[row_sums==0]=1.0
    W = W / row_sums
    I = _morans_I_vec(Z_orig, W)
    return pd.DataFrame({
            "metabolite": metabs,
            "Morans_I":    I
            }).sort_values(by="Morans_I", ascending=False).reset_index(drop=True), W

def spatvar_metabolite(mdata, MODE="fixed", ALPHA_VALUE=0.1, ALPHAS_GRID=None, cv_folds=5, n_jobs=-1, threshold = 0.2):
    spots = list(mdata.obs_names.values)
    XY = mdata.obsm['spatial'].values
    X_df = mdata.mod['metabolite'].to_df()
    mI_df, W = moransI(X_df, XY)
    
    # Region
    reg_series = mdata.obsm['region'].astype(object).where(
            mdata.obsm['region'].notna(), 'UNK'
        )
    D_df = pd.get_dummies(reg_series, drop_first=False)
    D = D_df.to_numpy(dtype=float)
    X = _zscore_cols(X_df.values)
    B_D, residuals, rank, s = np.linalg.lstsq(D, X, rcond=None)
    X_hat = D @ B_D 
    R_region = X - X_hat
    proj_check = D.T @ R_region   # R x M
    Z_resid = _zscore_cols(R_region)
    I_after_region = _morans_I_vec(Z_resid, W)
    mI_df_region = pd.DataFrame({ "metabolite": X_df.columns.tolist(), "I_after_region":I_after_region }).sort_values(by="I_after_region", ascending=False).reset_index(drop=True)

    # Cell type
    P_df = mdata.obsm['cell_type'].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    zero_var = P_df.std(axis=0, ddof=0) == 0
    if zero_var.any():
        zero_cols = P_df.columns[zero_var].tolist()
        P_df = P_df.loc[:, ~zero_var]
        print("Dropped zero-variance columns:", zero_cols)
    ct_names = P_df.columns.tolist()
    K = len(ct_names)
    n, M = X_df.shape
    P = P_df.to_numpy(dtype=float)
    P = P - P.mean(axis=0, keepdims=True)
    sd = P.std(axis=0, ddof=0); sd[sd==0]=1.0
    P = P / sd
    Y0 = _zscore_cols(X_df.values)
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(fit_lasso_target)(
                j, P, Y0[:, j],
                mode=MODE,
                alpha_value=ALPHA_VALUE,
                alphas_grid=ALPHAS_GRID,
                cv=cv_folds
            )
            for j in range(M)
        )
    B = np.zeros((K, M), dtype=float)
    alphas = np.zeros(M, dtype=float)
    R = np.zeros((n, M), dtype=float)

    for item in results:
        j, coef, alpha, resid = item
        B[:, j] = coef
        alphas[j] = alpha if (alpha is not None) else np.nan
        R[:, j] = resid
    thr = 1e-8
    nonzero_per_met = (np.abs(B) > thr).sum(axis=0)
    nonzero_per_CT  = (np.abs(B) > thr).sum(axis=1)
    ct_summary = pd.DataFrame({
        "celltype": ct_names,
        "nonzero_count": nonzero_per_CT,
        "nonzero_frac": nonzero_per_CT / M,
        "coef_L2": np.sqrt((B**2).sum(axis=1))
    }).sort_values("coef_L2", ascending=False).reset_index(drop=True)
    coef_df = pd.DataFrame(B, index=ct_names, columns=X_df.columns.tolist())

    Zr = _zscore_cols(R)
    mI_df_cell_type = _morans_I_vec(Zr, W)
    mI_df_cell_type = pd.DataFrame({"metabolite": X_df.columns.tolist(), "I_before": _morans_I_vec(_zscore_cols(Y0), W), "I_after_celltype": mI_df_cell_type})
    for df in (mI_df_cell_type, mI_df_region):
        df["metabolite"] = df["metabolite"].astype(str)
    mI_df_cell_type = mI_df_cell_type.drop_duplicates(subset=["metabolite"])
    mI_df_region = mI_df_region.drop_duplicates(subset=["metabolite"])
    merged = pd.merge(mI_df_cell_type, mI_df_region, on="metabolite", how="inner")
    order = pd.Index(map(str, X_df.columns.tolist()))
    merged["__ord__"] = pd.Categorical(merged["metabolite"], categories=order, ordered=True)
    merged = merged.sort_values("__ord__").drop(columns="__ord__").reset_index(drop=True)
    flag_cols = ["I_before", "I_after_region", "I_after_celltype"]
    for c in flag_cols:
        merged[f"{c}_gt_{str(threshold).replace('.','p')}"] = (merged[c] > threshold).astype(int)
    gt_names = [f"{c}_gt_{str(threshold).replace('.','p')}" for c in flag_cols]
    merged["all_three_gt_threshold"] = merged[gt_names].min(axis=1).astype(int)

    mdata.uns['spatvar_metabolite_celltype_coef'] = coef_df
    mdata.uns['spatvar_metabolite_celltype_summary'] = ct_summary
    mdata.uns['spatvar_metabolite'] = mI_df
    mdata.uns['spatvar_metabolite_region'] = mI_df_region
    mdata.uns['spatvar_metabolite_celltype'] = mI_df_cell_type
    mdata.uns['spatvar_metabolite_combined'] = merged
    return mdata
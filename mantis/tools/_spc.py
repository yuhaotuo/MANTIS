import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from matplotlib.backends.backend_pdf import PdfPages
from joblib import Parallel, delayed 
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from bisect import bisect_left

def _zscore_cols(A):
    mu = A.mean(axis=0, keepdims=True)
    sd = A.std(axis=0, ddof=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (A - mu) / sd

def _morans_I_vec(Y, W):
    n = Y.shape[0]; S0 = float(W.sum())
    num = (Y * (W @ Y)).sum(axis=0)
    den = (Y**2).sum(axis=0)
    return (n / S0) * (num / den)

def slugify(s: str):
    return re.sub(r'[\\/:*?"<>|\s]+', "_", str(s))

def scatter_map(ax, XY, values, title="", vmin=None, vmax=None, cmap="viridis"):
    sc = ax.scatter(XY[:,0], XY[:,1], c=values, cmap=cmap, s=15, edgecolors="none",
                    vmin=vmin, vmax=vmax)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal"); ax.grid(False)
    return sc

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

def parse_bin_index(df):
    bins = {}
    for _, r in df.iterrows():
        b = int(r["bin_index"])
        toks = [t.strip() for t in str(r["metabolites"]).split(";") if t.strip()]
        bins[b] = toks
    return bins

def to_float_mz(x: str) -> float:
    s = str(x).strip()
    if s.startswith("mz-"):
        s = s[3:]
    try:
        return float(s)
    except ValueError:
        return np.nan

def find_closest_matches(targets, refs, refs_sorted, tol=1e-4):
    """Map each target m/z (float) to nearest value in refs within tol"""
    mapping = {}
    for t in targets:
        if np.isnan(t):
            continue
        i = bisect_left(refs_sorted, t)
        candidates = []
        if i < len(refs_sorted): candidates.append(refs_sorted[i])
        if i > 0: candidates.append(refs_sorted[i-1])
        best = None
        best_d = np.inf
        for c in candidates:
            d = abs(c - t)
            if d < best_d:
                best_d = d
                best = c
        if best_d <= tol:
            mapping[t] = best
    return mapping

def lasso_null_for_metabolite(mname, msample, Pz, spots, alpha = 0.06):
    Ysh0 = _zscore_cols(msample)
    K = Pz.shape[1]
    n = len(spots)

    R = Ysh0.shape[1]
    RY   = np.empty_like(Ysh0, dtype=np.float32) 
    Bnul = np.zeros((K, R), dtype=np.float32)
    nnz_list = []

    mdl = Lasso(alpha=float(alpha), fit_intercept=False, max_iter=20000, tol=1e-6, random_state=0)
    for r in range(R):
        y = Ysh0[:, r]

        if np.max(np.abs((Pz * y[:, None]).mean(axis=0))) < 1e-3:
            RY[:, r] = y
            nnz_list.append(0)
            continue
        mdl.fit(Pz, y)
        coef = mdl.coef_.astype(np.float32)
        Bnul[:, r] = coef
        nnz = int((np.abs(coef) > 1e-8).sum())
        nnz_list.append(nnz)
        RY[:, r] = (y - mdl.predict(Pz)).astype(np.float32)


    Ys_null = _zscore_cols(RY)  # n × R
    return ({
        "metabolite": mname,
        "R_shuffles": int(R),
        "nnz_min": int(np.min(nnz_list)) if nnz_list else 0,
        "nnz_median": float(np.median(nnz_list)) if nnz_list else 0.0,
        "nnz_mean": float(np.mean(nnz_list)) if nnz_list else 0.0,
        "nnz_max": int(np.max(nnz_list)) if nnz_list else 0
    }, {"metabolite": mname, "residual": pd.DataFrame(RY, index=spots, columns=[f"shuf_{i}" for i in range(R)]), 
        "residual_zscore": pd.DataFrame(Ys_null, index=spots, columns=[f"shuf_{i}" for i in range(R)]), 
        "coefficients": pd.DataFrame(Bnul, index=[f"CT_{i}" for i in range(K)], columns=[f"shuf_{i}" for i in range(R)])})

def to_float_mz(x: str) -> float:
    s = str(x).strip()
    _float_pat = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')
    if s.startswith("mz-"):
        s = s[3:]
    m = _float_pat.search(s)
    if not m:
        return np.nan
    try:
        return float(m.group(0))
    except Exception:
        return np.nan

def _catalog_files_with_vals(dir_path: Path, suffix: str):
    files = list(dir_path.glob(f"*{suffix}"))
    vals, fpaths = [], []
    for fp in files:
        v = to_float_mz(fp.stem)
        if np.isfinite(v):
            vals.append(v); fpaths.append(fp)
    if not vals:
        return np.array([]), []
    order = np.argsort(vals)
    return np.asarray(vals, dtype=float)[order], [fpaths[i] for i in order]

def _resolve_by_tol(x: float, vals_sorted: np.ndarray, files_sorted: list, tol: float):
    if not np.isfinite(x) or vals_sorted.size == 0:
        return None
    i = bisect_left(vals_sorted, x)
    cand = []
    if i < vals_sorted.size: cand.append(i)
    if i > 0:                cand.append(i-1)
    if not cand:             return None
    best_i = min(cand, key=lambda j: abs(vals_sorted[j] - x))
    return files_sorted[best_i] if abs(vals_sorted[best_i] - x) <= tol else None

def load_null_residual_z_by_tol(mname: str):
    fp = _resolve_by_tol(to_float_mz(mname), vals_resid, files_resid, TOL)
    if fp is None:
        return None
    df = pd.read_csv(fp, index_col=0)
    df.index = df.index.astype(str)
    A = df.loc[pd.Index(spots).astype(str)].to_numpy(np.float32) 
    return _zscore_cols(A)

def sci_null_for_met_by_tol(Ysz, Xs, W, n, genes):
    SCI_n = (Xs.T @ (W @ Ysz)) / float(n)
    return SCI_n

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    x = np.asarray(pvals, dtype=float).ravel()
    n = x.size
    order = np.argsort(x)
    ranks = np.arange(1, n+1, dtype=float)
    q = np.empty_like(x)
    q[order] = (x[order] * n) / ranks
    q[order[::-1]] = np.minimum.accumulate(q[order[::-1]])
    return np.clip(q, 0, 1).reshape(pvals.shape)

def empirical_p_from_pooled_null_two_sided(real_vals: np.ndarray, null_vals: np.ndarray, smooth=True):

    x = np.asarray(real_vals, dtype=float).ravel()
    z = np.asarray(null_vals, dtype=float).ravel()
    z = z[np.isfinite(z)]
    x = x[np.isfinite(x)]
    if x.size == 0 or z.size == 0:
        return np.array([], dtype=float)

    z_sorted = np.sort(z)
    N0 = z_sorted.size

    left_count  = np.searchsorted(z_sorted, x, side="right")
    right_count = N0 - np.searchsorted(z_sorted, x, side="left")

    if smooth:
        p_left  = (left_count  + 1.0) / (N0 + 1.0)
        p_right = (right_count + 1.0) / (N0 + 1.0)
    else:
        p_left  = left_count  / N0
        p_right = right_count / N0

    p = 2.0 * np.minimum(p_left, p_right)
    return np.minimum(p, 1.0)

def spc_ct(mdata, mode = "fixed", alpha = 0.1, alphas_grid = np.logspace(-4, 1, 50), cv_folds = 5, n_jobs = -1, metric_col = "S", alpha_fdr = 0.05, tol = 1e-4):
    # TO CHECK:
    ##  tol was made to be same everywhere
    ## Same for other parameters
    met_df = mdata.mod['metabolite'].to_df()
    spots = met_df.index.tolist()
    metabs = met_df.columns.tolist()
    n, M = met_df.shape
    ct_names = mdata.obsm['cell_type'].columns.tolist()
    ct_df = mdata.obsm['cell_type'].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    zero_var = ct_df.std(axis=0, ddof=0) == 0
    if zero_var.any():
        zero_cols = ct_df.columns[zero_var].tolist()
        ct_df = ct_df.loc[:, ~zero_var]
    ct_names = ct_df.columns.tolist()
    K = len(ct_names)
    P = ct_df.to_numpy(dtype=float)
    P = P - P.mean(axis=0, keepdims=True)
    sd = P.std(axis=0, ddof=0); sd[sd==0]=1.0
    P = P / sd
    Y0 = _zscore_cols(met_df.values)
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(fit_lasso_target)(
            j, P, Y0[:, j],
            mode=mode,
            alpha_value=alpha,
            alphas_grid=alphas_grid,
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
    coef_df = pd.DataFrame(B, index=ct_names, columns=metabs)

    rna_df = mdata.mod['gene'].to_df()
    genes = rna_df.columns.tolist()
    X0 = _zscore_cols(rna_df.values)   # n x G
    G = X0.shape[1]
    XY = mdata.obsm['spatial'].values
    dmin = np.sqrt(2.0)
    D = squareform(pdist(XY, metric="euclidean"))
    W = np.exp(- D**2 / (2 * 5 * dmin**2))
    np.fill_diagonal(W, 0.0)
    row_sums = W.sum(axis=1, keepdims=True); row_sums[row_sums==0]=1.0
    W = W / row_sums

    results_X = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(fit_lasso_target)(
            i, P, X0[:, i],
            alpha_value=alpha,
            alphas_grid=alphas_grid,
            mode=mode,
            cv=cv_folds
        )
        for i in range(G)
    )
    B_X = np.zeros((K, G), dtype=float)
    alphas_X = np.zeros(G, dtype=float)
    R_X = np.zeros((n, G), dtype=float)
    for item in results_X:
        i, coef, alpha, resid = item
        B_X[:, i] = coef
        alphas_X[i] = alpha if (alpha is not None) else np.nan
        R_X[:, i] = resid
    nnz_per_gene = (np.abs(B_X) > 1e-8).sum(axis=0)
    nnz_per_CT_X = (np.abs(B_X) > 1e-8).sum(axis=1)
    gene_resid_df = pd.DataFrame(R_X, index=spots, columns=genes)
    Xs = _zscore_cols(R_X)
    gene_resid_z_df = pd.DataFrame(Xs, index=spots, columns=genes)
    coef_X_df = pd.DataFrame(B_X, index=ct_names, columns=genes)
    ct_summary_X = pd.DataFrame({
        "celltype": ct_names,
        "nonzero_count": nnz_per_CT_X,
        "nonzero_frac": nnz_per_CT_X / G,
        "coef_L2": np.sqrt((B_X**2).sum(axis=1))
    }).sort_values("coef_L2", ascending=False).reset_index(drop=True)

    Xs = _zscore_cols(R_X)
    Ys = _zscore_cols(R)
    S0 = float(W.sum()) if W.size else 1.0
    SCI_after_lasso = (Xs.T @ (W @ Ys)) / max(S0, 1.0)
    sci_df = pd.DataFrame(SCI_after_lasso, index=genes, columns=metabs)
    sci_long = sci_df.stack(dropna=False).reset_index()
    sci_long.columns = ['gene', 'metabolite', 'SPC_CT']
    sci_long = sci_long[np.isfinite(sci_long['SPC_CT'])]
    sci_long = sci_long.sort_values('SPC_CT', ascending=False)

    bin_map = parse_bin_index(mdata.uns['bin_index'])
    input_wide_csv = None
    df = sci_long.copy()
    df = df.rename(columns=lambda c: c.strip())  # remove trailing spaces

    # Add numeric m/z column
    df["mz_val"] = df["metabolite"].astype(str).map(to_float_mz)
    mz_vals = df["mz_val"].dropna().unique()
    mz_vals_sorted = np.sort(mz_vals)

    summary_rows = []
    results = {}
    for b, mets_in_bin in sorted(bin_map.items()):
        target_vals = np.array([to_float_mz(m) for m in mets_in_bin], dtype=float)
        target_vals = target_vals[~np.isnan(target_vals)]
        mapping = find_closest_matches(target_vals, mz_vals, mz_vals_sorted, tol=tol)

        matched_vals = list(mapping.values())
        sub = df[df["mz_val"].isin(matched_vals)].copy()
        results[str(b)] = sub
        if sub.empty:
            print(f"[SKIP] bin {b:02d}: no metabolites found within ±{tol:g}.")
            continue

        # out_csv = out_dir / f"bin_{b:02d}_pairs.csv"
        # sub.to_csv(out_csv, index=False)

        summary = {"bin_id": b, "pairs": int(len(sub)), "tol": tol}
        if "significant" in sub.columns:
            summary["significant_pairs"] = int(sub["significant"].sum())
            summary["fraction_significant"] = float(sub["significant"].mean())
        elif "q_bh" in sub.columns:
            sig = (sub["q_bh"] < alpha_fdr)
            summary["significant_pairs"] = int(sig.sum())
            summary["fraction_significant"] = float(sig.mean())
        elif metric_col in sub.columns:
            v = pd.to_numeric(sub[metric_col], errors="coerce")
            summary.update({
                f"{metric_col}_mean": float(np.nanmean(v)),
                f"{metric_col}_median": float(np.nanmedian(v)),
                f"{metric_col}_p90": float(np.nanpercentile(v.dropna(), 90)) if v.notna().any() else np.nan
            })
        else:
            summary["n_genes"] = int(sub["gene"].nunique())
            summary["n_metabolites"] = int(sub["metabolite"].nunique())

        summary_rows.append(summary)
    summary_df = pd.DataFrame(summary_rows).sort_values("bin_id").reset_index(drop=True)
    mdata.uns['spc_ct_results'] = results

    Pz = np.asarray(P, dtype=np.float32)
    # print(Pz, len(spots), metabs)
    summaries = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(lasso_null_for_metabolite)(m, np.array(mdata.mod['metabolites_sampled'][:, m].X), Pz, spots) for m in metabs
    )
    summary_df = pd.DataFrame([x[0] for x in summaries]).sort_values("metabolite").reset_index(drop=True)
    valid = summary_df["R_shuffles"] > 0

    bin_vals = []   # list[(mz_val, bin_id)]
    for _, r in mdata.uns['bin_index'].iterrows():
        b = int(r["bin_index"])
        toks = [t.strip() for t in str(r["metabolites"]).split(";") if t.strip()]
        for t in toks:
            v = to_float_mz(t)
            if np.isfinite(v):
                bin_vals.append((v, b))
    bin_vals_mz  = np.array([z[0] for z in bin_vals], dtype=float)
    bin_vals_bin = np.array([z[1] for z in bin_vals], dtype=int)

    m2b, miss_names = {}, []
    for m in metabs:
        x = to_float_mz(m)
        if not np.isfinite(x):
            miss_names.append(m); continue
        i = bisect_left(bin_vals_mz, x)
        cand_idx = []
        if i < len(bin_vals_mz): cand_idx.append(i)
        if i > 0:                cand_idx.append(i-1)
        if not cand_idx:
            miss_names.append(m); continue
        best_i = min(cand_idx, key=lambda j: abs(bin_vals_mz[j] - x))
        if abs(bin_vals_mz[best_i] - x) <= tol:
            m2b[m] = int(bin_vals_bin[best_i])
        else:
            miss_names.append(m)
    SCI_null_dict = {}
    have_null_cnt = 0
    for i in summaries:
        S = sci_null_for_met_by_tol(i[1]['residual_zscore'], Xs, W, n, genes)
        SCI_null_dict[i[1]['metabolite']] = S
        if S is not None and S.shape[1] > 0:
            have_null_cnt += 1
    print(f"[INFO] metabolites total={len(metabs)}, with usable NULL={have_null_cnt}")
    metab2idx = {m: j for j, m in enumerate(metabs)}
    G = len(genes)
    summary_rows = []
    bins_seen = sorted({m2b[m] for m in metabs if m in m2b})
    for b in bins_seen:
        mets = [m for m in metabs
                if (m2b.get(m) == b and (m in SCI_null_dict)
                    and (SCI_null_dict[m] is not None) and (SCI_null_dict[m].size > 0))]
        if not mets: 
            continue
        j_idx = [metab2idx[m] for m in mets]
        real = np.abs(SCI_after_lasso[:, j_idx].ravel())
        real = real[np.isfinite(real)]
        null = np.concatenate([np.abs(SCI_null_dict[m].values.ravel()) for m in mets])
        null = null[np.isfinite(null)]
        if real.size == 0 or null.size == 0: 
            continue
        summary_rows.append({
            "bin_id": b,
            "REAL_n": int(real.size), "REAL_mean": float(np.mean(real)), "REAL_median": float(np.median(real)),
            "NULL_n": int(null.size), "NULL_mean": float(np.mean(null)), "NULL_median": float(np.median(null)),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("bin_id").reset_index(drop=True)

    alpha = 0.05
    bin_dfs = []

    bins_seen = sorted({m2b[m] for m in metabs if m in m2b})
    for b in bins_seen:
        mets = [m for m in metabs
                if (m2b.get(m) == b and (m in SCI_null_dict)
                    and (SCI_null_dict[m] is not None) and (SCI_null_dict[m].size > 0))]
        if not mets:
            print(f"[SKIP] bin {b:02d}: no usable metabolites with NULL for FDR.")
            continue

        NULL_vals = np.concatenate([np.asarray(SCI_null_dict[m], dtype=float).ravel() for m in mets])
        NULL_vals = NULL_vals[np.isfinite(NULL_vals)]
        if NULL_vals.size == 0:
            print(f"[SKIP] bin {b:02d}: pooled NULL empty after cleaning.")
            continue

        parts_vals = []
        parts_genidx = []
        parts_metnames = []
        for m in mets:
            j = metab2idx[m]
            col = np.asarray(SCI_after_lasso[:, j], dtype=float)
            finite_idx = np.nonzero(np.isfinite(col))[0]
            if finite_idx.size == 0:
                continue
            vals = col[finite_idx]
            parts_vals.append(vals)
            parts_genidx.append(finite_idx)
            parts_metnames.append(np.repeat(m, vals.size))

        if not parts_vals:
            print(f"[SKIP] bin {b:02d}: no finite REAL SPC_CT values.")
            continue

        REAL_vals = np.concatenate(parts_vals)
        gene_idxs = np.concatenate(parts_genidx)
        met_names = np.concatenate(parts_metnames)

        # empirical p (pooled-null, two-sided) + BH within-bin
        p_all = empirical_p_from_pooled_null_two_sided(REAL_vals, NULL_vals, smooth=True)
        q_all = bh_fdr(p_all)

        gene_names = np.asarray(genes, dtype=object) if 'genes' in globals() else None
        if gene_names is None or gene_names.size <= np.max(gene_idxs):
            gene_col = gene_idxs.astype(int)
        else:
            gene_col = gene_names[gene_idxs]

        df_bin = pd.DataFrame({
            'bin_id': int(b),
            'gene': gene_col,
            'metabolite': met_names,
            'SPC_CT': REAL_vals,
            'pval': p_all,
            'qval': q_all
        })
        df_bin['is_significant'] = np.where(df_bin['qval'] < alpha, 'TRUE', 'FALSE')

        bin_dfs.append(df_bin)
    long_df = pd.concat(bin_dfs, ignore_index=True)
    long_df = long_df.sort_values(['bin_id', 'metabolite', 'gene']).reset_index(drop=True)
    mdata.uns['spc_ct'] = long_df
    return mdata

def residualize(mat, Rmat):
    n, p = mat.shape
    mat = np.asarray(mat, dtype=np.float32)
    resid = np.zeros_like(mat)
    mdl = LinearRegression(fit_intercept=True)

    for j in range(p):
        y = mat[:, j]
        mdl.fit(Rmat, y)
        y_pred = mdl.predict(Rmat)
        resid[:, j] = y - y_pred
    return resid

def build_residual_maker(region_series: pd.Series) -> np.ndarray:
    r = region_series.astype("category")
    Z = pd.get_dummies(r, drop_first=True)
    Z = np.c_[np.ones((Z.shape[0], 1), dtype=np.float32), Z.to_numpy(np.float32)]  # n × p
    # M = I - Z (Z^T Z)^{-1} Z^T
    ZtZ = Z.T @ Z
    inv = np.linalg.pinv(ZtZ)
    H   = Z @ inv @ Z.T
    M   = np.eye(Z.shape[0], dtype=np.float32) - H.astype(np.float32)
    return M

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    x = np.asarray(pvals, dtype=float).ravel()
    n = x.size
    order = np.argsort(x)
    ranks = np.arange(1, n + 1, dtype=float)
    q = np.empty_like(x)
    q[order] = (x[order] * n) / ranks
    q[order[::-1]] = np.minimum.accumulate(q[order[::-1]])
    return np.clip(q, 0, 1).reshape(pvals.shape)

def two_sided_empirical_p_vectorized(obs: np.ndarray, null_vals: np.ndarray) -> np.ndarray:
    x = np.asarray(obs, dtype=float).ravel()
    v = np.sort(np.asarray(null_vals, dtype=float).ravel())
    v = v[np.isfinite(v)]
    N = v.size
    if N == 0:
        return np.full_like(x, np.nan, dtype=float)
    idx_r = np.searchsorted(v, x, side='left')
    p_r = (N - idx_r + 1.0) / (N + 1.0)
    idx_l = np.searchsorted(v, x, side='right')
    p_l = (idx_l + 1.0) / (N + 1.0)
    p = 2.0 * np.minimum(p_l, p_r)
    return np.minimum(p, 1.0)

def ks_D(x, y):
    x = np.sort(np.asarray(x, dtype=float))
    y = np.sort(np.asarray(y, dtype=float))
    nx, ny = x.size, y.size
    if nx == 0 or ny == 0:
        return np.nan
    i = j = 0
    cdf_x = cdf_y = 0.0
    D = 0.0
    while i < nx and j < ny:
        if x[i] <= y[j]:
            i += 1
            cdf_x = i / nx
        else:
            j += 1
            cdf_y = j / ny
        D = max(D, abs(cdf_x - cdf_y))
    D = max(D, abs(1.0 - j / ny))
    D = max(D, abs(i / nx - 1.0))
    return float(D)

def spc_sd(mdata, alpha_fdr = 0.05, tol = 1e-4, use_abs = False):
    met_df = mdata.mod['metabolite'].to_df()
    rna_df = mdata.mod['gene'].to_df()
    genes = rna_df.columns.tolist()
    spots = met_df.index.tolist()
    metabs = met_df.columns.tolist()
    n, M = met_df.shape
    region = mdata.obsm['region'].astype("category")
    n, G = rna_df.shape
    M = met_df.shape[1]
    region_dummies = pd.get_dummies(region, drop_first=True)  # n × R-1
    Rmat = region_dummies.to_numpy(dtype=np.float32)

    XY = mdata.obsm['spatial'].values
    dmin = np.sqrt(2.0)
    D = squareform(pdist(XY, metric="euclidean"))
    W = np.exp(- D**2 / (2 * 5 * dmin**2))
    np.fill_diagonal(W, 0.0)
    row_sums = W.sum(axis=1, keepdims=True); row_sums[row_sums==0]=1.0
    W = W / row_sums

    rna_resid = residualize(rna_df.to_numpy(), Rmat)
    met_resid = residualize(met_df.to_numpy(), Rmat)
    Xs = _zscore_cols(rna_resid)
    Ys = _zscore_cols(met_resid)
    SCI_sd = (Xs.T @ (W @ Ys)) / float(n)

    msi_null = mdata.uns['metabolite_null'].to_df()
    null_msi = {col: msi_null[[col]].to_numpy(np.float32) for col in msi_null.columns}
    M = build_residual_maker(region)
    X_resid = M @ rna_df.to_numpy(np.float32)
    Xs = _zscore_cols(X_resid)

    n = W.shape[0]
    WX = W @ Xs 
    XW = Xs.T @ W 

    SCI_sd_null = {}
    for m in metabs:
        A = null_msi.get(m, None)
        if A is None or A.size == 0:
            continue
        Rm = M @ A
        Ys_null = _zscore_cols(Rm)
        SCI_m = (Xs.T @ (W @ Ys_null)) / float(n)
        df_out = pd.DataFrame(SCI_m, index=genes, columns=[f"shuf_{i}" for i in range(SCI_m.shape[1])])
        SCI_sd_null[m] = df_out
    counts = pd.DataFrame(
        [(m, arr.shape[1]) for m, arr in SCI_sd_null.items()],
        columns=["metabolite", "R_null"]
    ).sort_values("metabolite")

    bin_map = mdata.uns['bin_index'].copy()
    bin_vals = []   # list[(mz_val, bin_id)]
    for _, r in bin_map.iterrows():
        b = int(r["bin_index"])
        toks = [t.strip() for t in str(r["metabolites"]).split(";") if t.strip()]
        for t in toks:
            v = to_float_mz(t)
            if np.isfinite(v):
                bin_vals.append((v, b))
    bin_vals_mz  = np.array([z[0] for z in bin_vals], dtype=float)
    bin_vals_bin = np.array([z[1] for z in bin_vals], dtype=int)

    m2b, miss_names = {}, []
    for m in metabs:
        x = to_float_mz(m)
        if not np.isfinite(x):
            miss_names.append(m); continue
        i = bisect_left(bin_vals_mz, x)
        cand_idx = []
        if i < len(bin_vals_mz): cand_idx.append(i)
        if i > 0:                cand_idx.append(i-1)
        if not cand_idx:
            miss_names.append(m); continue
        best_i = min(cand_idx, key=lambda j: abs(bin_vals_mz[j] - x))
        if abs(bin_vals_mz[best_i] - x) <= tol:
            m2b[m] = int(bin_vals_bin[best_i])
        else:
            miss_names.append(m)

    G, M = SCI_sd.shape
    metab2idx = {m: j for j, m in enumerate(metabs)}
    bins_seen = sorted({m2b[m] for m in metabs if m in m2b})

    summary_rows = []
    overall_total = 0
    overall_sig = 0

    for b in bins_seen:
        mets = [m for m in metabs if m2b.get(m) == b and (m in SCI_sd_null) and (SCI_sd_null[m] is not None)]
        if not mets:
            print(f"[SKIP] bin {b:02d}: no usable metabolites with NULL.")
            continue

        j_idx = [metab2idx[m] for m in mets]

        REAL_vals = SCI_sd[:, j_idx].ravel()
        NULL_vals = np.concatenate([SCI_sd_null[m].to_numpy().ravel() for m in mets])

        REAL_vals = REAL_vals[np.isfinite(REAL_vals)]
        NULL_vals = NULL_vals[np.isfinite(NULL_vals)]
        if REAL_vals.size == 0 or NULL_vals.size == 0:
            print(f"[SKIP] bin {b:02d}: empty REAL or NULL after cleaning.")
            continue

        if use_abs:
            REAL_vals = np.abs(REAL_vals)
            NULL_vals = np.abs(NULL_vals)

        Dks = ks_D(REAL_vals, NULL_vals)

        p_emp = two_sided_empirical_p_vectorized(REAL_vals, NULL_vals)
        q_bh = bh_fdr(p_emp)
        sig = (q_bh < alpha_fdr)

        bin_pairs = REAL_vals.size
        perbin = pd.DataFrame({
            'SCI_real': REAL_vals,
            'p_emp': p_emp,
            'q_bh': q_bh,
            'significant': sig.astype(int)
        })

        n_sig = int(sig.sum())
        frac = float(n_sig / bin_pairs) if bin_pairs > 0 else np.nan
        summary_rows.append({
            'bin_id': b,
            'total_pairs': int(bin_pairs),
            'significant_pairs': n_sig,
            'fraction_significant': frac,
            'KS_D': Dks,
            'REAL_mean': float(np.mean(REAL_vals)),
            'NULL_mean': float(np.mean(NULL_vals)),
            'REAL_median': float(np.median(REAL_vals)),
            'NULL_median': float(np.median(NULL_vals)),
        })

        overall_total += bin_pairs
        overall_sig += n_sig
    summary_df = pd.DataFrame(summary_rows).sort_values('bin_id').reset_index(drop=True)
    overall_frac = (overall_sig / overall_total) if overall_total > 0 else np.nan
    overall_df = pd.DataFrame([{
        'all_bins_total_pairs': int(overall_total),
        'all_bins_significant_pairs': int(overall_sig),
        'all_bins_fraction_significant': float(overall_frac)
    }])
    merged_list = []
    metab2idx = {m: j for j, m in enumerate(metabs)}
    G = len(genes)

    for b in bins_seen:
        mets = [m for m in metabs if m2b.get(m) == b and (m in SCI_sd_null) and (SCI_sd_null[m] is not None)]
        if not mets:
            continue

        j_idx = [metab2idx[m] for m in mets]
        REAL_mat = SCI_sd[:, j_idx] 
        REAL_vals = REAL_mat.ravel()
        NULL_vals = np.concatenate([SCI_sd_null[m].to_numpy().ravel() for m in mets])


        REAL_vals = REAL_vals[np.isfinite(REAL_vals)]
        NULL_vals = NULL_vals[np.isfinite(NULL_vals)]
        if REAL_vals.size == 0 or NULL_vals.size == 0:
            continue

        if use_abs:
            REAL_vals = np.abs(REAL_vals)

        p_emp = two_sided_empirical_p_vectorized(REAL_vals, NULL_vals)
        q_bh = bh_fdr(p_emp)
        sig = (q_bh < alpha_fdr).astype(int)

        K = len(mets)
        gene_col = np.repeat(genes, K)
        metab_col = np.tile(mets, G)

        keep = np.isfinite(REAL_vals) & np.isfinite(p_emp) & np.isfinite(q_bh)
        if not np.any(keep):
            continue

        df_bin = pd.DataFrame({
            'gene': gene_col[keep],
            'metabolite': metab_col[keep],
            'SPC_SD': REAL_vals[keep],
            'pval': p_emp[keep],
            'qval': q_bh[keep],
            'is_significant': (q_bh[keep] < alpha_fdr)

        })

        merged_list.append(df_bin)
    merged_df = pd.concat(merged_list, ignore_index=True)
    mdata.uns['spc_sd'] = merged_df
    return mdata
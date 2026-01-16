import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
import math
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import networkx as nx
import re

def _build_weighted_radius_graph(coords, radius=None, l=None, weight_key="w"):

    if (radius is None) and (l is None):
        raise ValueError("radius or l should be supplied")
    if l is None:
        l = float(radius)
    if radius is None:
        radius = 2.0 * float(l)

    tree = cKDTree(coords)
    coo = tree.sparse_distance_matrix(tree, max_distance=radius, output_type="coo_matrix")
    G = nx.Graph()
    G.add_nodes_from(range(coords.shape[0]))
    inv_two_l2 = 1.0 / (2.0 * float(l) * float(l))

    for i, j, d in zip(coo.row, coo.col, coo.data):
        if i < j:
            if d <= l:
                w = 1.0
            else:
                w = float(np.exp(-(d * d) * inv_two_l2))
            if w > 0.0:
                G.add_edge(i, j, **{weight_key: w})
    return G

def _target_diag_M_from_matrix(G, X_mat, weight_key="w"):
    M = X_mat.shape[1]
    Mt = np.zeros(M, dtype=float)
    for u, v, data in G.edges(data=True):
        w = data.get(weight_key, 1.0)
        Mt += w * X_mat[u] * X_mat[v]
    return Mt

def _build_W_from_coords(coords_xy, l):
    D_condensed = pdist(coords_xy, metric="euclidean")
    dist = squareform(D_condensed)
    W = np.zeros_like(dist, dtype=float)

    mask = dist > 0
    W[mask] = np.exp(-(dist[mask]**2) / (2.0 * l**2))
    np.fill_diagonal(W, 1.0)

    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    W /= row_sums
    return W

def _zscore_cols(A: np.ndarray) -> np.ndarray:
    mu = A.mean(axis=0, keepdims=True)
    sd = A.std(axis=0, ddof=0, keepdims=True)
    sd[sd == 0.0] = 1.0
    return (A - mu) / sd

def bh_fdr(p_values_1d):

    p = np.asarray(p_values_1d, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty_like(ranked)
    prev = 1.0
    for i in range(n-1, -1, -1):
        rank = i + 1
        q_i = ranked[i] * n / rank
        if q_i > prev:
            q_i = prev
        prev = q_i
        q[i] = q_i
    out = np.empty_like(q)
    out[order] = q
    return out

def pooled_empirical_p_two(real_mat, null_mat):
    real = real_mat.ravel()
    null_vals = null_mat.ravel()
    null_vals = null_vals[np.isfinite(null_vals)]

    null_sorted = np.sort(null_vals)
    N = null_sorted.size

    idx_right = np.searchsorted(null_sorted, real, side='right')
    count_le = idx_right
    count_ge = N - np.searchsorted(null_sorted, real, side='left')

    p_left  = (count_le + 1) / (N + 1)
    p_right = (count_ge + 1) / (N + 1)

    p_two = 2 * np.minimum(p_left, p_right)
    p_two = np.minimum(p_two, 1.0)
    return p_two.reshape(real_mat.shape)

def compute_genemet_sci(mdata, n_bins=10, alpha=0.05):
    dmin = math.sqrt(2.0); l = 4*dmin
    G = _build_weighted_radius_graph(mdata.obsm['spatial'].values, l=l, radius=float("inf"), weight_key="w")
    X_all = np.asarray(mdata.mod['metabolite'].to_df(), dtype=float)
    X_df = mdata.mod['metabolite'].to_df()
    X_rna = mdata.mod['gene'].to_df().to_numpy()
    met_names = X_df.columns.tolist()
    X_mean = X_all.mean(axis=0, keepdims=True)
    X_std  = X_all.std(axis=0, ddof=0, keepdims=True); X_std[X_std==0] = 1.0
    Xz = (X_all - X_mean) / X_std

    M0_varnorm = _target_diag_M_from_matrix(G, Xz, weight_key="w")
    M0 = np.asarray(M0_varnorm, dtype=float)
    q = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(M0, q, method="linear")
    edges = [bin_edges[0]]
    for v in bin_edges[1:]:
        if v > edges[-1]:
            edges.append(v)
    bin_edges = np.array(edges, dtype=float)
    k = len(bin_edges) - 1

    s = pd.Series(M0, index=pd.Index(met_names, name="metabolite"))
    idx = np.searchsorted(bin_edges, s.values, side="right") - 1
    idx = np.clip(idx, 0, k-1)

    bin_mets = [[] for _ in range(k)]
    for name, b in zip(s.index, idx):
        bin_mets[b].append(name)

    name_to_idx = {name: i for i, name in enumerate(met_names)}
    bin_counts = []
    bin_dfs = {}
    for i, mets in enumerate(bin_mets):
        if not mets:
            bin_counts.append(0)
            continue
        cols = [name_to_idx[m] for m in mets if m in name_to_idx]
        bin_counts.append(len(cols))
        if cols:
            df_bin = pd.DataFrame(
                X_all[:, cols],
                columns=[met_names[j] for j in cols],
                index=pd.Index(mdata.obs_names, name="spot_id")
            )
            left, right = float(bin_edges[i]), float(bin_edges[i+1])
            bin_dfs[i+1] = df_bin
    l = dmin
    W = _build_W_from_coords(mdata.obsm['spatial'].values, l=l)
    S0 = float(W.sum())
    if S0 == 0: S0 = 1.0
    genemet_sci_bins = {}
    for i in bin_dfs:
        X_met = bin_dfs[i].to_numpy(dtype=float)
        X_rna_std = (X_rna - X_rna.mean(axis=0, keepdims=True)) / np.maximum(X_rna.std(axis=0, ddof=0, keepdims=True), 1e-12)
        X_met_std = (X_met - X_met.mean(axis=0, keepdims=True)) / np.maximum(X_met.std(axis=0, ddof=0, keepdims=True), 1e-12)
        WY = W.dot(X_met_std)
        WY  = W @ X_met_std
        SCI = (X_rna_std.T @ WY) / S0
        sci_df = pd.DataFrame(SCI, index=mdata.mod['gene'].to_df().columns, columns=bin_dfs[i].columns)
        genemet_sci_bins[i] = sci_df
    
    X_met_null_df = mdata.uns['metabolite_null'].to_df()
    D = squareform(pdist(mdata.obsm['spatial'].values, metric="euclidean"))
    pos = D > 0
    dmin = float(D[pos].min()) if np.any(pos) else np.finfo(float).eps
    l = dmin
    W = _build_W_from_coords(mdata.obsm['spatial'].values, l=l)
    S0  = float(W.sum())
    if S0 == 0: S0 = 1.0
    genemet_sci_bins_null = {}
    for i in bin_dfs:
        cols_present = [c for c in bin_dfs[i].columns if c in X_met_null_df.columns]
        X_met_null = X_met_null_df[cols_present].to_numpy(dtype=float)
        X_rna_std = _zscore_cols(X_rna)
        X_met_null_std = _zscore_cols(X_met_null)

        WY = W.dot(X_met_null_std)
        WY  = W @ X_met_null_std
        SCI = (X_rna_std.T @ WY) / S0
        sci_df = pd.DataFrame(SCI, index=mdata.mod['gene'].to_df().columns, columns=bin_dfs[i].columns)
        genemet_sci_bins_null[i] = sci_df
    results = {}
    for n in genemet_sci_bins:
        common_genes = genemet_sci_bins[n].index.intersection(genemet_sci_bins_null[n].index)
        common_mets  = genemet_sci_bins[n].columns.intersection(genemet_sci_bins_null[n].columns)
        real_df = genemet_sci_bins[n].loc[common_genes, common_mets]
        null_df = genemet_sci_bins_null[n].loc[common_genes, common_mets]
        p_two = pooled_empirical_p_two(real_df.values, null_df.values)
        q_two = bh_fdr(p_two.ravel()).reshape(p_two.shape)

        sig_mask = (q_two < alpha)

        res = []
        genes = real_df.index.tolist()
        mets  = real_df.columns.tolist()
        for i, g in enumerate(genes):
            for j, m in enumerate(mets):
                res.append({
                    "gene": g,
                    "metabolite": m,
                    "SCI": float(real_df.iloc[i, j]),
                    "p_value": float(p_two[i, j]),
                    "q_value": float(q_two[i, j]),
                    "significant": bool(sig_mask[i, j])
                })
        res_df = pd.DataFrame(res)
        print(int(sig_mask.sum()))
        results[n] = res_df
    dfs = []
    for i in results:
        results[i]["bin_index"] = i
        dfs.append(results[i])
    merged_eq = pd.concat(dfs, ignore_index=True, sort=False)
    meta_cols = ["method", "bin_index", "bin_min", "bin_max", "bin_label", "source_file"]
    front = [c for c in meta_cols if c in merged_eq.columns]
    rest = [c for c in merged_eq.columns if c not in front]
    merged_eq = merged_eq[front + rest]
    summary = (
        merged_eq.groupby(["bin_index"], dropna=False)
                .size().reset_index(name="significant_pairs")
    )
    mdata.uns['genemet_sci'] = merged_eq
    mdata.uns['genemet_sci_summary'] = summary
    return mdata
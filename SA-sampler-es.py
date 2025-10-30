#!/usr/bin/env python
# coding: utf-8

# In[72]:


import os, psutil, time, threading

def monitor_memory(interval=1):
    process = psutil.Process(os.getpid())
    while True:
        mem = process.memory_info().rss / 1024**2
        print(f"[MEM] {mem:.1f} MB")
        time.sleep(interval)

threading.Thread(target=monitor_memory, daemon=True).start()


# # Input

# In[73]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[74]:


# rna_csv = "/storage/scratch1/2/yhao306/sma-brain/data-sm/V11L12-038_A1_RNA_raw_counts.csv"
# rna_df = pd.read_csv(rna_csv, index_col=0)
# msi_csv = "/storage/scratch1/2/yhao306/sma-brain/data-sm/V11L12-038_A1_MSI_raw_counts_spotRows.csv"
# msi_df = pd.read_csv(msi_csv, index_col=0)
coord_csv = "/storage/scratch1/2/yhao306/sma-brain/data-sm/V11L12-038_A1_spot_coordinates.csv"
coords_df = pd.read_csv(coord_csv, index_col=0) 

# rna_mat = rna_df.T


# In[ ]:





# In[75]:


# rna_csv = "rna_norm.csv"
# rna_norm = pd.read_csv(rna_csv, index_col=0)
rna_norm_hvg = "rna_norm_hvg.csv"
rna_norm_hvg = pd.read_csv(rna_norm_hvg, index_col=0)


# In[76]:


# msi_csv = "msi_norm.csv"
# msi_norm = pd.read_csv(msi_csv, index_col=0)
msi_norm_hvg = "msi_norm_hvg.csv"
msi_norm_hvg = pd.read_csv(msi_norm_hvg, index_col=0)


# In[77]:


print(msi_norm_hvg)


# In[78]:


# prefixes = ('Rpl', 'Rps', 'Mrp', 'mt-')

# cols_to_drop = [col for col in rna_norm.columns if col.startswith(prefixes)]
# rna_norm.drop(columns=cols_to_drop, inplace=True)


# In[79]:


common_barcodes = rna_norm_hvg.index
msi_norm = msi_norm_hvg.loc[common_barcodes]
coords_df = coords_df.loc[common_barcodes]


# In[80]:


region_csv = "/storage/scratch1/2/yhao306/sma-brain/data-sm/sma/V11L12-038/V11L12-038_A1/output_data/V11L12-038_A1_RNA/outs/RegionLoupe.csv"
region_df = pd.read_csv(region_csv, index_col=0)


# In[81]:


print(region_df)


# In[82]:


region_df.index = region_df.index + "_1"
print(region_df)


# In[83]:


# common_barcodes = rna_norm.index
# region_df = region_df.loc[common_barcodes]
# print(region_df)


# In[84]:


# region_df.index.name = 'barcode'
# coords_df.index.name = 'barcode'
# coords_small = coords_df[['x', 'y']]
# df_merged = region_df.join(coords_small, how='inner')
# print(df_merged.head())


# # MCMC

# In[85]:


import os, math, csv, random
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import re


# In[86]:


print(coords_df)


# In[87]:


def align_inputs(coords_df: pd.DataFrame, msi_norm_hvg: pd.DataFrame):
    common = coords_df.index.intersection(msi_norm_hvg.index)
    if len(common) == 0:
        raise ValueError("No overlapping spots between coords_df and msi_norm_hvg.")
    coords = coords_df.loc[common, ["x","y"]].to_numpy(dtype=float)
    X_mat  = msi_norm_hvg.loc[common].to_numpy(dtype=float)
    spot_ids = common.tolist()
    met_names = msi_norm_hvg.columns.tolist()
    return coords, X_mat, spot_ids, met_names


# In[88]:


def slugify(name: str) -> str:
    s = re.sub(r"[^\w\.\u4e00-\u9fa5-]+", "-", str(name))
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "metabolite"


# In[89]:


# target = "289.92275500000005"
# if target in msi_norm_hvg.columns:
#     chosen = target
# else:
#     target_val = float(target)
#     cols_float = pd.to_numeric(msi_norm_hvg.columns, errors="coerce")
#     idx = int(np.nanargmin(np.abs(cols_float - target_val)))
#     chosen = msi_norm_hvg.columns[idx]
#     print(f"match close：{chosen}")

# msi_norm_hvg_162 = msi_norm_hvg[[chosen]].copy()
# X1 = msi_norm_hvg_162.to_numpy(dtype=float)


# In[ ]:


missing = ['556.2165150000001']

msi_norm_hvg_162 = msi_norm_hvg[missing].copy()

X1 = msi_norm_hvg_162.to_numpy(dtype=float)


# In[91]:


# target = "162.11293999999998"
# if target in msi_norm_hvg.columns:
#     chosen = target
# else:
#     target_val = float(target)
#     cols_float = pd.to_numeric(msi_norm_hvg.columns, errors="coerce")
#     idx = int(np.nanargmin(np.abs(cols_float - target_val)))
#     chosen = msi_norm_hvg.columns[idx]
#     print(f"match close：{chosen}")

# msi_norm_hvg_162 = msi_norm_hvg[[chosen]].copy()
# X1 = msi_norm_hvg_162.to_numpy(dtype=float)


# In[92]:


print(msi_norm_hvg_162)


# In[93]:


def build_weighted_radius_graph(coords, radius=None, l=None, weight_key="w"):

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

def target_diag_M_from_matrix(G, X_mat, weight_key="w"):
    M = X_mat.shape[1]
    Mt = np.zeros(M, dtype=float)
    for u, v, data in G.edges(data=True):
        w = data.get(weight_key, 1.0)
        Mt += w * X_mat[u] * X_mat[v]
    return Mt


# In[94]:


def local_delta_diag_perm_weighted(G, pi, X_mat, u, v, weight_key="w"):
    Xu_old = X_mat[pi[u]]
    Xv_old = X_mat[pi[v]]
    delta = np.zeros(X_mat.shape[1], dtype=float)
    for n, data in G[u].items():
        if n == v:
            continue
        w = data.get(weight_key, 1.0)
        Xn = X_mat[pi[n]]
        delta -= w * Xu_old * Xn
        delta += w * Xv_old * Xn

    for n, data in G[v].items():
        if n == u:
            continue
        w = data.get(weight_key, 1.0)
        Xn = X_mat[pi[n]]
        delta -= w * Xv_old * Xn
        delta += w * Xu_old * Xn

    return delta


# In[95]:


# def local_delta_diag_perm_weighted(G, pi, X_mat, u, v, weight_key="w"):
#     """
#     """
#     Xu = X_mat[pi[u]]
#     Xv = X_mat[pi[v]]
#     dM = np.zeros_like(Xu, dtype=float)

#     for k in G.neighbors(u):
#         if k == v: 
#             continue
#         w = G[u][k].get(weight_key, 1.0)
#         Xk = X_mat[pi[k]]
#         dM += w * (Xv - Xu) * Xk

#     for k in G.neighbors(v):
#         if k == u:
#             continue
#         w = G[v][k].get(weight_key, 1.0)
#         Xk = X_mat[pi[k]]
#         dM += w * (Xu - Xv) * Xk

#     return dM


# In[ ]:


# def precalc_neighbor_sums(G, pi, X_mat, weight_key="w"):
#     n = G.number_of_nodes()
#     p = X_mat.shape[1]
#     S = np.zeros((n, p), dtype=float)
#     for i in range(n):
#         for k in G.neighbors(i):
#             w = G[i][k].get(weight_key, 1.0)
#             S[i] += w * X_mat[pi[k]]
#     return S

# def local_delta_diag_perm_weighted_cached(G, pi, X_mat, u, v, S, weight_key="w"):
#     """
#     用缓存 S 计算 dM，与原 local_delta_diag_perm_weighted 等价。
#     公式（排除互为邻居的那一项）：
#       dM = (Xv - Xu) * [ (S[u] - 1_{u~v} w_uv * Xv) - (S[v] - 1_{v~u} w_vu * Xu) ]
#     """
#     Xu = X_mat[pi[u]]
#     Xv = X_mat[pi[v]]

#     if G.has_edge(u, v):
#         w_uv = G[u][v].get(weight_key, 1.0)
#         w_vu = G[v][u].get(weight_key, 1.0)
#     else:
#         w_uv = 0.0
#         w_vu = 0.0

#     Su_excl_v = S[u] - (w_uv * Xv if w_uv else 0.0)
#     Sv_excl_u = S[v] - (w_vu * Xu if w_vu else 0.0)

#     dM = (Xv - Xu) * (Su_excl_v - Sv_excl_u)
#     return dM


# def apply_swap_and_update_cache(G, pi, X_mat, u, v, S, weight_key="w", swap_pi=True):
#     """
#     当 (u,v) 交换被接受时，更新 S，并可选择是否在这里交换 pi。
#     记 Δ = X_u - X_v，其中 X_u = X_mat[pi[u]]，X_v = X_mat[pi[v]]（交换前的取值）
#     更新规则（仅受影响的节点）：
#       ∀k∈N(u)\{v}: S[k] -= w_{k,u} * Δ
#       ∀k∈N(v)\{u}: S[k] += w_{k,v} * Δ
#       若 u~v: S[u] += w_{u,v} * Δ,  S[v] -= w_{v,u} * Δ
#     最后若 swap_pi=True，则在此函数内部执行 pi[u], pi[v] 交换。
#     """
#     X_u = X_mat[pi[u]].copy()
#     X_v = X_mat[pi[v]].copy()
#     Delta = X_u - X_v  # Δ

#     # 更新与 u 相邻的所有节点（这些 S[k] 包含 w_{k,u} * X_{pi[u]}）
#     for k in G.neighbors(u):
#         if k == v:
#             continue
#         w_ku = G[k][u].get(weight_key, 1.0)
#         S[k] -= w_ku * Delta

#     # 更新与 v 相邻的所有节点（这些 S[k] 包含 w_{k,v} * X_{pi[v]}）
#     for k in G.neighbors(v):
#         if k == u:
#             continue
#         w_kv = G[k][v].get(weight_key, 1.0)
#         S[k] += w_kv * Delta

#     # 自身行修正（各自邻域里“对方”的那一项变化）
#     if G.has_edge(u, v):
#         w_uv = G[u][v].get(weight_key, 1.0)
#         w_vu = G[v][u].get(weight_key, 1.0)
#         S[u] += w_uv * Delta     # 原 w_uv * X_v -> w_uv * X_u
#         S[v] -= w_vu * Delta     # 原 w_vu * X_u -> w_vu * X_v

#     if swap_pi:
#         pi[u], pi[v] = pi[v], pi[u]


# # ========= 你的原加权 dM（保留，作为后备或对照测试用） =========
# def local_delta_diag_perm_weighted(G, pi, X_mat, u, v, weight_key="w"):
#     """
#     原始 O(deg(u)+deg(v)) 版本（未改动）。可用于回退或对照。
#     """
#     Xu = X_mat[pi[u]]
#     Xv = X_mat[pi[v]]
#     dM = np.zeros_like(Xu, dtype=float)

#     for k in G.neighbors(u):
#         if k == v:
#             continue
#         w = G[u][k].get(weight_key, 1.0)
#         Xk = X_mat[pi[k]]
#         dM += w * (Xv - Xu) * Xk

#     for k in G.neighbors(v):
#         if k == u:
#             continue
#         w = G[v][k].get(weight_key, 1.0)
#         Xk = X_mat[pi[k]]
#         dM += w * (Xu - Xv) * Xk

#     return dM

def precalc_neighbor_sums(G, pi, X, weight_key="w"):
    """
    计算 S[i] = ∑_{k∈N(i)} w_{ik} * X[pi[k]]
    X: (N, p)  已中心化的矩阵
    返回形状: (N, p)
    """
    N, p = X.shape
    S = np.zeros((N, p), dtype=float)
    for i in range(N):
        for k in G.neighbors(i):
            w = G[i][k].get(weight_key, 1.0)
            S[i] += w * X[pi[k]]
    return S


# ========= 基于 S 的 dM（加权） =========
def local_delta_diag_perm_weighted_cached(G, pi, X, u, v, S, weight_key="w"):
    Xu = X[pi[u]]
    Xv = X[pi[v]]

    if G.has_edge(u, v):
        w_uv = G[u][v].get(weight_key, 1.0)
        w_vu = G[v][u].get(weight_key, 1.0)
    else:
        w_uv = 0.0
        w_vu = 0.0

    Su_excl_v = S[u] - (w_uv * Xv if w_uv else 0.0)
    Sv_excl_u = S[v] - (w_vu * Xu if w_vu else 0.0)
    return (Xv - Xu) * (Su_excl_v - Sv_excl_u)

def apply_swap_and_update_cache(G, pi, X, u, v, S, weight_key="w", swap_pi=True):
    Xu = X[pi[u]].copy()
    Xv = X[pi[v]].copy()
    Delta = Xu - Xv
    for k in G.neighbors(u):
        if k == v: 
            continue
        w_ku = G[k][u].get(weight_key, 1.0)
        S[k] -= w_ku * Delta
    for k in G.neighbors(v):
        if k == u:
            continue
        w_kv = G[k][v].get(weight_key, 1.0)
        S[k] += w_kv * Delta

    if G.has_edge(u, v):
        w_uv = G[u][v].get(weight_key, 1.0)
        w_vu = G[v][u].get(weight_key, 1.0)
        S[u] += w_uv * Delta
        S[v] -= w_vu * Delta

    if swap_pi:
        pi[u], pi[v] = pi[v], pi[u]




# In[97]:


def snapshot_after_only(coords, X_mat, pi, met_idx, out_path, title_info=""):
    z_after = X_mat[pi, met_idx]
    col_all = X_mat[:, met_idx]
    vmin = float(col_all.min())
    vmax = float(col_all.max())

    fig, ax = plt.subplots(figsize=(6,8))
    sc = ax.scatter(coords[:,0], coords[:,1], c=z_after, s=6, vmin=vmin, vmax=vmax)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.invert_yaxis()
    if title_info:
        ax.set_title(title_info, fontsize=11)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# # Metropolis-Hastings

# #################################### NEW Ver. #######################

# In[98]:


def _energy(M, M0, metric='l1'):
    d = np.abs(M - M0)
    if metric == 'l1':
        return float(np.sum(d))
    else:
        raise ValueError(f"unknown energy metric: {metric}")


# In[99]:


# def mcmc_shuffle_diag_delta_seqshots(
#     G, coords, X_mat,
#     n_accept=5000,
#     t=None,
#     t_scale=1.0,
#     eps=None,
#     seed=None,
#     acc_every=50,
#     pre_snap_every=50,
#     snapshot_metabolites=None,
#     met_names=None,
#     topk_for_snapshot=6,
#     snapshot_root="snapshots_after_only",
#     log_csv=None,
#     save_initial=True,
#     print_M=False,
#     save_M_history=True,
#     monitor_cols=None,
#     print_mean=True,

#     pre_shuffle_swaps=1000,
#     weight_key="w",
#     energy_metric="l1",

#     callback=None,
#     callback_every=None,
#     callback_kwargs=None,

#     t_scale_start=2.0,
#     t_scale_min=0.1,
#     t_step=0.1,
#     t_scale_every=100,
#     t_scale_count_by="accepted",
#     t_step_units="scale", 
# ):
#     import os, csv, math, random
#     import numpy as np

#     try:
#         import pandas as pd
#         if isinstance(X_mat, (pd.DataFrame, pd.Series)):
#             X_mat = X_mat.to_numpy()
#     except Exception:
#         pass

#     rng = random.Random(seed)
#     N, M = X_mat.shape
#     nodes = list(range(N))
#     Xc = X_mat - X_mat.mean(axis=0, keepdims=True)
#     Mt = target_diag_M_from_matrix(G, Xc, weight_key=weight_key)

#     if eps is None:
#         nz = np.abs(Mt[np.abs(Mt) > 0])
#         base = np.median(nz) if nz.size else 1.0
#         eps = 1e-9 * base
    
#     use_weighted = (weight_key is not None)
#     def local_delta(u, v):
#         if use_weighted:
#             return local_delta_diag_perm_weighted(G, pi, Xc, u, v, weight_key=weight_key)
#         else:
#             return local_delta_diag_perm(G, pi, Xc, u, v)

#     def _counter(accepted, proposals, pre_done):
#         if t_scale_count_by == "proposal":
#             return proposals
#         elif t_scale_count_by == "global":
#             return pre_done + accepted
#         else:  # "accepted"
#             return accepted

#     pi = np.arange(N, dtype=int)

#     if snapshot_metabolites is None:
#         idx = np.argsort(-np.abs(Mt))
#         snap_idx = list(idx[:min(topk_for_snapshot, M)])
#     else:
#         if met_names is None:
#             snap_idx = [int(i) for i in snapshot_metabolites]
#         else:
#             name_to_idx = {n:i for i,n in enumerate(met_names)}
#             snap_idx = [name_to_idx[m] if isinstance(m, str) else int(m) for m in snapshot_metabolites]

#     met_dirs = {}
#     for m in snap_idx:
#         mname = met_names[m] if met_names else f"m{m}"
#         mdir = os.path.join(snapshot_root, slugify(mname))
#         os.makedirs(mdir, exist_ok=True)
#         met_dirs[m] = mdir
#         if save_initial:
#             init_title = f"{met_names[m] if met_names else f'm{m}'} | acc=0, seed={seed}"
#             snapshot_after_only(coords, X_mat, pi, m,
#                                 out_path=os.path.join(mdir, "acc000000.png"),
#                                 title_info=init_title)

#     accepted = 0
#     proposals = 0
#     touched = np.zeros(N, dtype=bool)

#     writer = None; f = None
#     if log_csv:
#         log_dir = os.path.dirname(log_csv)
#         if log_dir:
#             os.makedirs(log_dir, exist_ok=True)
#         f = open(log_csv, "w", newline="", encoding="utf-8")
#         writer = csv.writer(f)
#         writer.writerow(["phase","accepted_idx","u","v","coverage","Ecur","dE","acc_prob"])

#     Mcur = Mt.copy()
#     Ecur = _energy(Mcur, Mt, energy_metric)
#     Ms_log = [] if save_M_history else None
#     base_mean = float(np.mean(X_mat))

#     if save_M_history:
#         Ms_log.append(Mcur.copy())

#     if callback_every is None:
#         callback_every = acc_every
#     if callback_kwargs is None:
#         callback_kwargs = {}

#     try:
#         if callback is not None:
#             print(f"[mcmc->cb] init: global_acc=0, accepted=0, lenMs={len(Ms_log) if Ms_log is not None else None}")
#             callback(Ms_log, Mt, 0, 0, pi,
#                      coords=coords, X_mat=X_mat, met_names=met_names,
#                      T=None, t=None, t_scale=float(t_scale_start), pre_done=0, **callback_kwargs)
#     except Exception as e:
#         print(f"[callback warning] init failed: {e}")

#     deltaM_samples = []
#     pre_done = 0
#     if pre_shuffle_swaps and pre_shuffle_swaps > 0:
#         while pre_done < pre_shuffle_swaps:
#             u, v = rng.sample(nodes, 2)
#             if np.allclose(Xc[pi[u]], Xc[pi[v]], atol=1e-12, rtol=0):
#                 continue
#             dlt = local_delta(u, v)
#             deltaM_samples.append(float(np.sum(np.abs(dlt))))
#             pi[u], pi[v] = pi[v], pi[u]
#             Mcur = Mcur + dlt
#             pre_done += 1

#             if save_M_history:
#                 Ms_log.append(Mcur.copy())

#             if pre_snap_every and (pre_done % pre_snap_every == 0):
#                 for m in snap_idx:
#                     mdir = met_dirs[m]
#                     title = f"{met_names[m] if met_names else f'm{m}'} | acc={pre_done}, seed={seed}"
#                     snapshot_after_only(coords, X_mat, pi, m,
#                                         out_path=os.path.join(mdir, f"acc{pre_done:06d}.png"),
#                                         title_info=title)

#             if callback is not None and callback_every and (pre_done % callback_every == 0):
#                 try:
#                     ts_now = float(t_scale_start)
#                     print(f"[mcmc->cb] pre:  global_acc={pre_done}, accepted=0, lenMs={len(Ms_log)}")
#                     callback(Ms_log, Mt, 0, pre_done, pi,
#                              coords=coords, X_mat=X_mat, met_names=met_names,
#                              T=None, t=None, t_scale=ts_now, pre_done=pre_done, **callback_kwargs)
#                 except Exception as e:
#                     print(f"[callback warning] pre acc={pre_done}: {e}")

#         Ecur = _energy(Mcur, Mt, energy_metric)
#         if writer:
#             writer.writerow(["pre", pre_done, -1, -1, 0.0, Ecur, 0.0, 1.0])

#     if t is None:
#         t = float(np.mean(deltaM_samples)) if deltaM_samples else 1.0
#     if (not np.isfinite(t)) or (t <= 0):
#         t = 1.0

#     T_base  = float(t)
#     T_start = float(t_scale_start) * T_base
#     T_min   = float(t_scale_min)   * T_base
#     K_steps = max(1, int(n_accept // max(1, t_scale_every)))

#     if t_step in (None, "auto", "auto_linear"):
#         if t_step_units == "temperature":
#             eff_t_step = (T_start - T_min) / K_steps
#         else:  # "scale"
#             eff_t_step = (float(t_scale_start) - float(t_scale_min)) / K_steps
#     else:
#         eff_t_step = float(t_step)

#     def _get_T(accepted, proposals, pre_done):
#         cnt = _counter(accepted, proposals, pre_done)
#         steps = int(cnt // max(1, t_scale_every))
#         if t_step_units == "temperature":
#             T_now = T_start - steps * eff_t_step
#             if T_now < T_min: T_now = T_min
#             return T_now
#         else:
#             ts = float(t_scale_start) - steps * eff_t_step
#             if ts < float(t_scale_min): ts = float(t_scale_min)
#             return T_base * ts

#     if t_step_units == "temperature":
#         print(f"[info] temperature base t = {T_base:.6g}")
#         print(f"[info] schedule in T-units: T_start={T_start:.6g}, T_min={T_min:.6g}, "
#               f"step={eff_t_step:.6g} every={t_scale_every} ({t_scale_count_by}), K≈{K_steps}")
#     else:
#         print(f"[info] temperature base t = {T_base:.6g}")
#         print(f"[info] schedule in scale-units: start={t_scale_start:.6g}, min={t_scale_min:.6g}, "
#               f"step={eff_t_step:.6g} every={t_scale_every} ({t_scale_count_by}), K≈{K_steps}")

#     while accepted < n_accept:
#         T = _get_T(accepted, proposals, pre_done)

#         proposals += 1
#         u, v = rng.sample(nodes, 2)
#         if np.allclose(Xc[pi[u]], Xc[pi[v]], atol=1e-12, rtol=0):
#             continue

#         dlt  = local_delta(u, v)
#         Mnew = Mcur + dlt
#         Enew = _energy(Mnew, Mt, energy_metric)
#         dE   = Enew - Ecur
#         x    = dE / T

#         if dE <= 0:
#             acc_prob = 1.0
#             accept = True
#         else:
#             u01 = rng.random()
#             accept = (u01 == 0.0) or (math.log(u01) < -x)
#             acc_prob = math.exp(-x) if x < 700 else 0.0

#         if accept:
#             pi[u], pi[v] = pi[v], pi[u]
#             Mcur = Mnew
#             Ecur = Enew
#             accepted += 1
#             touched[u] = True; touched[v] = True

#             if print_M:
#                 print(f"[acc={accepted}] (global={pre_done+accepted}) E={Ecur:.6g} dE={dE:.3g} acc={acc_prob:.3f}")

#             if print_mean:
#                 cur_mean = float(np.mean(X_mat[pi]))
#                 print(f"[acc={accepted}] mean_abundance = {cur_mean:.6g} (baseline {base_mean:.6g})")

#             if save_M_history:
#                 Ms_log.append(Mcur.copy())

#             global_acc = pre_done + accepted

#             if acc_every and (global_acc % acc_every == 0):
#                 for m in snap_idx:
#                     mdir = met_dirs[m]
#                     title = f"{met_names[m] if met_names else f'm{m}'} | acc={global_acc}, seed={seed}"
#                     snapshot_after_only(coords, X_mat, pi, m,
#                                         out_path=os.path.join(mdir, f"acc{global_acc:06d}.png"),
#                                         title_info=title)

#             if callback is not None and callback_every and (global_acc % callback_every == 0):
#                 try:
#                     T_now  = T
#                     ts_now = (T_now / T_base) if (T_base > 0) else None
#                     print(f"[mcmc->cb] mh:   global_acc={global_acc}, accepted={accepted}, lenMs={len(Ms_log)}")
#                     callback(Ms_log, Mt, accepted, global_acc, pi,
#                              coords=coords, X_mat=X_mat, met_names=met_names,
#                              T=T_now, t=T_base, t_scale=ts_now, pre_done=pre_done, **callback_kwargs)
#                 except Exception as e:
#                     print(f"[callback warning] mh acc={global_acc}: {e}")

#     if writer:
#         f.close()

#     T_final = _get_T(accepted, proposals, pre_done)

#     return {
#         "pi": pi,
#         "M0": Mt,
#         "M_final": Mcur,
#         "M_history": (np.stack(Ms_log, 0) if save_M_history else None),
#         "eps": eps,
#         "accepted": accepted,
#         "proposals": proposals,
#         "coverage": float(touched.mean()),
#         "snapshot_metabolites": [met_names[m] if met_names else m for m in snap_idx],
#         "snapshot_root": snapshot_root,
#         "seed": seed,
#         "energy_metric": energy_metric,
#         "pre_shuffle_swaps": pre_shuffle_swaps,
#         "T": T_final, "T_base": T_base, "T_scale": (T_final / T_base if T_base > 0 else None),
#         "T_start": T_start, "T_min": T_min,
#         "t_step": eff_t_step,
#         "t_step_units": t_step_units,
#         "t_scale_start": t_scale_start,
#         "t_scale_min": t_scale_min,
#         "t_scale_every": t_scale_every,
#         "t_scale_count_by": t_scale_count_by,
#         "pre_done": pre_done,
#     }


# In[ ]:


def mcmc_shuffle_diag_delta_seqshots(
    G, coords, X_mat,
    n_accept=5000,                   # ← 可设为 None，表示不以“目标接受数”为终止条件
    t=None,
    t_scale=1.0,
    eps=None,
    seed=None,
    acc_every=150,
    pre_snap_every=300,
    snapshot_metabolites=None,
    met_names=None,
    topk_for_snapshot=6,
    snapshot_root="snapshots_after_only",
    log_csv=None,
    save_initial=True,
    print_M=False,
    save_M_history=True,
    monitor_cols=None,
    print_mean=False,

    pre_shuffle_swaps=1000,
    weight_key="w",
    energy_metric="l1",

    callback=None,
    callback_every=None,
    callback_kwargs=None,

    t_scale_start=2.0,
    t_scale_min=0.1,
    t_step=0.1,
    t_scale_every=100,
    t_scale_count_by="accepted",
    t_step_units="scale",

    # ---- Early stopping (new) ----
    stop_on_energy=True,
    energy_abs_tol=None,     # 若 None，自动根据数据量级设定
    energy_rel_tol=1e-5,     # 相对阈值（相对 E_init）
    k_consec=200,   
    stop_on_plateau=True,
    patience_accept=1000,
    energy_min_drop=0.0,
    max_total_steps=30000
):
    import os, csv, math, random

    # ---- 若 X_mat 是 pandas，则转 numpy ----
    try:
        import pandas as pd
        if isinstance(X_mat, (pd.DataFrame, pd.Series)):
            X_mat = X_mat.to_numpy()
    except Exception:
        pass

    rng = random.Random(seed)
    N, M = X_mat.shape
    nodes = list(range(N))

    Xc = X_mat - X_mat.mean(axis=0, keepdims=True)
    Mt = target_diag_M_from_matrix(G, Xc, weight_key=weight_key)


    if eps is None:
        nz = np.abs(Mt[np.abs(Mt) > 0])
        base = np.median(nz) if nz.size else 1.0
        eps = 1e-9 * base

    use_weighted = (weight_key is not None)

    pi = np.arange(N, dtype=int)


    S = None
    if use_weighted:
        S = precalc_neighbor_sums(G, pi, Xc, weight_key=weight_key)

    # local_delta（优先缓存）
    def local_delta(u, v):
        if use_weighted:
            if S is not None:
                return local_delta_diag_perm_weighted_cached(G, pi, Xc, u, v, S, weight_key=weight_key)
            else:
                return local_delta_diag_perm_weighted(G, pi, Xc, u, v, weight_key=weight_key)
        else:
            return local_delta_diag_perm(G, pi, Xc, u, v)

    # 快照索引
    if snapshot_metabolites is None:
        idx = np.argsort(-np.abs(Mt))
        snap_idx = list(idx[:min(topk_for_snapshot, M)])
    else:
        if met_names is None:
            snap_idx = [int(i) for i in snapshot_metabolites]
        else:
            name_to_idx = {n:i for i,n in enumerate(met_names)}
            snap_idx = [name_to_idx[m] if isinstance(m, str) else int(m) for m in snapshot_metabolites]

    met_dirs = {}
    for m in snap_idx:
        mname = met_names[m] if met_names else f"m{m}"
        mdir = os.path.join(snapshot_root, slugify(mname))
        os.makedirs(mdir, exist_ok=True)
        met_dirs[m] = mdir
        if save_initial:
            init_title = f"{met_names[m] if met_names else f'm{m}'} | acc=0, seed={seed}"
            snapshot_after_only(coords, X_mat, pi, m,
                                out_path=os.path.join(mdir, "acc000000.png"),
                                title_info=init_title)

    accepted = 0
    proposals = 0
    touched = np.zeros(N, dtype=bool)

    writer = None; f = None
    if log_csv:
        log_dir = os.path.dirname(log_csv)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        f = open(log_csv, "w", newline="", encoding="utf-8")
        writer = csv.writer(f)
        writer.writerow(["phase","accepted_idx","u","v","coverage","Ecur","dE","acc_prob"])

    Mcur = Mt.copy()
    Ecur = _energy(Mcur, Mt, energy_metric)
    Ms_log = [] if save_M_history else None
    base_mean = float(np.mean(X_mat))
    if save_M_history:
        Ms_log.append(Mcur.copy())

    # 退火计数器
    if callback_every is None:
        callback_every = acc_every
    if callback_kwargs is None:
        callback_kwargs = {}

    def _counter(accepted, proposals, pre_done):
        if t_scale_count_by == "proposal":
            return proposals
        elif t_scale_count_by == "global":
            return pre_done + accepted
        else:  # "accepted"
            return accepted

    # callback init
    try:
        if callback is not None:
            print(f"[mcmc->cb] init: global_acc=0, accepted=0, lenMs={len(Ms_log) if Ms_log is not None else None}")
            callback(Ms_log, Mt, 0, 0, pi,
                     coords=coords, X_mat=X_mat, met_names=met_names,
                     T=None, t=None, t_scale=float(t_scale_start), pre_done=0, **callback_kwargs)
    except Exception as e:
        print(f"[callback warning] init failed: {e}")

    E_init = float(Ecur)
    if energy_abs_tol is None:
        nz = np.abs(Mt[np.abs(Mt) > 0])
        scale = (np.median(nz) * Mt.size) if nz.size else 1.0
        energy_abs_tol_eff = max(eps * Mt.size, 1e-8 * scale)
    else:
        energy_abs_tol_eff = float(energy_abs_tol)

    best_E = float(Ecur)
    best_at_accept = 0
    consec_ok = 0
    accept_since_start = 0
    early_stop_reason = None

    deltaM_samples = []
    pre_done = 0
    while pre_shuffle_swaps and (pre_done < pre_shuffle_swaps) and (proposals < max_total_steps):
        proposals += 1
        u, v = rng.sample(nodes, 2)

        dlt = local_delta(u, v)
        if use_weighted and (S is not None):
            apply_swap_and_update_cache(G, pi, Xc, u, v, S, weight_key=weight_key, swap_pi=True)
        else:
            pi[u], pi[v] = pi[v], pi[u]

        Mcur = Mcur + dlt
        pre_done += 1
        if save_M_history:
            Ms_log.append(Mcur.copy())

        deltaM_samples.append(float(np.sum(np.abs(dlt))))

        if pre_snap_every and (pre_done % pre_snap_every == 0):
            for m in snap_idx:
                mdir = met_dirs[m]
                title = f"{met_names[m] if met_names else f'm{m}'} | acc={pre_done}, seed={seed}"
                snapshot_after_only(coords, X_mat, pi, m,
                                    out_path=os.path.join(mdir, f"acc{pre_done:06d}.png"),
                                    title_info=title)
        if callback is not None and callback_every and (pre_done % callback_every == 0):
            try:
                ts_now = float(t_scale_start)
                print(f"[mcmc->cb] pre:  global_acc={pre_done}, accepted=0, lenMs={len(Ms_log)}")
                callback(Ms_log, Mt, 0, pre_done, pi,
                         coords=coords, X_mat=X_mat, met_names=met_names,
                         T=None, t=None, t_scale=ts_now, pre_done=pre_done, **callback_kwargs)
            except Exception as e:
                print(f"[callback warning] pre acc={pre_done}: {e}")

    Ecur = _energy(Mcur, Mt, energy_metric)
    if writer:
        writer.writerow(["pre", pre_done, -1, -1, 0.0, Ecur, 0.0, 1.0])

    if t is None:
        t = float(np.mean(deltaM_samples)) if deltaM_samples else 1.0
    if (not np.isfinite(t)) or (t <= 0):
        t = 1.0

    T_base  = float(t)
    T_start = float(t_scale_start) * T_base
    T_min   = float(t_scale_min)   * T_base
    K_steps = max(1, int((n_accept or 1) // max(1, t_scale_every)))

    if t_step in (None, "auto", "auto_linear"):
        if t_step_units == "temperature":
            eff_t_step = (T_start - T_min) / K_steps
        else:
            eff_t_step = (float(t_scale_start) - float(t_scale_min)) / K_steps
    else:
        eff_t_step = float(t_step)

    def _get_T(accepted, proposals, pre_done):
        cnt = _counter(accepted, proposals, pre_done)
        steps = int(cnt // max(1, t_scale_every))
        if t_step_units == "temperature":
            T_now = T_start - steps * eff_t_step
            if T_now < T_min: T_now = T_min
            return T_now
        else:
            ts = float(t_scale_start) - steps * eff_t_step
            if ts < float(t_scale_min): ts = float(t_scale_min)
            return T_base * ts

    n_accept_cap = n_accept if (n_accept is not None) else max_total_steps

    if t_step_units == "temperature":
        print(f"[info] temperature base t = {T_base:.6g}")
        print(f"[info] schedule in T-units: T_start={T_start:.6g}, T_min={T_min:.6g}, "
              f"step={eff_t_step:.6g} every={t_scale_every} ({t_scale_count_by}), K≈{K_steps}")
    else:
        print(f"[info] temperature base t = {T_base:.6g}")
        print(f"[info] schedule in scale-units: start={t_scale_start:.6g}, min={t_scale_min:.6g}, "
              f"step={eff_t_step:.6g} every={t_scale_every} ({t_scale_count_by}), K≈{K_steps}")


    while (accepted < n_accept_cap) and (proposals < max_total_steps):
        T = _get_T(accepted, proposals, pre_done)

        proposals += 1
        u, v = rng.sample(nodes, 2)

        dlt  = local_delta(u, v)
        Mnew = Mcur + dlt
        Enew = _energy(Mnew, Mt, energy_metric)
        dE   = Enew - Ecur
        x    = dE / T

        if dE <= 0:
            acc_prob = 1.0
            accept = True
        else:
            u01 = rng.random()
            accept = (u01 == 0.0) or (math.log(u01) < -x)
            acc_prob = math.exp(-x) if x < 700 else 0.0

        if accept:
            if use_weighted and (S is not None):
                apply_swap_and_update_cache(G, pi, Xc, u, v, S, weight_key=weight_key, swap_pi=True)
            else:
                pi[u], pi[v] = pi[v], pi[u]

            Mcur = Mnew
            Ecur = Enew
            accepted += 1
            accept_since_start += 1
            touched[u] = True; touched[v] = True

            if writer:
                writer.writerow(["mh", accepted, u, v, float(touched.mean()), Ecur, dE, acc_prob])
            if print_M:
                print(f"[acc={accepted}] (global={pre_done+accepted}) E={Ecur:.6g} dE={dE:.3g} acc={acc_prob:.3f}")
            if print_mean:
                cur_mean = float(np.mean(X_mat[pi]))
                print(f"[acc={accepted}] mean_abundance = {cur_mean:.6g} (baseline {base_mean:.6g})")
            if save_M_history:
                Ms_log.append(Mcur.copy())

            global_acc = pre_done + accepted
            if acc_every and (global_acc % acc_every == 0):
                for m in snap_idx:
                    mdir = met_dirs[m]
                    title = f"{met_names[m] if met_names else f'm{m}'} | acc={global_acc}, seed={seed}"
                    snapshot_after_only(coords, X_mat, pi, m,
                                        out_path=os.path.join(mdir, f"acc{global_acc:06d}.png"),
                                        title_info=title)
            if callback is not None and callback_every and (global_acc % callback_every == 0):
                try:
                    T_now  = T
                    ts_now = (T_now / T_base) if (T_base > 0) else None
                    print(f"[mcmc->cb] mh:   global_acc={global_acc}, accepted={accepted}, lenMs={len(Ms_log)}")
                    callback(Ms_log, Mt, accepted, global_acc, pi,
                             coords=coords, X_mat=X_mat, met_names=met_names,
                             T=T_now, t=T_base, t_scale=ts_now, pre_done=pre_done, **callback_kwargs)
                except Exception as e:
                    print(f"[callback warning] mh acc={global_acc}: {e}")

            improved = (Ecur <= best_E - energy_min_drop)
            if improved:
                best_E = float(Ecur)
                best_at_accept = accept_since_start

            if stop_on_energy and (
                (Ecur <= energy_abs_tol_eff) or (Ecur <= energy_rel_tol * E_init)
            ):
                consec_ok += 1
            else:
                consec_ok = 0
            if stop_on_plateau and (accept_since_start - best_at_accept >= patience_accept):
                early_stop_reason = (
                    f"plateau(no improvement ≥{energy_min_drop} in last "
                    f"{patience_accept} accepts; E={Ecur:.6g})"
                )
                break

            if stop_on_energy and consec_ok >= k_consec:
                early_stop_reason = (
                    f"energy_threshold(E={Ecur:.3g} ≤ max({energy_abs_tol_eff:.3g}, "
                    f"{energy_rel_tol:.1e}*E_init)) for {k_consec} accepts"
                )
                break


    if writer:
        f.close()

    T_final = _get_T(accepted, proposals, pre_done)

    return {
        "pi": pi,
        "M0": Mt,
        "M_final": Mcur,
        "M_history": (np.stack(Ms_log, 0) if save_M_history else None),
        "eps": eps,
        "accepted": accepted,
        "proposals": proposals,
        "coverage": float(touched.mean()),
        "snapshot_metabolites": [met_names[m] if met_names else m for m in snap_idx],
        "snapshot_root": snapshot_root,
        "seed": seed,
        "energy_metric": energy_metric,
        "pre_shuffle_swaps": pre_shuffle_swaps,
        "T": T_final, "T_base": T_base, "T_scale": (T_final / T_base if T_base > 0 else None),
        "T_start": T_start, "T_min": T_min,
        "t_step": eff_t_step,
        "t_step_units": t_step_units,
        "t_scale_start": t_scale_start,
        "t_scale_min": t_scale_min,
        "t_scale_every": t_scale_every,
        "t_scale_count_by": t_scale_count_by,
        "pre_done": pre_done,
        "early_stop_reason": early_stop_reason,
        "best_E": best_E,
        "E_init": E_init,
        "energy_abs_tol_eff": energy_abs_tol_eff,
        "energy_rel_tol": energy_rel_tol,
        "k_consec": k_consec,
        "patience_accept": patience_accept,
        "energy_min_drop": energy_min_drop,
        "max_total_steps": max_total_steps,
    }


# In[101]:


#############--------New-version---------##################


# In[ ]:


def mcmc_shuffle_diag_delta_seqshots(
    G, coords, X_mat,
    n_accept=None,
    t=None,
    t_scale=1.0,
    eps=None,
    seed=None,
    acc_every=150,
    pre_snap_every=300,
    snapshot_metabolites=None,
    met_names=None,
    topk_for_snapshot=6,
    snapshot_root="snapshots_after_only",
    log_csv=None,
    save_initial=True,
    print_M=False,
    save_M_history=True,
    monitor_cols=None,
    print_mean=True,

    pre_shuffle_swaps=1000,
    weight_key="w", 
    energy_metric="l1",

    callback=None,
    callback_every=None,
    callback_kwargs=None,

    t_scale_start=2.0,
    t_scale_min=0.1,
    t_step=0.1,
    t_scale_every=100,
    t_scale_count_by="accepted",
    t_step_units="scale",

    stop_on_energy=True,
    energy_abs_tol=None,
    energy_rel_tol=1e-6,
    k_consec=200,
    stop_on_plateau=True,
    patience_accept=8000,
    energy_min_drop=0.0,
    stable_window=500,
    stable_std_tol=None, 
    min_accept_before_stop=2000,
    max_total_steps=20000
):
    import os, csv, math, random
    from collections import deque

    # pandas -> numpy
    try:
        import pandas as pd
        if isinstance(X_mat, (pd.DataFrame, pd.Series)):
            X_mat = X_mat.to_numpy()
    except Exception:
        pass

    rng = random.Random(seed)
    N, P = X_mat.shape
    nodes = list(range(N))

    Xc = X_mat - X_mat.mean(axis=0, keepdims=True)
    Mt = target_diag_M_from_matrix(G, Xc, weight_key=weight_key)

    # eps
    if eps is None:
        nz = np.abs(Mt[np.abs(Mt) > 0])
        base = np.median(nz) if nz.size else 1.0
        eps = 1e-9 * base

    pi = np.arange(N, dtype=int)
    S = precalc_neighbor_sums(G, pi, Xc, weight_key=weight_key)

    def local_delta(u, v):
        return local_delta_diag_perm_weighted_cached(G, pi, Xc, u, v, S, weight_key=weight_key)

    if snapshot_metabolites is None:
        idx = np.argsort(-np.abs(Mt))
        snap_idx = list(idx[:min(topk_for_snapshot, P)])
    else:
        if met_names is None:
            snap_idx = [int(i) for i in snapshot_metabolites]
        else:
            name_to_idx = {n:i for i,n in enumerate(met_names)}
            snap_idx = [name_to_idx[m] if isinstance(m, str) else int(m) for m in snapshot_metabolites]

    met_dirs = {}
    for m in snap_idx:
        mname = met_names[m] if met_names else f"m{m}"
        mdir = os.path.join(snapshot_root, slugify(mname))
        os.makedirs(mdir, exist_ok=True)
        met_dirs[m] = mdir
        if save_initial:
            init_title = f"{met_names[m] if met_names else f'm{m}'} | acc=0, seed={seed}"
            snapshot_after_only(coords, X_mat, pi, m,
                                out_path=os.path.join(mdir, "acc000000.png"),
                                title_info=init_title)

    accepted = 0
    proposals = 0
    touched = np.zeros(N, dtype=bool)
    writer = None; f = None
    if log_csv:
        log_dir = os.path.dirname(log_csv)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        f = open(log_csv, "w", newline="", encoding="utf-8")
        writer = csv.writer(f)
        writer.writerow(["phase","accepted_idx","u","v","coverage","Ecur","dE","acc_prob"])

    Mcur = Mt.copy()
    Ecur = _energy(Mcur, Mt, energy_metric)
    Ms_log = [] if save_M_history else None
    base_mean = float(np.mean(X_mat))
    if save_M_history:
        Ms_log.append(Mcur.copy())

    if callback_every is None:
        callback_every = acc_every
    if callback_kwargs is None:
        callback_kwargs = {}

    def _counter(accepted, proposals, pre_done):
        if t_scale_count_by == "proposal":
            return proposals
        elif t_scale_count_by == "global":
            return pre_done + accepted
        else:
            return accepted

    deltaM_samples = []
    pre_done = 0
    while pre_shuffle_swaps and (pre_done < pre_shuffle_swaps) and (proposals < max_total_steps):
        proposals += 1
        u, v = rng.sample(nodes, 2)

        dlt = local_delta(u, v)
        apply_swap_and_update_cache(G, pi, Xc, u, v, S, weight_key=weight_key, swap_pi=True)
        Mcur = Mcur + dlt
        pre_done += 1

        if save_M_history:
            Ms_log.append(Mcur.copy())

        deltaM_samples.append(float(np.sum(np.abs(dlt))))

        if pre_snap_every and (pre_done % pre_snap_every == 0):
            for m in snap_idx:
                mdir = met_dirs[m]
                title = f"{met_names[m] if met_names else f'm{m}'} | acc={pre_done}, seed={seed}"
                snapshot_after_only(coords, X_mat, pi, m,
                                    out_path=os.path.join(mdir, f"acc{pre_done:06d}.png"),
                                    title_info=title)

        if callback is not None and callback_every and (pre_done % callback_every == 0):
            try:
                ts_now = float(t_scale_start)
                callback(Ms_log, Mt, 0, pre_done, pi,
                         coords=coords, X_mat=X_mat, met_names=met_names,
                         T=None, t=None, t_scale=ts_now, pre_done=pre_done, **callback_kwargs)
            except Exception as e:
                print(f"[callback warning] pre acc={pre_done}: {e}")

    Ecur = _energy(Mcur, Mt, energy_metric)
    if writer:
        writer.writerow(["pre", pre_done, -1, -1, 0.0, Ecur, 0.0, 1.0])

    E_init = float(Ecur)
    if energy_abs_tol is None:
        nz = np.abs(Mt[np.abs(Mt) > 0])
        scale = (np.median(nz) * Mt.size) if nz.size else 1.0
        energy_abs_tol_eff = max(eps * Mt.size, 1e-8 * scale)
    else:
        energy_abs_tol_eff = float(energy_abs_tol)

    best_E = float(Ecur)
    best_at_accept = 0
    consec_ok = 0
    accept_since_start = 0
    early_stop_reason = None

    energy_window = deque(maxlen=int(stable_window))
    if stable_std_tol is None:
        stable_std_tol_eff = max(1e-6 * E_init, 1e-12)
    else:
        stable_std_tol_eff = float(stable_std_tol)

    if t is None:
        t = float(np.mean(deltaM_samples)) if deltaM_samples else 1.0
    if (not np.isfinite(t)) or (t <= 0):
        t = 1.0

    T_base  = float(t)
    T_start = float(t_scale_start) * T_base
    T_min   = float(t_scale_min)   * T_base
    K_steps = max(1, int((n_accept or 1) // max(1, t_scale_every)))

    if t_step in (None, "auto", "auto_linear"):
        eff_t_step = (T_start - T_min) / K_steps if t_step_units == "temperature" \
                     else (float(t_scale_start) - float(t_scale_min)) / K_steps
    else:
        eff_t_step = float(t_step)

    def _get_T(accepted, proposals, pre_done):
        cnt = _counter(accepted, proposals, pre_done)
        steps = int(cnt // max(1, t_scale_every))
        if t_step_units == "temperature":
            T_now = T_start - steps * eff_t_step
            return T_now if T_now >= T_min else T_min
        else:
            ts = float(t_scale_start) - steps * eff_t_step
            if ts < float(t_scale_min): ts = float(t_scale_min)
            return T_base * ts

    n_accept_cap = n_accept if (n_accept is not None) else max_total_steps

    while (accepted < n_accept_cap) and (proposals < max_total_steps):
        T = _get_T(accepted, proposals, pre_done)

        proposals += 1
        u, v = rng.sample(nodes, 2)

        dlt  = local_delta(u, v)
        Mnew = Mcur + dlt
        Enew = _energy(Mnew, Mt, energy_metric)
        dE   = Enew - Ecur
        x    = dE / T

        if dE <= 0:
            acc_prob = 1.0
            accept = True
        else:
            u01 = rng.random()
            accept = (u01 == 0.0) or (math.log(u01) < -x)
            acc_prob = math.exp(-x) if x < 700 else 0.0

        if accept:
            apply_swap_and_update_cache(G, pi, Xc, u, v, S, weight_key=weight_key, swap_pi=True)
            Mcur = Mnew
            Ecur = Enew
            accepted += 1
            accept_since_start += 1
            touched[u] = True; touched[v] = True

            energy_window.append(Ecur)

            if writer:
                writer.writerow(["mh", accepted, u, v, float(touched.mean()), Ecur, dE, acc_prob])
            if print_M:
                print(f"[acc={accepted}] (global={pre_done+accepted}) E={Ecur:.6g} dE={dE:.3g} acc={acc_prob:.3f}")
            if print_mean:
                cur_mean = float(np.mean(X_mat[pi]))
                print(f"[acc={accepted}] mean_abundance = {cur_mean:.6g} (baseline {base_mean:.6g})")
            if save_M_history:
                Ms_log.append(Mcur.copy())

            global_acc = pre_done + accepted
            if acc_every and (global_acc % acc_every == 0):
                for m in snap_idx:
                    mdir = met_dirs[m]
                    title = f"{met_names[m] if met_names else f'm{m}'} | acc={global_acc}, seed={seed}"
                    snapshot_after_only(coords, X_mat, pi, m,
                                        out_path=os.path.join(mdir, f"acc{global_acc:06d}.png"),
                                        title_info=title)
            if callback is not None and callback_every and (global_acc % callback_every == 0):
                try:
                    T_now  = T
                    ts_now = (T_now / T_base) if (T_base > 0) else None
                    callback(Ms_log, Mt, accepted, global_acc, pi,
                             coords=coords, X_mat=X_mat, met_names=met_names,
                             T=T_now, t=T_base, t_scale=ts_now, pre_done=pre_done, **callback_kwargs)
                except Exception as e:
                    print(f"[callback warning] mh acc={global_acc}: {e}")

            # —— 记录改进
            if Ecur <= best_E - energy_min_drop:
                best_E = float(Ecur)
                best_at_accept = accept_since_start

            if stop_on_energy and ((Ecur <= energy_abs_tol_eff) or (Ecur <= energy_rel_tol * E_init)):
                consec_ok += 1
            else:
                consec_ok = 0

            if accept_since_start >= int(min_accept_before_stop):

                if len(energy_window) == energy_window.maxlen:
                    mean_E = float(np.mean(energy_window))
                    std_E  = float(np.std(energy_window))
                    mean_ok = (mean_E <= energy_abs_tol_eff) or (mean_E <= energy_rel_tol * E_init)
                    std_ok  = (std_E <= stable_std_tol_eff)
                    if mean_ok and std_ok:
                        early_stop_reason = (
                            f"stable_window(W={energy_window.maxlen}, mean={mean_E:.3g}, "
                            f"std={std_E:.3g}, abs_tol={energy_abs_tol_eff:.3g}, "
                            f"rel_tol={energy_rel_tol:.1e}*E_init)"
                        )
                        break

                if stop_on_energy and consec_ok >= int(k_consec):
                    early_stop_reason = (
                        f"energy_threshold(after {min_accept_before_stop} accepts; "
                        f"E={Ecur:.3g} ≤ max({energy_abs_tol_eff:.3g}, "
                        f"{energy_rel_tol:.1e}*E_init)) for {k_consec} accepts"
                    )
                    break

                if stop_on_plateau and (accept_since_start - best_at_accept >= int(patience_accept)):
                    early_stop_reason = (
                        f"plateau(no improvement ≥{energy_min_drop} in last "
                        f"{patience_accept} accepts; E={Ecur:.6g})"
                    )
                    break

    if writer:
        f.close()

    T_final = _get_T(accepted, proposals, pre_done)

    return {
        "pi": pi,
        "M0": Mt,
        "M_final": Mcur,
        "M_history": (np.stack(Ms_log, 0) if save_M_history else None),
        "eps": eps,
        "accepted": accepted,
        "proposals": proposals,
        "coverage": float(touched.mean()),
        "snapshot_metabolites": [met_names[m] if met_names else m for m in snap_idx],
        "snapshot_root": snapshot_root,
        "seed": seed,
        "energy_metric": energy_metric,
        "pre_shuffle_swaps": pre_shuffle_swaps,
        "T": T_final, "T_base": T_base, "T_scale": (T_final / T_base if T_base > 0 else None),
        "T_start": T_start, "T_min": T_min,
        "t_step": eff_t_step,
        "t_step_units": t_step_units,
        "t_scale_start": t_scale_start,
        "t_scale_min": t_scale_min,
        "t_scale_every": t_scale_every,
        "t_scale_count_by": t_scale_count_by,
        "pre_done": pre_done,

        "early_stop_reason": early_stop_reason,
        "best_E": best_E,
        "E_init": E_init,
        "energy_abs_tol_eff": energy_abs_tol_eff,
        "energy_rel_tol": energy_rel_tol,
        "k_consec": k_consec,
        "stable_window": stable_window,
        "stable_std_tol": stable_std_tol if stable_std_tol is not None else stable_std_tol_eff,
        "min_accept_before_stop": min_accept_before_stop,
        "patience_accept": patience_accept,
        "energy_min_drop": energy_min_drop,
        "max_total_steps": max_total_steps,
    }


# In[103]:


def plot_callback(Ms_log, Mt, accepted, global_acc, pi, coords=None, X_mat=None, met_names=None, state={}, **kwargs):

    import numpy as np
    import matplotlib.pyplot as plt
    if Ms_log is None or len(Ms_log) == 0:
        return

    if 'fig' not in state:
        plt.ion()
        fig, ax = plt.subplots(figsize=(8,4.5))
        line, = ax.plot([], [], label="Mnew[:,0]")
        hline = ax.axhline(float(Mt[0]), linestyle='--', label='M0[0]')
        ax.set_xlabel("Accepted step")
        ax.set_ylabel("M after swap (Mnew) [met0]")
        ax.set_title("Realtime Mnew trend (met0)")
        ax.legend()
        state.update({'fig': fig, 'ax': ax, 'line': line, 'hline': hline})

    fig = state['fig']; ax = state['ax']; line = state['line']; hline = state['hline']

    y = np.asarray(Ms_log)[:, 0]
    xs = np.arange(1, len(y) + 1)
    line.set_data(xs, y)
    ax.relim(); ax.autoscale_view()
    fig.canvas.draw()
    plt.pause(0.001)


# In[104]:


def plot_callback(Ms_log, Mt, accepted, global_acc, pi,
                  coords=None, X_mat=None, met_names=None,
                  state=None, figsize=(12,5), **kwargs):
    import numpy as np
    import matplotlib.pyplot as plt
    from IPython.display import display

    if Ms_log is None or len(Ms_log) == 0:
        return

    if state is None:
        state = {}

    y = np.asarray(Ms_log)[:, 0]
    x = np.arange(0, len(y))

    if 'fig' not in state:
        fig, ax = plt.subplots(figsize=figsize)
        (line,) = ax.plot(x, y, label="M[:,0]")
        ax.axhline(float(Mt[0]), linestyle='--', label='M0[0]')
        ax.set_xlabel("Global step index (0 = M0)")
        ax.set_ylabel("M after swap (met0)")
        ax.set_title("Realtime M trend (met0)")
        ax.legend(loc='best')
        state['fig'] = fig
        state['ax'] = ax
        state['line'] = line
        backend = plt.get_backend().lower()
        inline_like = ('inline' in backend) or ('ipykernel' in backend)
        state['inline_like'] = inline_like

        if inline_like:
            state['display_handle'] = display(fig, display_id=True)
        else:
            plt.ion()
            fig.show()

    fig = state['fig']; ax = state['ax']; line = state['line']
    line.set_data(x, y)
    ax.set_xlim(0, max(100, len(y)-1))
    ymin = min(float(Mt[0]), float(np.min(y)))
    ymax = max(float(Mt[0]), float(np.max(y)))
    if ymin == ymax:
        ax.set_ylim(ymin - 0.5, ymax + 0.5)
    else:
        pad = 0.06 * (ymax - ymin)
        ax.set_ylim(ymin - pad, ymax + pad)

    if state.get('inline_like', False):
        state['display_handle'].update(fig)
    else:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()



# In[105]:


cols = msi_norm_hvg_162.columns.tolist()
X1   = msi_norm_hvg_162.to_numpy(dtype=float)
met_names = [str(c) for c in cols]


# In[106]:


print(met_names)


# In[107]:


print(X1)


# In[108]:


# coords, X1, spot_ids, met_names = align_inputs(coords_df, msi_norm_hvg_162)
# X1 = np.asarray(X1, dtype=float)

# import math, numpy as np
# dmin = math.sqrt(2.0)
# l = 4 * dmin
# G = build_weighted_radius_graph(coords, l=l, radius=float('inf'), weight_key="w")

# state = {}
# res = mcmc_shuffle_diag_delta_seqshots(
#     G, coords, X1,
#     n_accept=30000,
#     pre_shuffle_swaps=5000,
#     save_M_history=True,
#     met_names=met_names,

#     acc_every=200,
#     snapshot_root = "results_by_metabolite_l=4",
#     callback=plot_callback,
#     callback_every=200,
#     callback_kwargs={'state': state},
#     topk_for_snapshot=len(met_names),

#     t=None,
#     t_scale_start=2.0,
#     t_scale_min=0.1,
#     t_step="auto",
#     t_scale_every=100,
#     t_scale_count_by="accepted",
#     t_step_units="temperature",

#     seed=None,
#     weight_key="w",
#     energy_metric="l1",
#     print_M=True,
# )

# M_hist = res["M_history"]
# M0     = res["M0"]

# print("accepted =", res["accepted"], "proposals =", res["proposals"])
# print("M_history shape:", None if M_hist is None else M_hist.shape)
# print("Temperature used: T =", res.get("T"), "| base =", res.get("T_base"), "| scale =", res.get("T_scale"))

# expected_len = 1 + res["pre_shuffle_swaps"] + res["accepted"]
# if M_hist is not None:
#     print("Expected length:", expected_len, " | actual:", M_hist.shape[0])
#     print("Last equals M_final:", np.allclose(M_hist[-1], res["M_final"]))

    


# In[ ]:


from pathlib import Path
import numpy as np
import pandas as pd
import re

def slugify(s: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s]+', "_", str(s))

def ensure_mdir_for_met(met_name: str, root="results_by_metabolite", subdir="metabolite",
                        snapshot_dirs=None) -> Path:
    if isinstance(snapshot_dirs, dict) and met_name in snapshot_dirs:
        mdir = Path(snapshot_dirs[met_name])
    else:
        mdir = Path(root) / slugify(met_name) / subdir
    mdir.mkdir(parents=True, exist_ok=True)
    return mdir

def save_shuffled_msis(X_mat, met_names, spot_ids, res,
                       root="results_by_metabolite", subdir="metabolite",
                       prefix="msi_norm_hvg_162_shuffled",
                       also_save_combined=False, float_fmt="%.6g"):
    from datetime import datetime
    X_mat = np.asarray(X_mat, dtype=float)
    pi = np.asarray(res["pi"], dtype=int)
    X_shuf = X_mat[pi, :]
    idx = pd.Index(spot_ids, name="spot_id")
    snapshot_dirs = res.get("snapshot_dirs", None)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_tag = f"{ts}_acc{res.get('accepted','NA')}_pre{res.get('pre_done','NA')}_seed{res.get('seed','NA')}"

    for j, met in enumerate(met_names):
        mdir = ensure_mdir_for_met(str(met), root=root, subdir=subdir, snapshot_dirs=snapshot_dirs)
        out_csv = mdir / f"{prefix}_{run_tag}.csv"
        pd.DataFrame({str(met): X_shuf[:, j]}, index=idx).to_csv(out_csv, float_format=float_fmt)
        meta_txt = mdir / f"{prefix}_{run_tag}.meta.txt"
        meta_txt.write_text(
            "run_tag: " + run_tag + "\n"
            f"accepted: {res.get('accepted')}\n"
            f"pre_shuffle_swaps: {res.get('pre_shuffle_swaps')}\n"
            f"seed: {res.get('seed')}\n"
            f"energy_metric: {res.get('energy_metric')}\n"
        )
        
def run_independent_per_metabolite(
    G, coords, coords_df, msi_df,
    n_accept=25000, pre_shuffle_swaps=5000,
    base_seed=10, 
    snapshot_root="results_by_metabolite",
    weight_key="w", energy_metric="l1",
    save_M_history=True,
    save_shuffled_csv=True
):
    coords_aligned, X_all, spot_ids, met_names = align_inputs(coords_df, msi_df)
    X_all = np.asarray(X_all, dtype=float)
    results = {}

    for j, met in enumerate(met_names):
        X_col = X_all[:, [j]]
        seed_j = int(base_seed + j)
        res = mcmc_shuffle_diag_delta_seqshots(
            G, coords_aligned, X_col,
            n_accept=n_accept,
            pre_shuffle_swaps=pre_shuffle_swaps,
            save_M_history=save_M_history,
            met_names=[met],
            snapshot_metabolites=[met],
            topk_for_snapshot=len(met_names),
            snapshot_root=snapshot_root,
            callback=None, callback_every=None,
            t=None,
            t_scale_start=2.0, t_scale_min=0.1,
            t_step="auto", t_scale_every=100,
            t_scale_count_by="accepted",
            t_step_units="temperature",
            seed=seed_j,
            weight_key=weight_key,
            energy_metric=energy_metric,
            print_M=False, print_mean=False,
        )
        results[str(met)] = res

        if save_shuffled_csv:
            save_shuffled_msis(X_col, [met], spot_ids, res,
                               root=snapshot_root, subdir="metabolite",
                               also_save_combined=False)

        print(f"[done] {met} | seed={seed_j} | accepted={res['accepted']}")

    return results

# msi_norm_hvg_162 = msi_norm_hvg.iloc[:, :5].copy()

import math
dmin = math.sqrt(2.0)
l = 4 * dmin
G = build_weighted_radius_graph(coords_df[["x","y"]].values, l=l, radius=float('inf'), weight_key="w")

# results = run_independent_per_metabolite(
#     G=G, coords=coords_df[["x","y"]].values, coords_df=coords_df,
#     msi_df=msi_norm_hvg_162,
#     n_accept=25000, pre_shuffle_swaps=5000,
#     base_seed=123,
#     snapshot_root="results_by_metabolite_l4_n",
#     weight_key="w", energy_metric="l1",
#     save_M_history=True,
#     save_shuffled_csv=True
# )


results = run_independent_per_metabolite(
    G=G,
    coords=coords_df[["x","y"]].values,
    coords_df=coords_df,
    msi_df=msi_norm_hvg_162,
    n_accept=None,
    pre_shuffle_swaps=3500,
    base_seed=7697,
    snapshot_root="results_by_metabolite_l4_n_es1",
    weight_key="w",
    energy_metric="l1",
    save_M_history=True,
    save_shuffled_csv=True
)


# M_hist = res["M_history"]
# M0     = res["M0"]

# print("accepted =", res["accepted"], "proposals =", res["proposals"])
# print("M_history shape:", None if M_hist is None else M_hist.shape)
# print("Temperature used: T =", res.get("T"), "| base =", res.get("T_base"), "| scale =", res.get("T_scale"))

# expected_len = 1 + res["pre_shuffle_swaps"] + res["accepted"]
# if M_hist is not None:
#     print("Expected length:", expected_len, " | actual:", M_hist.shape[0])
#     print("Last equals M_final:", np.allclose(M_hist[-1], res["M_final"]))


# In[110]:


# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import re
# from pathlib import Path

# def slugify(s: str) -> str:
#     return re.sub(r'[\\/:*?"<>|\s]+', "_", str(s))

# def plot_mtrend_series(y, y0, out_png, step=100):
#     y = np.asarray(y).reshape(-1)
#     x = np.arange(1, len(y)+1)
#     if step and step > 1:
#         x = x[::step]; y = y[::step]
#     plt.figure(figsize=(8, 4.5))
#     plt.plot(x, y, label="Mnew")
#     plt.hlines(float(y0), xmin=x[0], xmax=x[-1], linestyles="dashed", label="M0")
#     plt.xlabel("Accepted step"); plt.ylabel("M after swap (Mnew)")
#     plt.title("Mnew trend")
#     plt.legend(); plt.tight_layout()
#     plt.savefig(out_png, dpi=220); plt.close()

# M_history = res["M_history"]        # shape ≈ (pre_done+accepted+1, 5)
# Mt        = res["M0"]               # shape = (5,)
# root      = Path(res["snapshot_root"])

# for j, name in enumerate(met_names):
#     mdir = root / slugify(name)
#     mdir.mkdir(parents=True, exist_ok=True)
#     out_png = mdir / "M_trend.png"
#     plot_mtrend_series(M_history[:, j], Mt[j], out_png, step=100)
#     print(f"saved: {out_png}")


# In[ ]:


import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import re

def slugify(s: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s]+', "_", str(s))

def plot_mtrend_series(M_series, M0_scalar, out_png, step=100, title=None, dpi=300):

    M_series = np.asarray(M_series).ravel()
    x = np.arange(M_series.size)
    if step and step > 1:
        x = x[::step]
        y = M_series[::step]
    else:
        y = M_series

    plt.figure(figsize=(5.2, 2.8))
    plt.plot(x, y, lw=1.2)
    plt.axhline(float(M0_scalar), ls="--", lw=0.9, alpha=0.7)
    plt.xlabel("step (pre + accepted)")
    plt.ylabel("M")
    if title:
        plt.title(title)
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=dpi)
    plt.close()

def save_all_mtrends(res_or_results, met_names=None,
                     root_default="results_by_metabolite_4l_es1", subdir="metabolite",

                     step=50, dpi=300):

    if isinstance(res_or_results, dict) and res_or_results and isinstance(next(iter(res_or_results.values())), dict):
        for met, r in res_or_results.items():
            M_hist = r["M_history"]
            M0     = r["M0"]
            if M_hist.ndim == 2:
                series = M_hist[:, 0]
            else:
                series = M_hist
            M0_val = float(M0[0]) if np.ndim(M0) else float(M0)

            base = r.get("snapshot_root", root_default)
            mdir = Path(base) / slugify(str(met)) / subdir
            out_png = mdir / "M_trend.png"
            plot_mtrend_series(series, M0_val, out_png, step=step, title=f"Mtrend — {met}", dpi=dpi)
            print("saved:", out_png)

    else:
        r = res_or_results
        M_hist = r["M_history"]
        M0     = r["M0"]
        if met_names is None:
            met_names = [f"met_{j}" for j in range(M_hist.shape[1])]

        base = r.get("snapshot_root", root_default)
        for j, name in enumerate(met_names):
            mdir = Path(base) / slugify(str(name)) / subdir
            out_png = mdir / "M_trend.png"
            plot_mtrend_series(M_hist[:, j], float(M0[j]), out_png, step=step, title=f"Mtrend — {name}", dpi=dpi)
            print("saved:", out_png)


# In[112]:


save_all_mtrends(results, met_names=met_names, step=50, dpi=300)


# In[113]:


coords, X1, spot_ids, met_names = align_inputs(coords_df, msi_norm_hvg_162)

# import math, numpy as np
# dmin = math.sqrt(2)
# l = 7*dmin
# G = build_weighted_radius_graph(coords, l=l, radius=float('inf'), weight_key="w")

# state = {}
# res = mcmc_shuffle_diag_delta_seqshots(
#     G, coords, X1,
#     n_accept=50000,
#     pre_shuffle_swaps=5000,
#     save_M_history=True,

#     acc_every=200,
#     snapshot_root="snapshots_190_changingT_l=7_500",
#     callback=plot_callback,
#     callback_every=200,
#     callback_kwargs={'state': state},

#     t=None,
#     t_scale_start=2.0,
#     t_scale_min=0.1,
#     t_step="auto",
#     t_scale_every=100,
#     t_scale_count_by="accepted",
#     t_step_units="temperature", 

#     seed=None,
#     weight_key="w",
#     energy_metric="l1",
#     print_M=True,
# )

# M_hist = res["M_history"]
# M0     = res["M0"]

# print("accepted =", res["accepted"], "proposals =", res["proposals"])
# print("M_history shape:", None if M_hist is None else M_hist.shape)
# print("Temperature used: T =", res.get("T"), "| base =", res.get("T_base"), "| scale =", res.get("T_scale"))

# expected_len = 1 + res["pre_shuffle_swaps"] + res["accepted"]
# if M_hist is not None:
#     print("Expected length:", expected_len, " | actual:", M_hist.shape[0])
#     print("Last equals M_final:", np.allclose(M_hist[-1], res["M_final"]))


# In[114]:


# import numpy as np
# import pandas as pd

# pi_final = np.asarray(res["pi"], dtype=int)

# X_shuf = X1[pi_final]
# if X_shuf.ndim == 1:
#     X_shuf = X_shuf[:, None]

# df_coords = pd.DataFrame(
#     X_shuf,
#     index=spot_ids, 
#     columns=(met_names if 'met_names' in locals() and met_names is not None
#              else list(msi_norm_hvg_162.columns))
# )

# msi_norm_hvg_162_shuffled = df_coords.reindex(
#     index=msi_norm_hvg_162.index,
#     columns=msi_norm_hvg_162.columns
# )

# assert msi_norm_hvg_162_shuffled.shape == msi_norm_hvg_162.shape
# print("Done. msi_norm_hvg_162_shuffled shape:", msi_norm_hvg_162_shuffled.shape)


# In[ ]:


import numpy as np, pandas as pd, re
from pathlib import Path

def slugify(s: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s]+', "_", str(s))

assert 'X1' in locals(), "X1 is not defined"
assert 'spot_ids' in locals(), "spot_ids is not defined"
assert 'msi_norm_hvg_162' in locals(), "msi_norm_hvg_162 is not defined"

cols = list(msi_norm_hvg_162.columns) if 'msi_norm_hvg_162' in locals() else list(met_names)
assert X1.shape[1] == len(cols), f"X1 columns ={X1.shape[1]} do not match colunmn names ={len(cols)}"

N, M = X1.shape
if 'results' in locals() and isinstance(results, dict) and results:
    shuf_cols = []
    for j, met in enumerate(cols):
        rj = results.get(met, results.get(str(met)))
        if rj is None:
            raise KeyError(f"results do not have key {met!r}")
        pi_j = np.asarray(rj['pi'], dtype=int)
        assert pi_j.shape[0] == N, f"{met} 的 pi length do not match the row number of X1"
        shuf_cols.append(X1[pi_j, j][:, None])
    X_shuf = np.hstack(shuf_cols)
elif 'res' in locals() and isinstance(res, dict) and 'pi' in res:
    pi_final = np.asarray(res['pi'], dtype=int)
    assert pi_final.shape[0] == N, "pi cannot match X1"
    X_shuf = X1[pi_final, :]
else:
    raise RuntimeError("Cannot fine results")

df_coords = pd.DataFrame(X_shuf, index=spot_ids, columns=cols)
msi_norm_hvg_162_shuffled = df_coords.reindex(
    index=msi_norm_hvg_162.index,
    columns=msi_norm_hvg_162.columns
)
assert msi_norm_hvg_162_shuffled.shape == msi_norm_hvg_162.shape
print("Done. msi_norm_hvg_162_shuffled shape:", msi_norm_hvg_162_shuffled.shape)

if 'results' in locals() and isinstance(results, dict) and results:
    root_out = next(iter(results.values())).get('snapshot_root', 'results_by_metabolite_4l_n_es1')
elif 'res' in locals() and isinstance(res, dict):
    root_out = res.get('snapshot_root', 'results_by_metabolite_4l_n_es1')
else:
    root_out = 'results_by_metabolite_4l_n_es'

subdir = "metabolite"

for met in msi_norm_hvg_162_shuffled.columns:
    mdir = Path(root_out) / slugify(str(met)) / subdir
    mdir.mkdir(parents=True, exist_ok=True)
    out_path = mdir / "msi_norm_hvg_shuffled.csv"
    msi_norm_hvg_162_shuffled[[met]].to_csv(out_path, float_format="%.6g")


# In[116]:


print(msi_norm_hvg_162_shuffled)


# In[117]:


# pi_final = next((res[k] for k in ("pi", "perm", "pi_final") if k in res), None)
# assert pi_final is not None
# pi_final = np.asarray(pi_final, dtype=int)
# X_final = X1[pi_final]


# In[118]:


print(rna_norm_hvg)


# In[119]:


from scipy.spatial.distance import pdist, squareform
coords_mat = coords_df[["x", "y"]].values
D_condensed = pdist(coords_mat, metric="euclidean")
dist = squareform(D_condensed)  

l=  dmin
W = np.zeros_like(dist)
mask = dist > 0
W[mask] = np.exp(-dist[mask]**2 / (2*l**2))
np.fill_diagonal(W, 1)
W /= W.sum()

# W = np.exp(-dist**2/(2*l**2))
# np.fill_diagonal(W, 0.0)
# row = W.sum(axis=1, keepdims=True); row[row==0]=1.0
# W = W / row

X = rna_norm_hvg.values
Y = msi_norm_hvg_162_shuffled.values

X_mean = X.mean(axis=0, keepdims=True); X_stdv = X.std(axis=0, ddof=0, keepdims=True); X_stdv[X_stdv==0]=1.0
Y_mean = Y.mean(axis=0, keepdims=True); Y_stdv = Y.std(axis=0, ddof=0, keepdims=True); Y_stdv[Y_stdv==0]=1.0
X_std = (X - X_mean) / X_stdv
Y_std = (Y - Y_mean) / Y_stdv

WY = W.dot(Y_std) 

gene_cols  = rna_norm_hvg.columns.tolist()
metab_cols = msi_norm_hvg_162_shuffled.columns.tolist()

n_genes  = X_std.shape[1]
n_metabs = Y_std.shape[1]

SCI_mat = X_std.T @ WY

SCI_df_2 = pd.DataFrame(SCI_mat, index=gene_cols, columns=metab_cols)
print(SCI_df_2.shape)
print(SCI_df_2.iloc[:5, :5])
# SCI_df_2.to_csv("SCI_162_shuffle.csv", index = True)


# In[120]:


# desc = SCI_df_2['289.92275500000005'].describe()
# print(desc)
# plt.figure()
# plt.hist(SCI_df_2['289.92275500000005'], bins=50)
# plt.xlabel('SCI')
# plt.ylabel('Frequency')
# plt.title('Distribution of SCI_shuffled')
# plt.tight_layout()
# plt.show()


# In[121]:


from scipy.spatial.distance import pdist, squareform
coords_mat = coords_df[["x", "y"]].values
D_condensed = pdist(coords_mat, metric="euclidean")
dist = squareform(D_condensed)  

l= dmin
W = np.zeros_like(dist)
mask = dist > 0
W[mask] = np.exp(-dist[mask]**2 / (2*l**2))
np.fill_diagonal(W, 1)
W /= W.sum()

# W = np.exp(-dist**2/(2*l**2))
# np.fill_diagonal(W, 0.0)
# row = W.sum(axis=1, keepdims=True); row[row==0]=1.0
# W = W / row

X = rna_norm_hvg.values
Y = msi_norm_hvg_162.values

X_mean = X.mean(axis=0, keepdims=True); X_stdv = X.std(axis=0, ddof=0, keepdims=True); X_stdv[X_stdv==0]=1.0
Y_mean = Y.mean(axis=0, keepdims=True); Y_stdv = Y.std(axis=0, ddof=0, keepdims=True); Y_stdv[Y_stdv==0]=1.0
X_std = (X - X_mean) / X_stdv
Y_std = (Y - Y_mean) / Y_stdv

WY = W.dot(Y_std) 

gene_cols  = rna_norm_hvg.columns.tolist()
metab_cols = msi_norm_hvg_162.columns.tolist()

n_genes  = X_std.shape[1]
n_metabs = Y_std.shape[1]

SCI_mat = X_std.T @ WY

SCI_df_0 = pd.DataFrame(SCI_mat, index=gene_cols, columns=metab_cols)
print(SCI_df_0.iloc[:5, :5])
# SCI_df_2.to_csv("SCI_162_shuffle.csv", index = True)


# In[122]:


# desc = SCI_df_0['289.92275500000005'].describe()
# print(desc)
# plt.figure()
# plt.hist(SCI_df_0['289.92275500000005'], bins=50)
# plt.xlabel('SCI')
# plt.ylabel('Frequency')
# plt.title('Distribution of SCI')
# plt.tight_layout()
# plt.show()


# In[ ]:


import re
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def slugify(s: str) -> str:

    return re.sub(r'[\\/:*?"<>|\\s]+', "_", str(s))

def ensure_mdir_for_met(
    met_name: str,
    snapshot_root: str = "results_by_metabolite_l=4_es1",
    subdir: str = "metabolite",
    snapshot_dirs: Optional[Dict[str, str]] = None
) -> Path:

    if snapshot_dirs is not None and met_name in snapshot_dirs:
        mdir = Path(snapshot_dirs[met_name])
    else:
        mdir = Path(snapshot_root) / slugify(met_name) / subdir
    mdir.mkdir(parents=True, exist_ok=True)
    return mdir

def save_histograms_per_metabolite(
    SCI_df: pd.DataFrame,
    *,
    label: str = "SCI_shuffled",
    bins: int = 50,
    snapshot_root: str = "results_by_metabolite_l4_n_es1",
    subdir: str = "metabolite",
    snapshot_dirs: Optional[Dict[str, str]] = None
):
    for met in SCI_df.columns:
        s = pd.to_numeric(SCI_df[met], errors="coerce").dropna()
        mdir = ensure_mdir_for_met(met, snapshot_root, subdir, snapshot_dirs)

        desc_csv = mdir / f"{label}_desc.csv"
        s.describe().to_csv(desc_csv, header=False)
        out_png = mdir / f"hist_{label}.png"
        plt.figure(figsize=(6, 4))
        plt.hist(s.values, bins=bins)
        plt.xlabel("SCI"); plt.ylabel("Frequency")
        plt.title(f"Distribution of {label} — {met}")
        plt.tight_layout()
        plt.savefig(out_png, dpi=220)
        plt.close()

        print("saved:", desc_csv)
        print("saved:", out_png)



# In[ ]:


snapshot_dirs = None
if 'res' in locals() and isinstance(res, dict) and 'snapshot_dirs' in res:
    snapshot_dirs = res['snapshot_dirs']

save_histograms_per_metabolite(
    SCI_df_2, label="SCI_shuffled",
    snapshot_root="results_by_metabolite_l4_n_es1",
    subdir="metabolite",
    snapshot_dirs=snapshot_dirs
)

save_histograms_per_metabolite(
    SCI_df_0, label="SCI",
    snapshot_root="results_by_metabolite_l4_n_es1",
    subdir="metabolite",
    snapshot_dirs=snapshot_dirs
)


# In[ ]:


import re
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

def slugify(s: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s]+', "_", str(s))

def ensure_mdir_for_met(
    met_name: str,
    root: str = "results_by_metabolite_l4_n_es1",
    subdir: str = "metabolite",
    snapshot_dirs: Optional[Dict[str, str]] = None
) -> Path:

    if snapshot_dirs is not None and met_name in snapshot_dirs:
        mdir = Path(snapshot_dirs[met_name])
    else:
        mdir = Path(root) / slugify(met_name) / subdir
    mdir.mkdir(parents=True, exist_ok=True)
    return mdir

def empirical_p_and_z(null_vals: np.ndarray, obs_vals: np.ndarray):

    obs = np.asarray(obs_vals, dtype=float)
    nv  = pd.to_numeric(pd.Series(null_vals), errors="coerce").dropna().to_numpy()

    if nv.size == 0:
        F = np.full_like(obs, 0.5, dtype=float)
    else:
        arr = np.sort(nv)
        k = np.searchsorted(arr, obs, side="right")
        F = (k + 0.5) / (arr.size + 1.0)
        F = np.clip(F, 1e-12, 1-1e-12)

    p_two = 2.0 * np.minimum(F, 1.0 - F)
    z_prob = norm.ppf(F)
    return p_two, z_prob

def save_per_metabolite_combined(
    SCI_df_0: pd.DataFrame,  
    SCI_df_2: pd.DataFrame,
    *,
    root: str = "results_by_metabolite_l4_n_es1",
    subdir: str = "metabolite",
    snapshot_dirs: Optional[Dict[str,str]] = None,
    also_save_raw_cols: bool = True,
    also_save_hists: bool = False,
    bins: int = 50
):
    SCI_df_0 = SCI_df_0.copy()
    SCI_df_2 = SCI_df_2.reindex(index=SCI_df_0.index, columns=SCI_df_0.columns)

    genes = SCI_df_0.index
    for met in SCI_df_0.columns:
        s0 = pd.to_numeric(SCI_df_0[met], errors="coerce")  # 实测
        s2 = pd.to_numeric(SCI_df_2[met], errors="coerce")  # null

        mdir = ensure_mdir_for_met(met, root=root, subdir=subdir, snapshot_dirs=snapshot_dirs)

        if also_save_raw_cols:
            s0.to_frame(name=met).to_csv(mdir / "SCI_df_0.csv", index=True)
            s2.to_frame(name=met).to_csv(mdir / "SCI_df_2.csv", index=True)

        p_two, z_prob = empirical_p_and_z(s2.values, s0.values)

        combined = pd.DataFrame({
            "actual": s0.values,
            "null":   np.full_like(s0.values, s2.mean(skipna=True), dtype=float),
            "p":      p_two,
            "z":      z_prob,
        }, index=genes)
        combined.index.name = "gene"
        out_csv = mdir / "SCI_combined_actual_null_p_z.csv"
        combined.to_csv(out_csv, float_format="%.6g")

        if also_save_hists:
            plt.figure(figsize=(6,4))
            plt.hist(s0.dropna().values, bins=bins)
            plt.xlabel("SCI"); plt.ylabel("Frequency")
            plt.title(f"Distribution of SCI — {met}")
            plt.tight_layout(); plt.savefig(mdir / "hist_SCI.png", dpi=220); plt.close()

            plt.figure(figsize=(6,4))
            plt.hist(s2.dropna().values, bins=bins)
            plt.xlabel("SCI"); plt.ylabel("Frequency")
            plt.title(f"Distribution of SCI_shuffled — {met}")
            plt.tight_layout(); plt.savefig(mdir / "hist_SCI_shuffled.png", dpi=220); plt.close()

        print(f"saved: {out_csv}")


snapshot_dirs = None
if 'res' in locals() and isinstance(res, dict) and 'snapshot_dirs' in res:
    snapshot_dirs = res['snapshot_dirs']

save_per_metabolite_combined(
    SCI_df_0=SCI_df_0,
    SCI_df_2=SCI_df_2,
    root="results_by_metabolite_l4_n_es1",
    subdir="metabolite",
    snapshot_dirs=snapshot_dirs,
    also_save_raw_cols=True, 
    also_save_hists=False, 
    bins=50
)


# In[ ]:





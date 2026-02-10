import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import anndata as ad
from scipy.spatial import cKDTree
import math

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

def target_diag_M_from_matrix(G, X_mat, weight_key="w"):
    M = X_mat.shape[1]
    Mt = np.zeros(M, dtype=float)
    for u, v, data in G.edges(data=True):
        w = data.get(weight_key, 1.0)
        Mt += w * X_mat[u] * X_mat[v]
    return Mt

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

def _energy(M, M0, metric='l1'):
    d = np.abs(M - M0)
    return float(np.sum(d))

def mcmc_shuffle_diag_delta_seqshots(
    G, coords, X_mat,
    n_accept=None,
    t=None,
    t_scale=1.0,
    eps=None,
    seed=7697,
    acc_every=150,
    pre_snap_every=300,
    snapshot_metabolites=None,
    met_names=None,
    topk_for_snapshot=6,
    snapshot_root="snapshots_after_only",
    log_csv=None,
    save_initial=False,
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

    # met_dirs = {}
    # for m in snap_idx:
    #     mname = met_names[m] if met_names else f"m{m}"
    #     mdir = os.path.join(snapshot_root, slugify(mname))
    #     os.makedirs(mdir, exist_ok=True)
    #     met_dirs[m] = mdir
    #     if save_initial:
    #         init_title = f"{met_names[m] if met_names else f'm{m}'} | acc=0, seed={seed}"
    #         snapshot_after_only(coords, X_mat, pi, m,
    #                             out_path=os.path.join(mdir, "acc000000.png"),
    #                             title_info=init_title)

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

        # if pre_snap_every and (pre_done % pre_snap_every == 0):
        #     for m in snap_idx:
        #         mdir = met_dirs[m]
        #         title = f"{met_names[m] if met_names else f'm{m}'} | acc={pre_done}, seed={seed}"
        #         snapshot_after_only(coords, X_mat, pi, m,
        #                             out_path=os.path.join(mdir, f"acc{pre_done:06d}.png"),
        #                             title_info=title)

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

            # global_acc = pre_done + accepted
            # if acc_every and (global_acc % acc_every == 0):
            #     for m in snap_idx:
            #         mdir = met_dirs[m]
            #         title = f"{met_names[m] if met_names else f'm{m}'} | acc={global_acc}, seed={seed}"
            #         snapshot_after_only(coords, X_mat, pi, m,
            #                             out_path=os.path.join(mdir, f"acc{global_acc:06d}.png"),
            #                             title_info=title)

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
        "M_history": (np.stack(Ms_log, 0).flatten() if save_M_history else None),
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


def sample(
    mdata, l,
    n_accept=25000, pre_shuffle_swaps=5000,
    base_seed=10, 
    weight_key="w", energy_metric="l1",
):
    G = build_weighted_radius_graph(mdata.obsm['spatial'].values, l=l, radius=float('inf'), weight_key="w")
    X_mat = np.asarray(mdata.mod['metabolite'].X)
    rows_sampled = []
    snapshots = []
    rows_order = []
    for j, met in enumerate(mdata.mod['metabolite'].var_names):
        X_col = X_mat[:, [j]]
        seed_j = int(base_seed + j)
        res = mcmc_shuffle_diag_delta_seqshots(
            G, mdata.obsm['spatial'].values, X_col,
            n_accept=n_accept,
            pre_shuffle_swaps=pre_shuffle_swaps,
            met_names=[met],
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
        shuffled_col = np.zeros_like(X_mat[:, j])
        shuffled_col[res['pi']] = X_mat[:, j]
        rows_sampled.append(shuffled_col)
        rows_order.append(res['pi'])
        snapshots.append(res['M_history'])
        print(f"[done] {met} | seed={seed_j} | accepted={res['accepted']}")
    sampled_df = pd.DataFrame(
        rows_sampled, 
        index=mdata.mod['metabolite'].var_names,
        columns=mdata.obs_names
    ).T
    
    order_df = pd.DataFrame(
        rows_order, 
        index=mdata.mod['metabolite'].var_names,
        columns=mdata.obs_names
    ).T

    snapshots_df = pd.DataFrame(
        snapshots, 
        index=mdata.mod['metabolite'].var_names
    ).T

    mdata.mod['metabolites_sampled'] = ad.AnnData(
        X=sampled_df.values,
        obs=pd.DataFrame(index=sampled_df.index),
        var=pd.DataFrame(index=sampled_df.columns)
    )
    X_rearranged = mdata.uns["metabolite_raw"].to_df().values[mdata.uns['mcmc_order'].values, np.arange(mdata.uns['mcmc_order'].values.shape[1])]
    mdata.uns["metabolite_null"] = ad.AnnData(
            X=X_rearranged,
            obs=pd.DataFrame(index=mdata.obs_names.values),
            var=pd.DataFrame(index=mdata.mod["metabolite"].var_names)
        )
    mdata.uns['mcmc_snapshots'] = snapshots_df
    mdata.uns['mcmc_order'] = order_df
    return mdata, G

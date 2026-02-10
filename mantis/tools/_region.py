import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde
from pathlib import Path
from scipy.stats import kstest
from scipy.stats import (
    chi2, lognorm, gamma, gengamma, weibull_min, fisk, burr, betaprime, geninvgauss,
    expon, lomax, genpareto, pareto, invweibull, invgamma, invgauss, loggamma, norm 
)

def _ad_statistic(x, cdf, eps=1e-12):

    x = np.sort(np.asarray(x, float))
    n = x.size
    u = np.clip(cdf(x), eps, 1.0 - eps)  # 防 log(0)
    i = np.arange(1, n + 1)
    A2 = -n - (1.0 / n) * np.sum((2 * i - 1) * (np.log(u) + np.log(1.0 - u[::-1])))
    return float(A2)

def _ad_pvalue_naive(A2, n):

    A = A2 * (1.0 + 0.75 / n + 2.25 / (n * n))
    if A < 0.2:
        p = 1.0 - np.exp(-13.436 + 101.14 * A - 223.73 * A * A)
    elif A < 0.34:
        p = 1.0 - np.exp(-8.318 + 42.796 * A - 59.938 * A * A)
    elif A < 0.6:
        p = np.exp(0.9177 - 4.279 * A - 1.38 * A * A)
    else:
        p = np.exp(1.2937 - 5.709 * A + 0.0186 * A * A)
    return float(np.clip(p, 0.0, 1.0)), float(A)

def fit_and_AD_naive(name, dist, vals, n):
    res = {"dist": name}
    try:
        params = dist.fit(vals, floc=0.0)
        res["params"] = params

        D_obs, p_ks = kstest(vals, lambda x: dist.cdf(x, *params))
        res["KS_D_naive"] = float(D_obs)
        res["KS_p_naive"] = float(p_ks)

        A2 = _ad_statistic(vals, lambda t: dist.cdf(t, *params))
        p_ad, A2_star = _ad_pvalue_naive(A2, n)
        res["AD_A2"] = float(A2)
        res["AD_A2_star"] = float(A2_star)
        res["AD_p_naive"] = float(p_ad)
    except Exception as e:
        res["error"] = str(e)
    return res

candidates = {
    "chi2":         chi2,         # df, loc, scale
    "lognorm":      lognorm,      # s,  loc, scale
    "gamma":        gamma,        # a,  loc, scale
    "gengamma":     gengamma,     # a, c, loc, scale
    "weibull_min":  weibull_min,  # c,  loc, scale
    "fisk":         fisk,         # c,  loc, scale (log-logistic)
    "burr":         burr,         # c, d, loc, scale
    "betaprime":    betaprime,    # a, b, loc, scale
    "geninvgauss":  geninvgauss,  # p, b, loc, scale (GIG)
    "expon":        expon,        # (loc, scale)
    "lomax":        lomax,        # (c, loc, scale)
    "genpareto":    genpareto,    # (c, loc, scale)
    "pareto":       pareto,       # (b, loc, scale)
    "invweibull":   invweibull,   # (c, loc, scale)
    "invgamma":     invgamma,     # (a, loc, scale)
    "invgauss":     invgauss,     # (mu, loc, scale)
    "loggamma":     loggamma,     # (c, loc, scale)
    "norm":         norm          # (loc, scale)
}

def _bh(pseries):
    p = pseries.to_numpy(dtype=float)
    mask = np.isfinite(p)
    q = np.full_like(p, np.nan, dtype=float)
    if mask.sum() > 0:
        psub = p[mask]
        order = np.argsort(psub)
        ranks = np.arange(1, len(psub)+1)
        qsub = psub[order] * len(psub) / ranks
        qsub = np.minimum.accumulate(qsub[::-1])[::-1]
        qsub = np.clip(qsub, 0, 1)
        tmp = np.empty_like(psub)
        tmp[order] = qsub
        q[mask] = tmp
    return pd.Series(q, index=pseries.index)

def _to_numeric_1d(obj):
    if isinstance(obj, pd.Series):
        arr = pd.to_numeric(obj, errors="coerce").to_numpy()
    elif isinstance(obj, pd.DataFrame):
        num = obj.select_dtypes(include=[np.number])
        arr = pd.to_numeric(num.to_numpy().ravel(), errors="coerce")
    else:
        arr = np.asarray(obj).ravel()
    arr = arr[np.isfinite(arr)]
    return arr

def _to_series(x):
    if isinstance(x, pd.Series):
        s = x.dropna().astype(float)
        return s
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            s = x.iloc[:,0].dropna().astype(float)
            return s
        raise ValueError("KL / KL_null DataFrame must be single-column for this function.")
    arr = np.asarray(x).ravel()
    return pd.Series(arr[np.isfinite(arr)], index=np.arange(len(arr))[~np.isnan(arr)].tolist()).astype(float)


def _get_KL(X_df, region_df, eps = 1e-12):
    mean_by_region = X_df.groupby(region_df, observed=False).mean()
    den = mean_by_region.sum(axis=0).replace(0, np.nan)
    prob = (mean_by_region / den).T.fillna(0.0)
    prob.index.name = "metabolite"
    prob.columns.name = "region"
    bgp = prob.mean(axis=0)
    bgp = bgp / bgp.sum()  

    P = prob.copy().clip(lower=0)
    P = P.div(P.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    q = bgp.reindex(P.columns).astype(float).clip(lower=0)
    q = q / q.sum()

    LR = np.log(P + eps).sub(np.log(q + eps), axis=1) 
    C  = P * LR 

    KL = C.sum(axis=1).rename("KL(p||q)")

    C.index.name = "metabolite"; C.columns.name = "region"
    LR.index.name = "metabolite"; LR.columns.name = "region"
    KL.index.name = "metabolite"
    return KL, C, LR

def regional_met(mdata, alpha=0.05, two_sided = False, clip_min=1e-15):
    X_df = mdata.uns['metabolite_raw'].to_df()
    region_df = mdata.obsm['region']
    KL, C, LR = _get_KL(X_df, region_df)
    X_df = mdata.uns['metabolite_null'].to_df()
    KL_null, C_null, LR_null = _get_KL(X_df, region_df)

    vals_real = _to_numeric_1d(KL)
    vals_null = _to_numeric_1d(KL_null)
    vals = np.asarray(vals_null, float)
    vals = vals[np.isfinite(vals)]
    vals = np.clip(vals, 1e-15, np.inf)
    n = vals.size
    rows = [fit_and_AD_naive(name, dist, vals, n) for name, dist in candidates.items()]
    fit_table = (
        pd.DataFrame(rows)
        .sort_values(by=["KS_p_naive"], ascending=False)
        .reset_index(drop=True)
    )
    best_fit = fit_table.iloc[0]['dist']
    s_null = _to_series(KL_null)
    s_real = _to_series(KL)
    n_before = s_null.size
    s_null_clipped = s_null.copy()
    if (s_null_clipped <= 0).any():
        s_null_clipped = s_null_clipped.clip(lower=clip_min)
    n_after = s_null_clipped.size
    params = candidates[best_fit].fit(s_null_clipped.values, floc=0.0)
    print(f"Best fit distribution for KL null: {best_fit} with params {params}")

    obs = s_real.copy()
    p_right = pd.Series(index=obs.index, dtype=float)
    p_two = pd.Series(index=obs.index, dtype=float)
    for idx, val in obs.items():
        if not np.isfinite(val):
            p_right.loc[idx] = np.nan
            p_two.loc[idx] = np.nan
            continue
        x = max(val, clip_min)
        try:
            logsf = candidates[best_fit].logsf(x, *params)
            logsf_clipped = np.maximum(logsf, np.log(np.finfo(float).tiny))
            pv = float(np.exp(logsf_clipped))
            p_right.loc[idx] = pv
            if two_sided:
                cdfv = candidates[best_fit].cdf(x, *params)
                pv2 = 2.0 * min(cdfv, 1.0 - cdfv)
                p_two.loc[idx] = float(pv2)
            else:
                p_two.loc[idx] = np.nan
        except Exception as e:
            p_right.loc[idx] = np.nan
            p_two.loc[idx] = np.nan

    q_right = _bh(p_right)
    q_two = None
    if two_sided:
        q_two = _bh(p_two)

    df = pd.DataFrame({
        "obs": obs,
        "p_right": p_right,
        "q_right": q_right
    }, index=obs.index)
    if two_sided:
        df["p_two_sided"] = p_two
        df["q_two_sided"] = q_two


    df["significant_right"] = df["q_right"] < float(alpha)
    total = df.shape[0]
    n_sig = int(df["significant_right"].sum())
    prop = n_sig / total if total>0 else np.nan

    print(f"Parametric significant (BH q < {alpha}) = {n_sig} ({prop:.2%})")
    df_output = pd.DataFrame(C.values, index=C.index, columns=C.columns)
    df_output.index.name = None
    df_output.columns.name = None
    df_output['p_value'] = df['p_right']
    df_output['q_value'] = df['q_right']
    df_output['KL'] = KL
    new_order = ['KL', 'p_value', 'q_value'] + [c for c in df_output.columns.tolist() if c not in ['KL', 'p_value', 'q_value']]
    df_output = df_output[new_order]
    mdata.uns['regional_met'] = df_output
    fit_table['params'] = fit_table['params'].astype(str)
    mdata.uns['regional_met_fit'] = {
        "best_fit": best_fit,
        "best_fit_params": list(params),
        "fit_table": fit_table
    }
    return mdata
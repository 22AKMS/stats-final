import argparse
import math
import pandas as pd
import numpy as np
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep


def pick_column(df, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    for c in df.columns:
        cl = c.lower()
        for name in candidates:
            if name.lower() == cl:
                return c
    return None


def normalize_label(series):
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(int)
    sl = s.astype(str).str.strip().str.lower()
    mapping = {
        "0": 0,
        "1": 1,
        "phishing": 0,
        "phish": 0,
        "malicious": 0,
        "legitimate": 1,
        "legit": 1,
        "benign": 1,
    }
    out = sl.map(mapping)
    if out.isna().any():
        raise ValueError(
            "Could not normalize label column to 0/1. Please ensure labels are 0/1 or phishing/legitimate."
        )
    return out.astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    df = pd.read_csv(args.csv_path)

    label_col = pick_column(
        df, ["label", "Label", "CLASS_LABEL", "class", "Class", "y", "Y"]
    )
    url_len_col = pick_column(df, ["URLLength", "urlLength", "url_length", "urllength"])
    https_col = pick_column(
        df, ["IsHTTPS", "ishttps", "HTTPS", "https", "https_flag", "HTTPSFlag"]
    )

    if label_col is None or url_len_col is None or https_col is None:
        raise ValueError(f"Missing required columns. Found columns: {list(df.columns)}")

    df = df.drop_duplicates()
    df = df.dropna(subset=[label_col, url_len_col, https_col]).copy()

    df[label_col] = normalize_label(df[label_col])
    df[url_len_col] = pd.to_numeric(df[url_len_col], errors="coerce")
    df[https_col] = pd.to_numeric(df[https_col], errors="coerce")

    df = df.dropna(subset=[url_len_col, https_col]).copy()
    df[https_col] = (df[https_col] > 0).astype(int)

    phish = df[df[label_col] == 0].copy()
    legit = df[df[label_col] == 1].copy()

    if phish.empty or legit.empty:
        raise ValueError(
            "One of the groups is empty after cleaning. Check label coding and filters."
        )

    def desc_len(x):
        x = x.to_numpy()
        return {
            "n": int(x.size),
            "mean": float(np.mean(x)),
            "sd": float(np.std(x, ddof=1)) if x.size > 1 else float("nan"),
            "median": float(np.median(x)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
        }

    ph_len = desc_len(phish[url_len_col].astype(float))
    lg_len = desc_len(legit[url_len_col].astype(float))

    ph_https_n = int(phish[https_col].sum())
    lg_https_n = int(legit[https_col].sum())
    ph_n = int(len(phish))
    lg_n = int(len(legit))
    ph_rate = ph_https_n / ph_n
    lg_rate = lg_https_n / lg_n

    print(f"N_total={len(df)}")
    print(f"n_phishing={ph_n} n_legitimate={lg_n}")
    print(
        f"URLLength(phishing): M={ph_len['mean']:.4f} SD={ph_len['sd']:.4f} Median={ph_len['median']:.0f} Min={ph_len['min']:.0f} Max={ph_len['max']:.0f}"
    )
    print(
        f"URLLength(legitimate): M={lg_len['mean']:.4f} SD={lg_len['sd']:.4f} Median={lg_len['median']:.0f} Min={lg_len['min']:.0f} Max={lg_len['max']:.0f}"
    )
    print(f"IsHTTPS(phishing): {ph_https_n}/{ph_n} p̂={ph_rate:.4f}")
    print(f"IsHTTPS(legitimate): {lg_https_n}/{lg_n} p̂={lg_rate:.4f}")
    print()

    alpha = args.alpha

    print("RQ1 Hypotheses:")
    print("H0: mu_phish - mu_legit = 0")
    print("HA: mu_phish - mu_legit != 0")
    d1 = DescrStatsW(phish[url_len_col].astype(float).to_numpy())
    d2 = DescrStatsW(legit[url_len_col].astype(float).to_numpy())
    cm = CompareMeans(d1, d2)
    t_stat, p_val, dfree = cm.ttest_ind(usevar="pooled")
    ci_low, ci_high = cm.tconfint_diff(alpha=alpha, usevar="pooled")
    mean_diff = d1.mean - d2.mean
    decision = "Reject H0" if p_val < alpha else "Fail to reject H0"
    print(f"Mean difference (phish - legit) = {mean_diff:.6f}")
    print(f"{int((1 - alpha) * 100)}% CI = [{ci_low:.6f}, {ci_high:.6f}]")
    print(f"t({int(round(dfree))}) = {t_stat:.6f}, p = {p_val:.6g} (two-sided)")
    print(f"Decision at alpha={alpha}: {decision}")
    print()

    print("RQ2 Hypotheses:")
    print("H0: p_phish - p_legit = 0")
    print("HA: p_phish - p_legit != 0")
    count = np.array([ph_https_n, lg_https_n], dtype=int)
    nobs = np.array([ph_n, lg_n], dtype=int)
    z_stat, p_val2 = proportions_ztest(
        count=count, nobs=nobs, alternative="two-sided", prop_var=True
    )
    ci2_low, ci2_high = confint_proportions_2indep(
        count1=ph_https_n,
        nobs1=ph_n,
        count2=lg_https_n,
        nobs2=lg_n,
        compare="diff",
        method="wald",
        alpha=alpha,
    )
    prop_diff = ph_rate - lg_rate
    decision2 = "Reject H0" if p_val2 < alpha else "Fail to reject H0"
    print(f"Proportion difference (phish - legit) = {prop_diff:.6f}")
    print(f"{int((1 - alpha) * 100)}% CI = [{ci2_low:.6f}, {ci2_high:.6f}]")
    print(f"z = {z_stat:.6f}, p = {p_val2:.6g} (two-sided)")
    print(f"Decision at alpha={alpha}: {decision2}")


if __name__ == "__main__":
    main()

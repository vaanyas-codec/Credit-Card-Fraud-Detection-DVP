"""
Leakage and temporal-validation experiments for the paper.

Part 1  Leakage quantification: the correct pipeline against deliberately leaky ones.
        The leaky scenarios are a methodological demonstration only; their scores are
        invalid and must never be reported as model performance.
Part 2  Temporal validation: train on an earlier period, test on a later one.

Used by run_full_experiments.py, reproduce_tables.py and make_figures.py.
Install the dependencies first: pip install -r requirements.txt
"""
import sys, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
QUICK = False
SEEDS = [42] if QUICK else [42, 1, 2, 3, 4]
VCOLS = [f"V{i}" for i in range(1, 29)]


def models(seed):
    return {
        "LR": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        "DT": DecisionTreeClassifier(class_weight="balanced", random_state=seed),
        "RF": RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=-1),
    }


def add_time(df):
    df = df.copy()
    df["Hour"] = (df["Time"] % 86400) // 3600
    df["Day"] = df["Time"] // 86400
    return df


# ---------- preprocessing: fit on one frame, apply to any frame ----------
def prep_fit(fit_df, cap=True, tcols=("Hour", "Day")):
    p = {"tcols": list(tcols)}
    amt = fit_df["Amount"]
    if cap:
        q1, q3 = amt.quantile([0.25, 0.75])
        p["lo"], p["hi"] = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        amt = amt.clip(p["lo"], p["hi"])
    p["s_amt"] = StandardScaler().fit(amt.to_frame())
    if tcols:
        p["s_t"] = StandardScaler().fit(fit_df[list(tcols)])
    return p


def prep_apply(df, p):
    X = df[VCOLS].copy()
    amt = df["Amount"]
    if "hi" in p:
        amt = amt.clip(p["lo"], p["hi"])
    X["Amount"] = p["s_amt"].transform(amt.to_frame()).ravel()
    if p["tcols"]:
        X[p["tcols"]] = p["s_t"].transform(df[p["tcols"]])
    return X


def metrics(model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {"AUPRC": average_precision_score(yte, proba),
            "F1": f1_score(yte, pred, zero_division=0),
            "MCC": matthews_corrcoef(yte, pred),
            "Precision": precision_score(yte, pred, zero_division=0),
            "Recall": recall_score(yte, pred, zero_division=0),
            "n_test": len(yte), "n_test_fraud": int(yte.sum())}


def top_k_by_separability(X, y, k=10):
    sep = {c: abs(2 * roc_auc_score(y, X[c]) - 1) for c in X.columns}
    return sorted(sep, key=sep.get, reverse=True)[:k]


# ======================= PART 1: leakage =======================
def part1(df):
    rows = []
    for seed in SEEDS:
        tr_i, te_i = train_test_split(np.arange(len(df)), test_size=0.2,
                                      stratify=df["Class"], random_state=seed)
        tr, te = df.iloc[tr_i], df.iloc[te_i]
        ytr, yte = tr["Class"], te["Class"]

        # A. CORRECT: everything fitted on train only
        p = prep_fit(tr)
        XA_tr, XA_te = prep_apply(tr, p), prep_apply(te, p)
        for name, m in models(seed).items():
            rows.append(dict(seed=seed, scenario="A. correct (fit on train only)", model=name,
                             **metrics(m, XA_tr, ytr, XA_te, yte)))

        # B. LEAKY scaling/capping: bounds + scalers fitted on ALL data
        pl = prep_fit(df)
        XB_tr, XB_te = prep_apply(tr, pl), prep_apply(te, pl)
        for name, m in models(seed).items():
            rows.append(dict(seed=seed, scenario="B. LEAKY: scaler/cap fitted before split", model=name,
                             **metrics(m, XB_tr, ytr, XB_te, yte)))

        # C. LEAKY feature selection: top-10 chosen using labels of ALL data
        cols_leaky = top_k_by_separability(prep_apply(df, pl), df["Class"])
        cols_ok = top_k_by_separability(XA_tr, ytr)
        for tag, cols in [("C0. correct feature selection (train only)", cols_ok),
                          ("C1. LEAKY: feature selection on all labels", cols_leaky)]:
            for name, m in models(seed).items():
                rows.append(dict(seed=seed, scenario=tag, model=name,
                                 **metrics(m, XA_tr[cols], ytr, XA_te[cols], yte)))

        # D. LEAKY SMOTE before split (the common published error)
        X_all = prep_apply(df, pl)
        Xs, ys = SMOTE(random_state=seed).fit_resample(X_all, df["Class"])
        Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(Xs, ys, test_size=0.2,
                                                       stratify=ys, random_state=seed)
        for name, m in models(seed).items():
            if QUICK and name != "RF":
                continue
            rows.append(dict(seed=seed, scenario="D. LEAKY: SMOTE before split (synthetic test set)",
                             model=name, **metrics(m, Xs_tr, ys_tr, Xs_te, ys_te)))

        # E. CORRECT SMOTE: resample the training split only, test on untouched data
        Xe, ye = SMOTE(random_state=seed).fit_resample(XA_tr, ytr)
        for name, m in models(seed).items():
            if QUICK and name != "RF":
                continue
            rows.append(dict(seed=seed, scenario="E. correct SMOTE (train only, real test)",
                             model=name, **metrics(m, Xe, ye, XA_te, yte)))
        print("part 1 seed done:", seed, flush=True)
    return pd.DataFrame(rows)


# ======================= PART 2: temporal =======================
def part2(df):
    df = df.sort_values("Time").reset_index(drop=True)
    splits = {
        "T1: train day 1 (<24 h) -> test day 2": df["Time"] < 86400,
        "T2: train first 36 h -> test last 12 h": df["Time"] < 129600,
    }
    rows = []
    for label, train_mask in splits.items():
        tr, te = df[train_mask], df[~train_mask]
        for tname, tcols in [("no time features", ()), ("Hour only", ("Hour",))]:
            p = prep_fit(tr, tcols=tcols)
            Xtr, Xte = prep_apply(tr, p), prep_apply(te, p)
            for name, m in models(42).items():
                r = metrics(m, Xtr, tr["Class"], Xte, te["Class"])
                rows.append(dict(split=label, features=tname, model=name,
                                 n_train=len(tr), n_train_fraud=int(tr["Class"].sum()), **r))
        # random stratified split of the SAME train fraction, for direct comparison
        frac = len(tr) / len(df)
        a, b = train_test_split(df, train_size=frac, stratify=df["Class"], random_state=42)
        p = prep_fit(a, tcols=())
        Xa, Xb = prep_apply(a, p), prep_apply(b, p)
        for name, m in models(42).items():
            r = metrics(m, Xa, a["Class"], Xb, b["Class"])
            rows.append(dict(split=label.split(":")[0] + " reference: random split, same train size",
                             features="no time features", model=name,
                             n_train=len(a), n_train_fraud=int(a["Class"].sum()), **r))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = add_time(pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "creditcard.csv"))
    pd.set_option("display.width", 200, "display.max_columns", 30)

    r1 = part1(df)
    r1.to_csv("leakage_raw.csv", index=False)
    summ = (r1.groupby(["scenario", "model"])[["AUPRC", "F1", "MCC", "Precision", "Recall"]]
              .agg(["mean", "std"]).round(3))
    summ.to_csv("leakage_summary.csv")
    print("\n=== PART 1: leakage (mean/std over seeds) ===\n", summ)

    r2 = part2(df)
    r2.to_csv("temporal_results.csv", index=False)
    print("\n=== PART 2: temporal validation ===\n",
          r2.round(3).to_string(index=False))

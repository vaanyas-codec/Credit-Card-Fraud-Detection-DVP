"""
Reproduces Tables II, III and IV of the paper (hold-out, 5-fold CV, ablations).

Protocol (as in the paper): 80/20 stratified split with seed 42; 5-fold stratified CV
(shuffled, seed 42) on all rows; every fitted quantity (IQR bounds, scalers) is fitted
on the training part only and refitted inside every fold. Default model settings,
class_weight='balanced', threshold 0.5.

Usage (needs leakage_and_temporal_experiments.py in the same folder):
    python reproduce_tables.py creditcard.csv
Writes: table2_holdout.csv, table3_cv.csv, table4_ablations.csv, reproduction_report.txt
The report compares every number with the values printed in the paper.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (average_precision_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score,
                             accuracy_score, confusion_matrix)
import leakage_and_temporal_experiments as exp

SEED = 42
MODELS = ["LR", "DT", "RF"]

# Values printed in the paper (used only for the comparison report).
PAPER_T2 = {  # prec, recall, F1, AUPRC, MCC, FP, FN
    "LR": (0.056, 0.918, 0.106, 0.721, 0.224, 1510, 8),
    "DT": (0.791, 0.735, 0.762, 0.582, 0.762, 19, 26),
    "RF": (0.949, 0.765, 0.847, 0.862, 0.852, 4, 23)}
PAPER_T3 = {  # metric -> {model: (mean, sd)}
    "Precision": {"LR": (0.062, 0.004), "DT": (0.786, 0.058), "RF": (0.960, 0.034)},
    "Recall":    {"LR": (0.913, 0.030), "DT": (0.748, 0.019), "RF": (0.758, 0.013)},
    "F1":        {"LR": (0.116, 0.007), "DT": (0.766, 0.027), "RF": (0.847, 0.018)},
    "AUPRC":     {"LR": (0.735, 0.028), "DT": (0.588, 0.043), "RF": (0.852, 0.030)},
    "MCC":       {"LR": (0.234, 0.009), "DT": (0.766, 0.028), "RF": (0.853, 0.019)}}
PAPER_T4_HOLD = {"LR": (0.721, 0.722, 0.711), "DT": (0.582, 0.563, 0.521), "RF": (0.862, 0.860, 0.861)}
PAPER_T4_CV = {"LR": (0.735, 0.732), "DT": (0.588, 0.582), "RF": (0.852, 0.851)}


def prep(train, test, cap=True, time=True):
    p = exp.prep_fit(train, cap=cap, tcols=("Hour", "Day") if time else ())
    return exp.prep_apply(train, p), exp.prep_apply(test, p)


def all_metrics(model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    return {"Precision": precision_score(yte, pred, zero_division=0),
            "Recall": recall_score(yte, pred), "F1": f1_score(yte, pred),
            "AUPRC": average_precision_score(yte, proba), "MCC": matthews_corrcoef(yte, pred),
            "ROC_AUC": roc_auc_score(yte, proba), "Accuracy": accuracy_score(yte, pred),
            "FP": int(fp), "FN": int(fn)}


def holdout(df, cap=True, time=True):
    tr, te = train_test_split(df, test_size=0.2, stratify=df["Class"], random_state=SEED)
    Xtr, Xte = prep(tr, te, cap, time)
    return {n: all_metrics(m, Xtr, tr["Class"], Xte, te["Class"])
            for n, m in exp.models(SEED).items()}


def cv(df, cap=True):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = {n: [] for n in MODELS}
    for tr_i, te_i in skf.split(df, df["Class"]):
        tr, te = df.iloc[tr_i], df.iloc[te_i]
        Xtr, Xte = prep(tr, te, cap, True)          # refit inside every fold
        for n, m in exp.models(SEED).items():
            rows[n].append(all_metrics(m, Xtr, tr["Class"], Xte, te["Class"]))
        print("  fold done", flush=True)
    return {n: pd.DataFrame(r) for n, r in rows.items()}


def main(path):
    df = exp.add_time(pd.read_csv(path))
    rep = []
    log = lambda s: (print(s, flush=True), rep.append(s))

    print("Hold-out (cap, time) ...", flush=True)
    h_cap = holdout(df, True, True)
    t2 = pd.DataFrame(h_cap).T[["Precision", "Recall", "F1", "AUPRC", "MCC", "ROC_AUC", "Accuracy", "FP", "FN"]]
    t2.round(3).to_csv("table2_holdout.csv")
    log("TABLE II  hold-out (seed 42)\n" + t2.round(3).to_string())
    for n in MODELS:
        p = PAPER_T2[n]; r = h_cap[n]
        mine = (r["Precision"], r["Recall"], r["F1"], r["AUPRC"], r["MCC"], r["FP"], r["FN"])
        bad = [f"{k}: paper {a} vs now {round(b,3)}" for k, a, b in
               zip(["Prec", "Rec", "F1", "AUPRC", "MCC", "FP", "FN"], p, mine)
               if abs(a - b) > (0.0006 if isinstance(a, float) else 0.5)]
        log(f"  {n}: " + ("matches paper" if not bad else "DIFFERS -> " + "; ".join(bad)))

    print("Hold-out (no cap) ...", flush=True)
    h_nocap = holdout(df, False, True)
    print("Hold-out (no time) ...", flush=True)
    h_notime = holdout(df, True, False)

    print("5-fold CV (cap) ...", flush=True)
    cv_cap = cv(df, True)
    print("5-fold CV (no cap) ...", flush=True)
    cv_nocap = cv(df, False)

    t3 = pd.DataFrame({n: {k: f"{d[k].mean():.3f} ± {d[k].std(ddof=1):.3f}"
                           for k in ["Precision", "Recall", "F1", "AUPRC", "MCC"]}
                       for n, d in cv_cap.items()})
    t3.to_csv("table3_cv.csv")
    log("\nTABLE III  5-fold CV (mean ± sample SD)\n" + t3.to_string())
    for k, d in PAPER_T3.items():
        for n in MODELS:
            m, s = cv_cap[n][k].mean(), cv_cap[n][k].std(ddof=1)
            ok = abs(m - d[n][0]) < 0.0006 and abs(s - d[n][1]) < 0.0006
            log(f"  {k:9s} {n}: paper {d[n][0]:.3f}±{d[n][1]:.3f}  now {m:.3f}±{s:.3f}  "
                + ("OK" if ok else "DIFFERS"))

    t4 = pd.DataFrame({
        "Hold-out cap / no cap / no time": {n: f"{h_cap[n]['AUPRC']:.3f} / {h_nocap[n]['AUPRC']:.3f} / {h_notime[n]['AUPRC']:.3f}" for n in MODELS},
        "5-fold CV cap / no cap": {n: f"{cv_cap[n]['AUPRC'].mean():.3f} / {cv_nocap[n]['AUPRC'].mean():.3f}" for n in MODELS}})
    t4.to_csv("table4_ablations.csv")
    log("\nTABLE IV  AUPRC under preprocessing variants\n" + t4.to_string())
    for n in MODELS:
        mine_h = (h_cap[n]["AUPRC"], h_nocap[n]["AUPRC"], h_notime[n]["AUPRC"])
        mine_c = (cv_cap[n]["AUPRC"].mean(), cv_nocap[n]["AUPRC"].mean())
        ok_h = all(abs(a - b) < 0.0006 for a, b in zip(PAPER_T4_HOLD[n], mine_h))
        ok_c = all(abs(a - b) < 0.0006 for a, b in zip(PAPER_T4_CV[n], mine_c))
        log(f"  {n}: hold-out {'OK' if ok_h else 'DIFFERS'}, CV {'OK' if ok_c else 'DIFFERS'}")
    open("reproduction_report.txt", "w").write("\n".join(rep))
    print("\nSaved table2_holdout.csv, table3_cv.csv, table4_ablations.csv, reproduction_report.txt")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "creditcard.csv")

"""
Regenerates Figures 1-4 of the paper from creditcard.csv (Fig. 5 is a screenshot of the
Streamlit dashboard and cannot be generated here).

Usage (needs leakage_and_temporal_experiments.py in the same folder):
    python make_figures.py creditcard.csv
Writes fig1_pipeline.png, fig2_exploratory.png, fig3_separability.png, fig4_pr_curves.png (300 dpi)
and prints the numbers quoted in the paper's text so you can check them.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import leakage_and_temporal_experiments as exp

BLUE, RED, GREY = "#4C78A8", "#E45756", "#888888"
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False})


def split(df):
    return train_test_split(df, test_size=0.2, stratify=df["Class"], random_state=42)


def fig1(n, nf, tr, te):
    fig, ax = plt.subplots(figsize=(6.5, 5.6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)

    def box(x, y, w, h, text, fc="white"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.2",
                                    fc=fc, ec="black", lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", lw=1))

    box(2.2, 8.9, 5.6, 0.9, f"creditcard.csv\n{n:,} transactions, {nf} fraud")
    arrow(5, 8.9, 5, 8.3)
    box(2.2, 7.4, 5.6, 0.8, "Stratified 80/20 split (seed 42)")
    arrow(3.6, 7.4, 2.6, 6.7); arrow(6.4, 7.4, 7.4, 6.7)
    box(0.3, 5.6, 4.4, 1.0, f"Training split\n{len(tr):,} transactions ({int(tr.Class.sum())} fraud)")
    box(5.3, 5.6, 4.4, 1.0, f"Test split\n{len(te):,} transactions ({int(te.Class.sum())} fraud)")
    arrow(2.5, 5.6, 2.5, 5.0); arrow(7.5, 5.6, 7.5, 5.0)
    box(0.3, 3.8, 4.4, 1.2, "Fit on train only:\nIQR bounds, scalers,\nHour/Day from Time", fc="#DDE8F5")
    box(5.3, 3.8, 4.4, 1.2, "Apply frozen\nbounds and scalers", fc="#DDE8F5")
    arrow(4.7, 4.4, 5.3, 4.4)
    arrow(2.5, 3.8, 2.5, 3.2); arrow(7.5, 3.8, 7.5, 3.2)
    box(0.3, 2.2, 4.4, 1.0, "Train LR, DT, RF\n(balanced weights)")
    box(5.3, 2.2, 4.4, 1.0, "Score trained models\non test split")
    arrow(2.5, 2.2, 4.0, 1.5); arrow(7.5, 2.2, 6.0, 1.5)
    box(1.7, 0.5, 6.6, 1.0, "Hold-out and 5-fold stratified CV\nprecision, recall, F1, AUPRC, MCC")
    fig.savefig("fig1_pipeline.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def fig2(tr, te, upper):
    fig, axes = plt.subplots(3, 1, figsize=(6, 9))
    # (a) class balance
    ax = axes[0]
    vals = {"Train": tr.Class.value_counts(), "Test": te.Class.value_counts()}
    x = np.arange(2); w = 0.35
    nb = [vals[k][0] for k in vals]; fb = [vals[k][1] for k in vals]
    b1 = ax.bar(x - w / 2, nb, w, color=BLUE, label="Normal"); b2 = ax.bar(x + w / 2, fb, w, color=RED, label="Fraud")
    ax.set_yscale("log"); ax.set_ylim(10, 1e6); ax.set_xticks(x); ax.set_xticklabels(list(vals))
    ax.set_ylabel("Transactions (log)"); ax.legend(frameon=True, loc="upper right")
    for b, v in list(zip(b1, nb)) + list(zip(b2, fb)):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.set_title(f"(a) Class balance ({100 * tr.Class.mean():.2f}% fraud in training, "
                 f"{100 * te.Class.mean():.2f}% in test)", fontsize=9)
    # (b) amount by class (raw, non-zero, log axis)
    ax = axes[1]
    pos = tr[tr.Amount > 0]
    bins = np.logspace(np.log10(pos.Amount.min()), np.log10(pos.Amount.max()), 40)
    for c, col, lab in [(0, BLUE, "Normal"), (1, RED, "Fraud")]:
        a = pos[pos.Class == c].Amount
        ax.hist(a, bins=bins, weights=np.ones(len(a)) / len(a), color=col, alpha=0.55, label=lab)
    ax.set_xscale("log"); ax.axvline(upper, color="black", ls="--", lw=1)
    ax.text(upper * 1.1, ax.get_ylim()[1] * 0.9, f"IQR cap\n{upper:.0f}", fontsize=8)
    ax.set_xlabel("Amount, non-zero (log scale)"); ax.set_ylabel("Share of class"); ax.legend()
    ax.set_title("(b) Amount, raw (training split)", fontsize=9)
    # (c) fraud rate per elapsed hour
    ax = axes[2]
    h = (tr.Time // 3600).astype(int)
    g = tr.groupby(h).Class.agg(["sum", "count"]); rate = 100 * g["sum"] / g["count"]
    colors = [RED if r > 0.8 else "#9ECAE1" for r in rate]
    ax.bar(rate.index, rate.values, color=colors, width=0.85)
    ax.axvline(23.5, color="black", ls=":", lw=1)
    ax.text(11, ax.get_ylim()[1] * 0.93, "cycle 1", ha="center", fontsize=8)
    ax.text(35, ax.get_ylim()[1] * 0.93, "cycle 2", ha="center", fontsize=8)
    ax.set_xlabel("Hours elapsed since first transaction"); ax.set_ylabel("Fraud rate (%)")
    ax.set_title("(c) Fraud rate per elapsed hour (training split); bars above 0.8% highlighted", fontsize=9)
    fig.tight_layout(); fig.savefig("fig2_exploratory.png", dpi=300); plt.close(fig)
    # numbers quoted in the text
    hr = (tr.Time % 86400 // 3600).astype(int)
    pooled = tr[hr == 2].Class.agg(["sum", "count"])
    print(f"[check] median amount fraud {tr[tr.Class==1].Amount.median():.2f} / normal {tr[tr.Class==0].Amount.median():.2f}"
          f" (paper 9.17 / 22.00)")
    print(f"[check] cap clips {100*(tr.Amount>upper).mean():.1f}% of rows, {100*(tr[tr.Class==1].Amount>upper).mean():.1f}% of fraud (paper 11.2 / 18.8)")
    print(f"[check] hour index 2 pooled: {int(pooled['sum'])} of {int(pooled['count'])} = {100*pooled['sum']/pooled['count']:.2f}% (paper 48 of 2,666 = 1.80%)")


def fig3(tr):
    feats = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    sep = pd.Series({f: abs(2 * roc_auc_score(tr.Class, tr[f]) - 1) for f in feats}).sort_values(ascending=False)
    top = sep.head(12); hi = ["V14", "V4", "V12", "V10"]
    fig, axes = plt.subplots(2, 1, figsize=(6, 7.5), gridspec_kw={"height_ratios": [1.5, 1]})
    ax = axes[0]
    ax.barh(top.index[::-1], top.values[::-1], color=[RED if f in hi else BLUE for f in top.index[::-1]])
    ax.set_xlabel("Univariate separability |2·AUC − 1|")
    rank = list(sep.index).index("Amount") + 1
    ax.text(0.98, 0.04, f"Amount ranks {rank} of 29", transform=ax.transAxes, ha="right", fontsize=8)
    ax.set_title("(a) Top 12 features by separability (training split)", fontsize=9)
    sub = axes[1]; sub.axis("off")
    for i, f in enumerate(hi):
        a = fig.add_axes([0.10 + i * 0.225, 0.08, 0.13, 0.27])
        a.boxplot([tr[tr.Class == 0][f], tr[tr.Class == 1][f]], showfliers=False,
                  medianprops=dict(color="darkorange"))
        a.set_xticklabels(["N", "F"]); a.set_xlabel(f)
    fig.text(0.5, 0.375, "(b) Distribution by class (N = normal, F = fraud; outliers hidden)", ha="center", fontsize=9)
    fig.savefig("fig3_separability.png", dpi=300); plt.close(fig)
    print("[check] top features:", ", ".join(f"{k} {v:.3f}" for k, v in sep.head(6).items()),
          f"| Amount {sep['Amount']:.3f}, rank {rank} (paper: V14 0.896, V4 0.884, V12 0.865, V10 0.831, V11 0.825, V3 0.824; Amount 0.116, rank 23)")
    print(f"[check] median V14 fraud {tr[tr.Class==1].V14.median():.2f} / normal {tr[tr.Class==0].V14.median():.2f} (paper -6.70 / 0.05)")


def fig4(tr, te):
    p = exp.prep_fit(tr); Xtr, Xte = exp.prep_apply(tr, p), exp.prep_apply(te, p)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for (name, m), col, lab in zip(exp.models(42).items(), ["#D62728", "#FFBF00", "#4C78A8"],
                                   ["Logistic Regression", "Decision Tree", "Random Forest"]):
        m.fit(Xtr, tr.Class); pr = m.predict_proba(Xte)[:, 1]
        P, R, _ = precision_recall_curve(te.Class, pr)
        ax.plot(R, P, color=col, label=f"{lab} (AUPRC {average_precision_score(te.Class, pr):.3f})")
    base = te.Class.mean()
    ax.axhline(base, color=GREY, ls=":", lw=1); ax.text(0.02, base + 0.015, f"no-skill ({base:.4f})", color=GREY, fontsize=8)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", bbox_to_anchor=(0.1, 0.12))
    fig.tight_layout(); fig.savefig("fig4_pr_curves.png", dpi=300); plt.close(fig)


if __name__ == "__main__":
    df = exp.add_time(pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "creditcard.csv"))
    tr, te = split(df)
    q1, q3 = tr.Amount.quantile([.25, .75]); upper = q3 + 1.5 * (q3 - q1)
    print(f"[check] IQR upper bound {upper:.2f} (paper 185.27)")
    fig1(len(df), int(df.Class.sum()), tr, te); print("fig1 done", flush=True)
    fig2(tr, te, upper); print("fig2 done", flush=True)
    fig3(tr); print("fig3 done", flush=True)
    fig4(tr, te); print("fig4 done", flush=True)

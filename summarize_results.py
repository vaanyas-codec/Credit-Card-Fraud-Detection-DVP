"""Summarise full_results.jsonl: mean/std over seeds per scenario and model."""
import sys, pandas as pd
path = sys.argv[1] if len(sys.argv) > 1 else "full_results.jsonl"
r = pd.read_json(path, lines=True)
r = r[~r.scenario.str.contains("DONE")]
print("Seeds completed per scenario:")
print(r.groupby("scenario").seed.nunique().to_string(), "\n")
cols = ["AUPRC", "F1", "MCC", "Precision", "Recall"]
s = r.groupby(["scenario", "model"])[cols].agg(["mean", "std"]).round(3)
pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)
print(s.to_string())
s.to_csv("summary_mean_std.csv")
print("\nSaved summary_mean_std.csv")

"""Render tables + plots from outputs/results/metrics.json (score.py).

Writes:
  summary.csv / summary.md      ranked headline table with 95% CIs + AUCs
  detection_by_type.csv         per-corruption detection (point +/- run std)
  operating_points.csv          FAR-budget confidence floor per model
  significance.md               McNemar vs best model (Holm-corrected)
  slices.csv                    disaggregated metrics by subgroup
  plots/<model>_{reliability,threshold_sweep,detection}.png
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

from utils import load_config, abspath


def fmt_ci(d, pct=False, nd=3):
    if not isinstance(d, dict) or d.get("point") is None:
        return "-"
    p, lo, hi = d["point"], d.get("lo"), d.get("hi")
    mul, suf = (100, "%") if pct else (1, "")
    if lo is None:
        return f"{p*mul:.{nd}f}{suf}"
    return f"{p*mul:.{nd}f} [{lo*mul:.{nd}f},{hi*mul:.{nd}f}]{suf}"


def fmt_ms(d, pct=False, nd=2):
    if not isinstance(d, dict) or d.get("mean") is None:
        return "-"
    m, s = d["mean"], d.get("std") or 0.0
    mul, suf = (100, "%") if pct else (1, "")
    return f"{m*mul:.{nd}f}±{s*mul:.{nd}f}{suf}"


def fmt_pt_std(d, pct=False, nd=1):
    if not isinstance(d, dict) or d.get("point") is None:
        return "-"
    m, s = d["point"], d.get("std") or 0.0
    mul, suf = (100, "%") if pct else (1, "")
    return f"{m*mul:.{nd}f}±{s*mul:.{nd}f}{suf}"


def build_table(models):
    rows = []
    for a in models:
        det = a.get("determinism", {})
        rows.append({
            "model": a["model"], "quant": a.get("quant", "-"),
            "rank_score": round(a.get("rank_score", 0), 3),
            "F1": fmt_ci(a["f1"]),
            "precision": fmt_ci(a["precision"]),
            "recall": fmt_ci(a["recall"]),
            "false_accept": fmt_ci(a["false_accept_rate"], pct=True),   # safety-critical, 95% CI
            "false_reject": fmt_ci(a["false_reject_rate"], pct=True),
            "FRR_equiv": fmt_ci(a.get("false_reject_rate_equiv"), pct=True),
            "ROC_AUC": fmt_ci(a.get("roc_auc")),
            "PR_AUC": fmt_ci(a.get("pr_auc")),
            "ECE": fmt_ci(a.get("ece")),
            "Brier": fmt_ci(a.get("brier")),
            "flip_rate": fmt_ci(a.get("flip_rate"), pct=True),
            "FAR_run_std": f"{(det.get('false_accept_rate') or 0)*100:.2f}%",
            "field_recall": fmt_ci(a.get("field_recall")),
            "schema_conf": fmt_ci(a.get("schema_conformance"), pct=True),
            "guided_interv": fmt_ci(a.get("guided_intervention_rate"), pct=True),
            "unusable": fmt_ci(a.get("unusable_rate"), pct=True),
            "lat_p50_s": fmt_ms(a.get("latency_p50_s")),
            "req_per_s": fmt_ms(a.get("throughput_req_s"), nd=1),
            "peak_gpu_MiB": fmt_ms(a.get("peak_gpu_mem_mib"), nd=0),
        })
    return pd.DataFrame(rows)


def detection_table(models):
    rows = []
    for a in models:
        row = {"model": a["model"]}
        for t, d in a.get("detection_rate", {}).items():
            row[t] = fmt_pt_std(d, pct=True)
        rows.append(row)
    return pd.DataFrame(rows)


def operating_table(models):
    rows = []
    for a in models:
        op = a.get("operating_point", {})
        rows.append({
            "model": a["model"],
            "far_budget": f"{op.get('far_budget', 0)*100:.1f}%",
            "met": op.get("met"),
            "conf_floor": op.get("threshold"),
            "FAR_at_floor": f"{(op.get('false_accept_rate') or 0)*100:.2f}%",
            "precision_signed": f"{(op.get('precision') or 0)*100:.1f}%" if op.get("precision") is not None else "-",
            "coverage_faithful": f"{(op.get('coverage_of_faithful') or 0)*100:.1f}%",
        })
    return pd.DataFrame(rows)


def write_slices(models, path):
    rows = []
    for a in models:
        for dim, vals in a.get("slices", {}).items():
            for val, m in vals.items():
                rows.append({"model": a["model"], "dimension": dim, "value": val,
                             "n": m["n"],
                             "accuracy": _r(m["accuracy"]),
                             "false_accept_rate": _r(m["false_accept_rate"]),
                             "false_reject_rate": _r(m["false_reject_rate"])})
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    return rows


def _r(v, nd=4):
    return round(v, nd) if v is not None else None


def significance_md(sig):
    if not sig or not sig.get("comparisons"):
        return "_only one model — no pairwise tests._"
    lines = [f"Best model: **{sig['best']}**. McNemar paired test on per-example "
             "correctness vs best, Holm-corrected.\n",
             "| model | b (best wrong, this right) | c (best right, this wrong) | p_raw | p_holm | sig |",
             "|---|---|---|---|---|---|"]
    for c in sig["comparisons"]:
        lines.append(f"| {c['model']} | {c['b']} | {c['c']} | {c['p']:.4f} | "
                     f"{c['p_holm']:.4f} | {'yes' if c['significant'] else 'no'} |")
    return "\n".join(lines)


# --------------------------------------------------------------------- plots
def plot_reliability(a, path):
    curve = [c for c in a.get("reliability", []) if c.get("accuracy") is not None]
    if not curve:
        return
    xs = [c["avg_conf"] for c in curve]; ys = [c["accuracy"] for c in curve]
    ece = (a.get("ece") or {}).get("point")
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    plt.plot(xs, ys, "o-", label=a["model"])
    plt.title(f"Reliability — {a['model']}" + (f" (ECE={ece:.3f})" if ece is not None else ""))
    plt.xlabel("confidence"); plt.ylabel("accuracy")
    plt.xlim(0, 1); plt.ylim(0, 1); plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()


def plot_sweep(a, path, floor):
    sw = a.get("threshold_sweep", [])
    if not sw:
        return
    ts = [p["threshold"] for p in sw]
    prec = [p.get("precision") for p in sw]
    far = [p.get("false_accept_rate") for p in sw]
    cov = [p.get("coverage_of_faithful") for p in sw]
    op = a.get("operating_point", {})
    fig, ax1 = plt.subplots(figsize=(6.2, 4.6))
    ax1.plot(ts, prec, "g-", label="precision (signed)")
    ax1.plot(ts, cov, "b:", label="coverage of faithful")
    ax1.set_xlabel("confidence floor"); ax1.set_ylabel("precision / coverage"); ax1.set_ylim(0, 1.02)
    ax2 = ax1.twinx()
    ax2.plot(ts, far, "r-", label="false-accept rate")
    ax2.set_ylabel("false-accept rate", color="r"); ax2.tick_params(axis="y", labelcolor="r")
    ax1.axvline(floor, color="gray", ls="--", alpha=0.6)
    if op.get("threshold") is not None:
        ax1.axvline(op["threshold"], color="k", ls="-", alpha=0.8)
        ax1.text(op["threshold"], 0.02, f" op={op['threshold']:.2f}", fontsize=8)
    ax1.set_title(f"Confidence-floor sweep — {a['model']}")
    l1, lb1 = ax1.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_detection(a, path):
    items = [(t, d["point"], d.get("std") or 0) for t, d in a.get("detection_rate", {}).items()
             if d.get("point") is not None]
    if not items:
        return
    items.sort()
    labels = [t for t, _, _ in items]; vals = [v for _, v, _ in items]; errs = [s for _, _, s in items]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, vals, yerr=errs, capsize=4, color="#4c72b0")
    plt.ylim(0, 1.02); plt.ylabel("detection rate"); plt.title(f"Per-corruption detection — {a['model']}")
    plt.xticks(rotation=35, ha="right"); plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    res_dir = abspath(cfg["paths"]["results_dir"])
    plot_dir = abspath(cfg["paths"]["plots_dir"])
    os.makedirs(plot_dir, exist_ok=True)

    with open(res_dir / "metrics.json") as f:
        data = json.load(f)
    models = data["models"]
    if not models:
        print("[report] no models"); return

    df = build_table(models); df.to_csv(res_dir / "summary.csv", index=False)
    ddf = detection_table(models); ddf.to_csv(res_dir / "detection_by_type.csv", index=False)
    odf = operating_table(models); odf.to_csv(res_dir / "operating_points.csv", index=False)
    write_slices(models, res_dir / "slices.csv")
    sig_md = significance_md(data.get("significance"))
    with open(res_dir / "significance.md", "w") as f:
        f.write(sig_md + "\n")

    floor = cfg["thresholds"]["production_floor"]
    for a in models:
        m = a["model"]
        plot_reliability(a, plot_dir / f"{m}_reliability.png")
        plot_sweep(a, plot_dir / f"{m}_threshold_sweep.png", floor)
        plot_detection(a, plot_dir / f"{m}_detection.png")

    env = models[0]
    md = ["# Policy-Faithfulness Model Evaluation\n",
          f"- vLLM: `{env.get('vllm_version','?')}` | GPU: `{env.get('gpu','?')}` | "
          f"seed {data.get('config_seed')} | runs/model {env.get('n_runs')} | "
          f"sklearn AUC: {data.get('have_sklearn')}\n",
          "- Positive class = catch the **unfaithful** JSON. Brackets are 95% CIs "
          "(cluster bootstrap over examples). FAR weighted heaviest in ranking.\n",
          "\n## Ranking (best first)\n", df.to_markdown(index=False),
          "\n\n## FAR-budget operating point (production confidence floor)\n",
          odf.to_markdown(index=False),
          "\n\n## Detection by corruption type (point ± run std)\n", ddf.to_markdown(index=False),
          "\n\n## Significance vs best model\n", sig_md,
          f"\n\n## Subgroups & errors\nSee `slices.csv` and `errors_<model>.csv`. "
          f"Plots in `{plot_dir}`.\n"]
    with open(res_dir / "summary.md", "w") as f:
        f.write("\n".join(md))

    print(tabulate(df, headers="keys", tablefmt="github", showindex=False))
    print("\nOperating point (FAR budget):")
    print(tabulate(odf, headers="keys", tablefmt="github", showindex=False))
    print("\n" + sig_md)
    print(f"\n[report] wrote summary.{{csv,md}}, operating_points.csv, detection_by_type.csv, "
          f"slices.csv, significance.md, error CSVs, and plots in {plot_dir}")


if __name__ == "__main__":
    main()

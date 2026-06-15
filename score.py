"""Score saved raw verdicts with data-science rigor. Re-runnable WITHOUT re-running
inference (reads outputs/raw/*.jsonl from run.py).

Positive class = "catch the unfaithful one" (gold_verdict == NOT_FAITHFUL).
Decision: predicted_accept = (verdict == FAITHFUL & schema-valid) = "sign it";
anything else = predicted_flag = "don't sign" (unusable/AMBIGUOUS counts as flag).

Beyond point metrics this produces, per model:
  - 95% CIs via CLUSTER bootstrap (resample examples, keep their repeated runs together)
    so uncertainty reflects sampling, not just run-to-run noise.
  - threshold-INDEPENDENT quality: ROC-AUC and PR-AUC on a continuous unfaithfulness
    score derived from verdict+confidence.
  - calibration: ECE, MCE, Brier, reliability curve.
  - FAR-budget operating point: the lowest confidence floor whose false-accept rate
    meets thresholds.far_budget, with the coverage you pay for it.
  - self-consistency: verdict flip-rate across repeated runs (temp 0 isn't bit-identical).
  - subgroup slices (lang, equivalence_hard, period, currency, ...).
  - error capture (false-accepts first) for human review.
Cross-model: McNemar paired test vs the best model, Holm-corrected.
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

import stats as st
from utils import load_config, abspath, read_jsonl

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False


# ----------------------------------------------------------------- decisions
def _decision(row):
    parsed, ok = row.get("parsed"), row.get("schema_ok")
    if ok and isinstance(parsed, dict):
        verdict = parsed.get("verdict", "UNUSABLE")
        try:
            conf = float(parsed.get("confidence", 0.5))
        except Exception:
            conf = 0.5
    else:
        verdict, conf = "UNUSABLE", 0.0
    return (verdict == "FAITHFUL"), verdict, conf, bool(ok)


def _score(verdict, conf, ok, unusable):
    if not ok:
        return unusable
    if verdict == "NOT_FAITHFUL":
        return 0.5 + 0.5 * conf
    if verdict == "FAITHFUL":
        return 0.5 - 0.5 * conf
    return unusable


# ------------------------------------------------------------ eval attributes
def load_attrs(cfg):
    p = abspath(cfg["paths"]["eval_dataset"])
    if not p.exists():
        return {}
    attrs = {}
    for r in read_jsonl(p):
        j = r.get("json", {})
        cats = j.get("categories", [])
        attrs[r["id"]] = {
            "english": r.get("english", ""),
            "lang": r.get("lang", "english"),
            "period": j.get("period", "?"),
            "currency": j.get("currency", "?"),
            "n_categories": len(cats) if isinstance(cats, list) else 1,
            "merchants_type": "any" if j.get("merchants") == "any" else "list",
            "equivalence_hard": bool(r.get("equivalence_hard", False)),
        }
    return attrs


# ----------------------------------------------- per-run metrics (for run std)
def run_headline(rows):
    tp = fp = fn = gf = gu = 0
    for r in rows:
        accept, verdict, conf, ok = _decision(r)
        flag = not accept
        gunf = r["gold_verdict"] == "NOT_FAITHFUL"
        if gunf:
            gu += 1
            tp += int(flag); fn += int(not flag)
        else:
            gf += 1
            fp += int(flag)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"f1": f1, "precision": prec, "recall": rec,
            "false_accept_rate": fn / gu if gu else 0.0,
            "false_reject_rate": fp / gf if gf else 0.0}


def run_detection(rows):
    hit, tot = defaultdict(int), defaultdict(int)
    for r in rows:
        if r["gold_verdict"] != "NOT_FAITHFUL":
            continue
        accept, *_ = _decision(r)
        tot[r["corruption_type"]] += 1
        hit[r["corruption_type"]] += int(not accept)
    return {t: hit[t] / tot[t] for t in tot}


# ------------------------------------------------------------- pooled metrics
def compute_pooled(model, rows_per_run, attrs, cfg):
    ecfg = cfg["evaluation"]
    unusable = ecfg.get("unusable_score", 0.5)
    ids, g_unf, accept, flag, conf, score, correct, ok = [], [], [], [], [], [], [], []
    g_mm, p_mm, verdicts, freeform_ok = [], [], [], []
    for rows in rows_per_run:
        for r in rows:
            a, v, c, o = _decision(r)
            s = _score(v, c, o, unusable)
            gunf = r["gold_verdict"] == "NOT_FAITHFUL"
            ids.append(r["id"]); g_unf.append(gunf); accept.append(a); flag.append(not a)
            conf.append(c); score.append(s); ok.append(o)
            correct.append((not a) == gunf); verdicts.append(v)
            g_mm.append({m["field"] for m in r.get("gold_mismatches", [])})
            p_mm.append({m.get("field") for m in ((r.get("parsed") or {}).get("mismatches") or [])
                         if isinstance(m, dict)})
            freeform_ok.append(r.get("freeform_schema_ok"))   # None if not probed
    n = len(ids)
    g_unf = np.array(g_unf); accept = np.array(accept); flag = np.array(flag)
    conf = np.array(conf, float); score = np.array(score, float)
    correct = np.array(correct); ok = np.array(ok)

    # clusters = positions sharing an example id
    by_id = defaultdict(list)
    for i, x in enumerate(ids):
        by_id[x].append(i)
    clusters = [np.array(v) for v in by_id.values()]

    def far(idx):
        du = g_unf[idx]; d = du.sum()
        return (accept[idx] & du).sum() / d if d else None

    def frr(idx):
        df = ~g_unf[idx]; d = df.sum()
        return (flag[idx] & df).sum() / d if d else None

    def precision(idx):
        pp = flag[idx].sum()
        return (flag[idx] & g_unf[idx]).sum() / pp if pp else None

    def recall(idx):
        d = g_unf[idx].sum()
        return (flag[idx] & g_unf[idx]).sum() / d if d else None

    def f1(idx):
        p, r = precision(idx), recall(idx)
        return 2 * p * r / (p + r) if (p and r and (p + r)) else (0.0 if (p is not None and r is not None) else None)

    def roc(idx):
        y = g_unf[idx]
        if not _HAVE_SK or y.all() or (~y).all():
            return None
        return roc_auc_score(y, score[idx])

    def prauc(idx):
        y = g_unf[idx]
        if not _HAVE_SK or y.sum() == 0:
            return None
        return average_precision_score(y, score[idx])

    stat_fns = {"false_accept_rate": far, "false_reject_rate": frr,
                "precision": precision, "recall": recall, "f1": f1,
                "roc_auc": roc, "pr_auc": prauc}
    boot = st.cluster_bootstrap(clusters, stat_fns,
                                n_boot=ecfg.get("bootstrap_iters", 1000),
                                seed=cfg["seed"], ci=ecfg.get("bootstrap_ci", 0.95))

    # calibration (pooled)
    ece, mce, reliability = _calibration(conf, correct)
    brier = float(np.mean((conf - correct.astype(float)) ** 2)) if n else None

    # field precision/recall on caught unfaithful
    f_rec, f_prec = [], []
    for i in range(n):
        if g_unf[i] and flag[i] and g_mm[i]:
            inter = len(g_mm[i] & p_mm[i])
            f_rec.append(inter / len(g_mm[i]))
            if p_mm[i]:
                f_prec.append(inter / len(p_mm[i]))

    # equivalence-hard FRR
    eq = np.array([attrs.get(x, {}).get("equivalence_hard", False) for x in ids])
    eq_faithful = eq & ~g_unf
    frr_equiv = (flag[eq_faithful].sum() / eq_faithful.sum()) if eq_faithful.sum() else None

    # self-consistency: flip-rate across runs
    flips = 0
    multi = 0
    for v in by_id.values():
        if len(v) > 1:
            multi += 1
            if len({bool(flag[i]) for i in v}) > 1:
                flips += 1
    flip_rate = flips / multi if multi else None

    # guided-decoding intervention: among probed examples, how often the FREE-FORM
    # (unconstrained) output failed the schema -> guided decoding had to intervene.
    probed = [bool(x) for x in freeform_ok if x is not None]
    intervention = (1.0 - (sum(probed) / len(probed))) if probed else None
    unusable_rate = float(1.0 - ok.mean()) if n else None   # guided output STILL invalid (rare)

    pooled = {
        **{k: boot[k] for k in stat_fns},
        "false_reject_rate_equiv": _pt(frr_equiv),
        "field_recall": _pt(float(np.mean(f_rec)) if f_rec else None),
        "field_precision": _pt(float(np.mean(f_prec)) if f_prec else None),
        "schema_conformance": _pt(float(ok.mean()) if n else None),
        "guided_intervention_rate": _pt(intervention),
        "unusable_rate": _pt(unusable_rate),
        "n_intervention_probed": len(probed),
        "ece": _pt(ece), "mce": _pt(mce), "brier": _pt(brier),
        "flip_rate": _pt(flip_rate),
        "reliability": reliability,
        "threshold_sweep": _sweep(accept, conf, g_unf, cfg),
        "operating_point": _operating_point(accept, conf, g_unf, cfg),
        "detection_rate": _detection_pooled(rows_per_run),
        "slices": _slices(cfg, ids, attrs, g_unf, accept, flag, correct),
    }
    consensus = _consensus(by_id, flag, g_unf)
    errors = _errors(model, by_id, ids, verdicts, conf, g_unf, flag, g_mm, p_mm, attrs, cfg)
    return pooled, consensus, errors


def _pt(v):
    return {"point": v, "lo": None, "hi": None}


def _calibration(conf, correct, n_bins=10):
    if len(conf) == 0:
        return None, None, []
    bins = np.linspace(0, 1, n_bins + 1)
    ece = mce = 0.0
    curve = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            curve.append({"bin_lo": float(lo), "bin_hi": float(hi), "count": 0,
                          "avg_conf": None, "accuracy": None})
            continue
        ac, acc = float(conf[mask].mean()), float(correct[mask].mean())
        gap = abs(ac - acc)
        ece += (cnt / len(conf)) * gap
        mce = max(mce, gap)
        curve.append({"bin_lo": float(lo), "bin_hi": float(hi), "count": cnt,
                      "avg_conf": ac, "accuracy": acc})
    return float(ece), float(mce), curve


def _sweep(accept, conf, g_unf, cfg):
    t = cfg["thresholds"]
    ts = np.round(np.arange(t["sweep_start"], t["sweep_end"] + 1e-9, t["sweep_step"]), 4)
    tot_u, tot_f = int(g_unf.sum()), int((~g_unf).sum())
    out = []
    for th in ts:
        signed = accept & (conf >= th)
        ns = int(signed.sum())
        nf = int((signed & ~g_unf).sum())
        out.append({"threshold": float(th), "n_signed": ns,
                    "precision": (nf / ns) if ns else None,
                    "false_accept_rate": ((ns - nf) / tot_u) if tot_u else 0.0,
                    "coverage_of_faithful": (nf / tot_f) if tot_f else 0.0})
    return out


def _operating_point(accept, conf, g_unf, cfg):
    budget = cfg["thresholds"].get("far_budget", 0.02)
    sweep = _sweep(accept, conf, g_unf, cfg)
    meeting = [p for p in sweep if p["false_accept_rate"] <= budget]
    chosen = min(meeting, key=lambda p: p["threshold"]) if meeting else min(
        sweep, key=lambda p: p["false_accept_rate"])
    return {"far_budget": budget, "met": bool(meeting),
            "threshold": chosen["threshold"], "false_accept_rate": chosen["false_accept_rate"],
            "precision": chosen["precision"], "coverage_of_faithful": chosen["coverage_of_faithful"]}


def _detection_pooled(rows_per_run):
    hit, tot = defaultdict(int), defaultdict(int)
    for rows in rows_per_run:
        for r in rows:
            if r["gold_verdict"] != "NOT_FAITHFUL":
                continue
            accept, *_ = _decision(r)
            tot[r["corruption_type"]] += 1
            hit[r["corruption_type"]] += int(not accept)
    return {t: hit[t] / tot[t] for t in sorted(tot)}


def _slices(cfg, ids, attrs, g_unf, accept, flag, correct):
    out = {}
    for dim in cfg["evaluation"].get("slices", []):
        vals = np.array([str(attrs.get(x, {}).get(dim, "?")) for x in ids])
        d = {}
        for val in sorted(set(vals)):
            m = vals == val
            gu = g_unf[m]; gf = ~g_unf[m]
            d[val] = {
                "n": int(m.sum()),
                "accuracy": float(correct[m].mean()) if m.sum() else None,
                "false_accept_rate": float((accept[m] & gu).sum() / gu.sum()) if gu.sum() else None,
                "false_reject_rate": float((flag[m] & gf).sum() / gf.sum()) if gf.sum() else None,
            }
        out[dim] = d
    return out


def _consensus(by_id, flag, g_unf):
    """Majority flag decision per example across runs -> correctness dict."""
    out = {}
    for x, pos in by_id.items():
        maj_flag = np.mean([flag[i] for i in pos]) >= 0.5
        gunf = bool(g_unf[pos[0]])
        out[x] = bool(maj_flag == gunf)
    return out


def _errors(model, by_id, ids, verdicts, conf, g_unf, flag, g_mm, p_mm, attrs, cfg):
    """Errors judged by cross-run CONSENSUS so the reported verdict matches the
    error type (no single-run/consensus contradiction)."""
    cap = cfg["evaluation"].get("error_examples_per_model", 100)
    recs = []
    for x, pos in by_id.items():
        gunf = bool(g_unf[pos[0]])
        maj_flag = np.mean([flag[i] for i in pos]) >= 0.5
        if maj_flag == gunf:
            continue
        # modal verdict + mean confidence across this example's runs (consistent display)
        vs = [verdicts[i] for i in pos]
        modal = max(set(vs), key=vs.count)
        i = pos[0]
        a = attrs.get(x, {})
        recs.append({
            "type": "false_accept" if gunf else "false_reject",
            "id": x, "lang": a.get("lang", ""), "english": a.get("english", ""),
            "gold_verdict": "NOT_FAITHFUL" if gunf else "FAITHFUL",
            "pred_verdict": modal,
            "confidence": round(float(np.mean([conf[i] for i in pos])), 3),
            "n_runs_flagged": int(sum(int(flag[i]) for i in pos)),
            "gold_fields": ";".join(sorted(g_mm[i])),
            "pred_fields": ";".join(sorted(y for y in p_mm[i] if y)),
        })
    recs.sort(key=lambda r: (r["type"] != "false_accept", -r["confidence"]))
    return recs[:cap]


# ------------------------------------------------------------------ assemble
def _mean_std(vals):
    vals = [v for v in vals if v is not None]
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals))} if vals else {"mean": None, "std": None}


def rank_score(agg, weights):
    s = 0.0
    for key, w in weights.items():
        v = agg.get(key)
        if isinstance(v, dict):
            v = v.get("point", v.get("mean"))
        if v is not None:
            s += w * v
    return s


def main():
    ap = argparse.ArgumentParser(description="Score saved raw verdicts.")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    raw_dir = abspath(cfg["paths"]["raw_outputs_dir"])
    res_dir = abspath(cfg["paths"]["results_dir"])
    os.makedirs(res_dir, exist_ok=True)
    attrs = load_attrs(cfg)
    if not _HAVE_SK:
        print("[score] scikit-learn missing -> ROC/PR-AUC will be null (pip install scikit-learn)")

    files = sorted(glob.glob(str(raw_dir / "*__run*.jsonl")))
    if not files:
        print(f"[score] no raw outputs in {raw_dir}; run.py first.")
        return
    by_model = defaultdict(list)
    for f in files:
        by_model[os.path.basename(f).split("__run")[0]].append(f)

    all_agg, consensus_all = [], {}
    for model, fs in sorted(by_model.items()):
        rows_per_run = [read_jsonl(f) for f in sorted(fs)]
        metas = []
        for f in sorted(fs):
            mp = f.replace(".jsonl", ".meta.json")
            if os.path.exists(mp):
                with open(mp) as mf:
                    metas.append(json.load(mf))
        run_h = [run_headline(r) for r in rows_per_run]
        run_d = [run_detection(r) for r in rows_per_run]
        pooled, consensus, errors = compute_pooled(model, rows_per_run, attrs, cfg)
        consensus_all[model] = consensus

        agg = {"model": model, "n_runs": len(rows_per_run)}
        agg.update(pooled)
        # run-to-run determinism (std across repeated runs at temp 0)
        agg["determinism"] = {k: _mean_std([h[k] for h in run_h])["std"]
                              for k in ["f1", "false_accept_rate", "false_reject_rate"]}
        # detection per type: pooled point + run std
        types = set().union(*[set(d) for d in run_d]) if run_d else set()
        agg["detection_rate"] = {t: {"point": pooled["detection_rate"].get(t),
                                     "std": _mean_std([d.get(t) for d in run_d])["std"]}
                                 for t in sorted(types)}
        # runtime stats from meta
        for k in ["latency_p50_s", "latency_p95_s", "tokens_per_s",
                  "throughput_req_s", "peak_gpu_mem_mib"]:
            agg[k] = _mean_std([m.get(k) for m in metas]) if metas else {"mean": None, "std": None}
        if metas:
            for k in ["hf_id", "quant", "vllm_version", "gpu"]:
                agg[k] = metas[0].get(k)
        agg["rank_score"] = rank_score(agg, cfg["ranking_weights"])
        agg["_errors"] = errors
        all_agg.append(agg)
        ci = agg["false_accept_rate"]
        print(f"[score] {model}: F1={agg['f1']['point']:.3f} "
              f"FAR={ci['point']:.3f} [{ci['lo']:.3f},{ci['hi']:.3f}] "
              f"AUC={_g(agg['roc_auc'])} flip={_g(agg['flip_rate'])}")

    all_agg.sort(key=lambda a: a["rank_score"], reverse=True)

    # cross-model significance: McNemar vs best model (Holm-corrected)
    significance = _significance(all_agg, consensus_all, cfg)

    # write outputs
    for a in all_agg:
        _write_errors(res_dir, a.pop("_errors"), a["model"])
    with open(res_dir / "metrics.json", "w") as f:
        json.dump({"config_seed": cfg["seed"], "have_sklearn": _HAVE_SK,
                   "models": all_agg, "significance": significance}, f, indent=2)
    print(f"[score] wrote {res_dir / 'metrics.json'} + per-model error CSVs. Next: python report.py")


def _g(d):
    v = d.get("point") if isinstance(d, dict) else d
    return f"{v:.3f}" if v is not None else "NA"


def _significance(all_agg, consensus_all, cfg):
    if len(all_agg) < 2:
        return {"best": all_agg[0]["model"] if all_agg else None, "comparisons": []}
    best = all_agg[0]["model"]
    bcon = consensus_all[best]
    comps, pvals = [], []
    for a in all_agg[1:]:
        m = a["model"]
        common = [k for k in bcon if k in consensus_all[m]]
        bc = [bcon[k] for k in common]
        mc = [consensus_all[m][k] for k in common]
        res = st.mcnemar(bc, mc)
        comps.append({"model": m, "vs": best, **res, "n_common": len(common)})
        pvals.append(res["p"])
    for c, adj in zip(comps, st.holm(pvals)):
        c["p_holm"] = adj
        c["significant"] = adj < cfg["evaluation"].get("alpha", 0.05)
    return {"best": best, "comparisons": comps}


def _write_errors(res_dir, errors, model):
    if not errors:
        return
    path = res_dir / f"errors_{model}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(errors[0].keys()))
        w.writeheader()
        w.writerows(errors)


if __name__ == "__main__":
    main()

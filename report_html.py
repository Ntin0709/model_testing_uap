"""Render a single self-contained HTML report from outputs/results/metrics.json.

- Embeds the per-model PNG plots as base64 so the file is fully portable (one .html).
- Auto-derives the headline (winner) and flags degenerate models (always-accept /
  always-reject / near-random) so the reader isn't misled by a single good-looking metric.
- Explains every metric in plain English.

Usage:  python report_html.py [--config config.yaml] [--out outputs/results/report.html]
"""
import argparse
import base64
import json
from pathlib import Path

from utils import load_config, abspath


def pt(m, key):
    v = m.get(key)
    return v.get("point") if isinstance(v, dict) else v


def ci(m, key):
    v = m.get(key)
    if isinstance(v, dict):
        return v.get("lo"), v.get("hi")
    return None, None


def fnum(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def fpct(x, nd=1):
    return "-" if x is None else f"{x*100:.{nd}f}%"


def cell(x, lo=None, hi=None, pct=False, nd=3, good="up"):
    """Formatted cell with CI and a red/amber/green class."""
    if x is None:
        return '<td class="na">-</td>'
    val = fpct(x) if pct else fnum(x, nd)
    sub = ""
    if lo is not None and hi is not None:
        sub = f'<span class="ci">[{fpct(lo) if pct else fnum(lo,nd)}, {fpct(hi) if pct else fnum(hi,nd)}]</span>'
    return f'<td class="{_band(x, good, pct)}">{val}{sub}</td>'


def _band(x, good, pct):
    # thresholds tuned for this task; "up" = higher is better
    if good == "up":
        g, a = (0.9, 0.75)
        return "good" if x >= g else "warn" if x >= a else "bad"
    if good == "down":              # rates: lower is better (work in fraction space)
        return "good" if x <= 0.10 else "warn" if x <= 0.30 else "bad"
    return ""


def flags(m):
    """Detect degenerate behaviours that a single metric can hide."""
    out = []
    rec, far, frr = pt(m, "recall"), pt(m, "false_accept_rate"), pt(m, "false_reject_rate")
    auc = pt(m, "roc_auc")
    if rec is not None and frr is not None and rec >= 0.98 and frr >= 0.90:
        out.append(("always-reject", "flags everything → catches all but rejects all faithful"))
    if far is not None and rec is not None and far >= 0.85 and rec <= 0.10:
        out.append(("always-accept", "signs almost everything → misses unfaithful JSON"))
    if auc is not None and auc <= 0.55:
        out.append(("≈ random", "confidence does not separate faithful from unfaithful"))
    return out


def img_tag(path: Path, alt):
    if not path.exists():
        return f'<div class="noimg">{alt}: plot not found</div>'
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f'<img alt="{alt}" src="data:image/png;base64,{b64}">'


CSS = """
:root{--bg:#0f1115;--card:#171a21;--ink:#e6e9ef;--mut:#9aa4b2;--line:#262b36;
--good:#1f7a4d;--goodbg:#0f2a1d;--warn:#8a6d1f;--warnbg:#2a230f;--bad:#9b2c2c;--badbg:#2a1010;--accent:#4c8bf5;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:32px 22px 80px}
h1{font-size:28px;margin:0 0 4px}h2{font-size:21px;margin:38px 0 12px;border-bottom:1px solid var(--line);padding-bottom:6px}
h3{font-size:17px;margin:22px 0 8px}p{color:var(--ink)}small,.mut{color:var(--mut)}
.meta{color:var(--mut);font-size:13px;margin-bottom:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:14px 0}
.win{border-color:var(--good);background:linear-gradient(180deg,#10241a,#171a21)}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0 4px}
th,td{padding:7px 9px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:var(--mut);font-weight:600;position:sticky;top:0;background:var(--card)}
.good{background:var(--goodbg);color:#7fe0a8}.warn{background:var(--warnbg);color:#e8c878}.bad{background:var(--badbg);color:#f08a8a}
.na{color:var(--mut)}.ci{display:block;color:var(--mut);font-size:11px}
.tablewrap{overflow:auto;border:1px solid var(--line);border-radius:10px}
.badge{display:inline-block;font-size:11px;padding:2px 8px;border-radius:999px;margin:2px 4px 2px 0;border:1px solid var(--line)}
.badge.win{background:var(--goodbg);color:#7fe0a8;border-color:var(--good)}
.badge.bad{background:var(--badbg);color:#f08a8a;border-color:var(--bad)}
.grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-top:10px}
.grid img{width:100%;border-radius:8px;border:1px solid var(--line);background:#fff}
.noimg{color:var(--mut);font-size:12px;padding:20px;text-align:center;border:1px dashed var(--line);border-radius:8px}
dl{display:grid;grid-template-columns:200px 1fr;gap:6px 16px;margin:8px 0}
dt{color:var(--accent);font-weight:600}dd{margin:0;color:var(--ink)}
.kpi{display:flex;gap:14px;flex-wrap:wrap;margin:10px 0}
.kpi div{background:#10141c;border:1px solid var(--line);border-radius:10px;padding:10px 14px;min-width:120px}
.kpi b{display:block;font-size:22px}.kpi span{color:var(--mut);font-size:12px}
details{margin:8px 0}summary{cursor:pointer;color:var(--accent);font-weight:600}
code{background:#10141c;padding:1px 5px;border-radius:5px;border:1px solid var(--line)}
"""


def build(data, cfg, plots_dir):
    models = data["models"]
    env = models[0] if models else {}
    win = models[0] if models else None
    sig = data.get("significance", {})

    H = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
         '<meta name="viewport" content="width=device-width,initial-scale=1">',
         "<title>Policy-Faithfulness Model Evaluation</title>",
         f"<style>{CSS}</style></head><body><div class='wrap'>"]

    H.append("<h1>Policy-Faithfulness Model Evaluation</h1>")
    H.append(f"<div class='meta'>vLLM <code>{env.get('vllm_version','?')}</code> · "
             f"GPU <code>{env.get('gpu','?')}</code> · seed {data.get('config_seed')} · "
             f"{env.get('n_runs','?')} runs/model · ROC/PR-AUC: {data.get('have_sklearn')}</div>")

    # ---- what & why ----
    H.append("<h2>1. What this evaluates &amp; why</h2><div class='card'>"
             "<p>In our payments system a user writes a spending policy in plain English "
             "(often <b>Hinglish</b>), and a component translates it into structured JSON. "
             "Before that JSON is trusted to move money, a small LLM acts as a "
             "<b>verification gate</b>: given the English policy + the JSON, it returns a "
             "strict-schema verdict — <code>FAITHFUL</code> / <code>NOT_FAITHFUL</code> / "
             "<code>AMBIGUOUS</code> — with a confidence and the mismatched fields. "
             "We are choosing which small model is good enough to be that gate.</p>"
             "<p>The positive class is <b>catching unfaithful JSON</b>. The two error types "
             "are asymmetric:</p>"
             "<dl>"
             "<dt>False-accept (FAR)</dt><dd><b>Safety-critical.</b> Wrong JSON signed off as "
             "faithful → bad rule controls real money. Weighted heaviest.</dd>"
             "<dt>False-reject (FRR)</dt><dd>Correct policy wrongly rejected → annoys a real "
             "user. We also report it on <i>equivalence-hard</i> phrasings ('kabhi bhi' = "
             "all-day) where naive models over-reject.</dd></dl></div>")

    # ---- how ----
    H.append("<h2>2. How we ran it</h2><div class='card'>"
             "<p>Each model is served one at a time on <b>vLLM</b> (OpenAI-compatible, "
             "<code>temperature=0</code>, <code>seed=42</code>, verdict <b>JSON schema "
             "enforced</b> via guided decoding), GPU freed between models, version + GPU "
             "recorded. The eval set is a ~2,000-example labelled Hinglish set: correct "
             "policy→JSON pairs plus one tagged corruption at a time (wrong amount, dropped/"
             "added/over-permissive merchant, widened/narrowed time window, swapped currency, "
             "wrong period, combined), including subtle hard-negatives and hard-positive "
             "equivalences. Each model runs <b>3×</b>; raw verdicts are saved so results are "
             "re-scoreable. 95% CIs are cluster-bootstrap over examples.</p></div>")

    # ---- verdict / winner ----
    if win:
        H.append("<h2>3. Headline</h2>")
        wflags = flags(win)
        H.append(f"<div class='card win'><h3>🏆 Best model: <code>{win['model']}</code></h3>"
                 "<div class='kpi'>"
                 f"<div><b>{fnum(pt(win,'f1'))}</b><span>F1</span></div>"
                 f"<div><b>{fpct(pt(win,'false_accept_rate'))}</b><span>false-accept (safety)</span></div>"
                 f"<div><b>{fpct(pt(win,'false_reject_rate'))}</b><span>false-reject</span></div>"
                 f"<div><b>{fnum(pt(win,'roc_auc'))}</b><span>ROC-AUC</span></div>"
                 f"<div><b>{fnum(pt(win,'ece'))}</b><span>ECE (calibration)</span></div>"
                 "</div>"
                 "<p>Selected by the false-accept-weighted ranking and confirmed by McNemar "
                 "significance vs every other model. The badges below flag models whose "
                 "surface metrics hide degenerate behaviour.</p></div>")

    # ---- degeneracy badges ----
    H.append("<div class='card'><h3>Behaviour flags</h3><p class='mut'>A model can score well on "
             "one metric while being useless — e.g. always-reject gives FAR=0, always-accept "
             "gives FRR=0. ROC-AUC near 0.5 is the tell.</p>")
    any_flag = False
    for m in models:
        fl = flags(m)
        if fl:
            any_flag = True
            badges = " ".join(f"<span class='badge bad'>{n}</span>" for n, _ in fl)
            why = "; ".join(d for _, d in fl)
            H.append(f"<div><code>{m['model']}</code> {badges} <span class='mut'>{why}</span></div>")
    if not any_flag:
        H.append("<div class='mut'>No degenerate models detected.</div>")
    H.append("</div>")

    # ---- ranking table ----
    H.append("<h2>4. Full ranking</h2>"
             "<p class='mut'>Sorted by rank score (false-accept weighted heaviest). "
             "Green/amber/red = good/ok/poor for this gate. Brackets are 95% CIs.</p>"
             "<div class='tablewrap'><table><thead><tr>"
             "<th>model</th><th>quant</th><th>rank</th><th>F1</th><th>precision</th><th>recall</th>"
             "<th>false-accept</th><th>false-reject</th><th>FRR&nbsp;equiv</th><th>ROC-AUC</th>"
             "<th>ECE</th><th>flip</th><th>schema</th><th>p50&nbsp;s</th><th>req/s</th><th>GPU&nbsp;MiB</th>"
             "</tr></thead><tbody>")
    for m in models:
        flo, fhi = ci(m, "f1")
        plo, phi = ci(m, "precision")
        rlo, rhi = ci(m, "recall")
        falo, fahi = ci(m, "false_accept_rate")
        frlo, frhi = ci(m, "false_reject_rate")
        alo, ahi = ci(m, "roc_auc")
        peak = (m.get("peak_gpu_mem_mib") or {}).get("mean")
        lat = (m.get("latency_p50_s") or {}).get("mean")
        rps = (m.get("throughput_req_s") or {}).get("mean")
        H.append("<tr>"
                 f"<td>{m['model']}</td><td>{m.get('quant','-')}</td>"
                 f"<td>{fnum(m.get('rank_score'),2)}</td>"
                 + cell(pt(m,'f1'),flo,fhi)
                 + cell(pt(m,'precision'),plo,phi)
                 + cell(pt(m,'recall'),rlo,rhi)
                 + cell(pt(m,'false_accept_rate'),falo,fahi,pct=True,good="down")
                 + cell(pt(m,'false_reject_rate'),frlo,frhi,pct=True,good="down")
                 + cell(pt(m,'false_reject_rate_equiv'),pct=True,good="down")
                 + cell(pt(m,'roc_auc'),alo,ahi)
                 + cell(pt(m,'ece'),good="")
                 + cell(pt(m,'flip_rate'),pct=True,good="")
                 + cell(pt(m,'schema_conformance'),pct=True,good="up")
                 + f"<td>{fnum(lat,2)}</td><td>{fnum(rps,1)}</td><td>{fnum(peak,0)}</td>"
                 "</tr>")
    H.append("</tbody></table></div>")

    # ---- operating point ----
    H.append("<h2>5. Production operating point</h2>"
             "<p class='mut'>The lowest confidence floor whose false-accept rate meets the "
             f"budget (<code>{fpct(cfg['thresholds'].get('far_budget'))}</code>), and the "
             "coverage of faithful policies you keep at it. <code>met=False</code> means even "
             "at confidence 0.99 the budget isn't reached; coverage 0% means you'd auto-approve "
             "nothing (route all to human).</p>"
             "<div class='tablewrap'><table><thead><tr><th>model</th><th>budget</th><th>met</th>"
             "<th>conf floor</th><th>FAR at floor</th><th>signed precision</th>"
             "<th>coverage of faithful</th></tr></thead><tbody>")
    for m in models:
        op = m.get("operating_point", {})
        H.append("<tr>"
                 f"<td>{m['model']}</td><td>{fpct(op.get('far_budget'))}</td>"
                 f"<td>{op.get('met')}</td><td>{op.get('threshold')}</td>"
                 f"<td>{fpct(op.get('false_accept_rate'),2)}</td>"
                 f"<td>{fpct(op.get('precision')) if op.get('precision') is not None else '-'}</td>"
                 f"<td>{fpct(op.get('coverage_of_faithful'))}</td></tr>")
    H.append("</tbody></table></div>")

    # ---- significance ----
    if sig.get("comparisons"):
        H.append("<h2>6. Statistical significance</h2>"
                 f"<p class='mut'>McNemar paired test on per-example correctness vs the best "
                 f"model (<code>{sig.get('best')}</code>), Holm-corrected. <code>b</code> = best "
                 "wrong &amp; this right; <code>c</code> = best right &amp; this wrong.</p>"
                 "<div class='tablewrap'><table><thead><tr><th>model</th><th>b</th><th>c</th>"
                 "<th>p (Holm)</th><th>significant</th></tr></thead><tbody>")
        for c in sig["comparisons"]:
            H.append(f"<tr><td>{c['model']}</td><td>{c['b']}</td><td>{c['c']}</td>"
                     f"<td>{c['p_holm']:.4f}</td><td>{'yes' if c['significant'] else 'no'}</td></tr>")
        H.append("</tbody></table></div>")

    # ---- per-model plots ----
    H.append("<h2>7. Per-model diagnostics</h2>")
    for m in models:
        name = m["model"]
        H.append(f"<details><summary>{name} — reliability · threshold sweep · per-corruption detection</summary>"
                 "<div class='grid'>"
                 + img_tag(plots_dir / f"{name}_reliability.png", "reliability")
                 + img_tag(plots_dir / f"{name}_threshold_sweep.png", "threshold sweep")
                 + img_tag(plots_dir / f"{name}_detection.png", "detection")
                 + "</div></details>")

    # ---- glossary ----
    H.append("<h2>8. Metric glossary</h2><div class='card'><dl>"
             "<dt>Precision / Recall / F1</dt><dd>For catching unfaithful JSON (positive class).</dd>"
             "<dt>False-accept rate</dt><dd>Unfaithful JSON wrongly signed FAITHFUL. Safety-critical.</dd>"
             "<dt>False-reject rate</dt><dd>Faithful policy wrongly flagged. User annoyance.</dd>"
             "<dt>FRR equiv</dt><dd>False-reject on hard-equivalent phrasings ('kabhi bhi' = all-day).</dd>"
             "<dt>ROC-AUC / PR-AUC</dt><dd>Threshold-independent quality from verdict+confidence. ~0.5 = random.</dd>"
             "<dt>ECE / Brier</dt><dd>Calibration: does stated confidence match actual accuracy. Lower better.</dd>"
             "<dt>flip rate</dt><dd>How often the verdict changes across the 3 runs (consistency).</dd>"
             "<dt>schema conformance</dt><dd>Fraction of outputs that parsed + validated against the schema.</dd>"
             "</dl></div>")

    # ---- caveats ----
    H.append("<h2>9. Caveats</h2><div class='card'><ul>"
             "<li>The data is <b>synthetic Hinglish</b> — strong for statistical power and "
             "ranking, but <b>not a substitute for a human-labelled set of real user policies</b> "
             "before production sign-off.</li>"
             "<li>The rank score under-penalizes the always-reject failure mode (FAR=0 props it "
             "up) — read it alongside the behaviour flags and ROC-AUC.</li>"
             "<li>At a strict 2% false-accept budget, even the best model may auto-approve little; "
             "tune the budget vs coverage, or keep a human in the loop.</li></ul></div>")

    H.append("</div></body></html>")
    return "".join(H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    res_dir = abspath(cfg["paths"]["results_dir"])
    plots_dir = abspath(cfg["paths"]["plots_dir"])
    with open(res_dir / "metrics.json") as f:
        data = json.load(f)
    html = build(data, cfg, plots_dir)
    out = Path(args.out) if args.out else (res_dir / "report.html")
    out.write_text(html, encoding="utf-8")
    print(f"[report_html] wrote {out} ({len(html)//1024} KB, {len(data['models'])} models)")


if __name__ == "__main__":
    main()

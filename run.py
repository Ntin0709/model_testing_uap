"""Inference loop. For every enabled model and every repeat run:
  1. start vLLM (serve.py), 2. run a small SEQUENTIAL latency probe for clean
  p50/p95, 3. run the full eval set CONCURRENTLY (vLLM continuous batching) with the
  verdict schema enforced, 4. save raw verdicts + run metadata, 5. tear down + free GPU.

Raw outputs are written per (model, run) so score.py can re-score without re-running
inference. Each model is repeated `runs_per_model` times to measure variance.
Runtime stats (isolated latency, throughput, peak GPU mem) are written to the .meta.json.
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import jsonschema
import numpy as np

from dataset import load_eval
from serve import ModelServer
from utils import (load_config, abspath, write_jsonl, load_schema, read_text,
                   render_user_prompt, gpu_info, gpu_mem_used_mib, vllm_version,
                   setup_local_cache)


class GPUPeakMonitor:
    def __init__(self, interval=2.0):
        self.interval, self.peak, self._stop = interval, 0.0, threading.Event()
        self._t = None

    def _loop(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, gpu_mem_used_mib())
            self._stop.wait(self.interval)

    def __enter__(self):
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._t:
            self._t.join(timeout=5)


def build_request_body(cfg, schema):
    mode = cfg["vllm"].get("guided_mode", "response_format")
    if mode == "response_format":
        return {"response_format": {
            "type": "json_schema",
            "json_schema": {"name": "verdict", "schema": schema, "strict": True}}}
    if mode == "guided_json":
        return {"extra_body": {"guided_json": schema}}
    raise ValueError(f"unknown guided_mode {mode}")


def validate(schema, obj):
    try:
        jsonschema.validate(obj, schema)
        return True
    except Exception:
        return False


def one_call(cfg, client, model_name, row, schema, system_prompt, user_tmpl,
             req_kwargs, run_idx):
    user_prompt = render_user_prompt(user_tmpl, row["english"], row["json"])
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]
    t0 = time.perf_counter()
    err, raw_text, usage = None, None, {}
    try:
        resp = client.chat.completions.create(
            model=model_name, messages=messages,
            temperature=cfg["temperature"], seed=cfg["seed"],
            max_tokens=cfg["max_tokens"], **req_kwargs)
        raw_text = resp.choices[0].message.content
        if resp.usage:
            usage = {"prompt_tokens": resp.usage.prompt_tokens,
                     "completion_tokens": resp.usage.completion_tokens}
    except Exception as e:
        err = repr(e)
    latency = time.perf_counter() - t0

    parsed, schema_ok = None, False
    if raw_text is not None:
        try:
            parsed = json.loads(raw_text)
            schema_ok = validate(schema, parsed)
        except Exception:
            parsed = None
    comp = usage.get("completion_tokens") or 0
    return {
        "id": row["id"], "run": run_idx,
        "gold_verdict": row["gold_verdict"], "gold_mismatches": row["gold_mismatches"],
        "corruption_type": row["corruption_type"], "hard_negative": row["hard_negative"],
        "equivalence_hard": row.get("equivalence_hard", False),
        "raw_text": raw_text, "parsed": parsed, "schema_ok": schema_ok, "error": err,
        "latency_s": latency, "completion_tokens": comp,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "tokens_per_s": (comp / latency) if (comp and latency > 0) else 0.0,
    }


def run_one(cfg, client, model_cfg, eval_rows, schema, system_prompt, user_tmpl, run_idx):
    name = model_cfg["name"]
    req_kwargs = build_request_body(cfg, schema)
    call = lambda r: one_call(cfg, client, name, r, schema, system_prompt,
                              user_tmpl, req_kwargs, run_idx)

    # 1) isolated latency probe (sequential) for clean SLA numbers
    probe_n = min(cfg.get("latency_probe_n", 50), len(eval_rows))
    probe_lat, probe_tps = [], []
    for r in eval_rows[:probe_n]:
        res = call(r)
        if res["error"] is None:
            probe_lat.append(res["latency_s"])
            if res["tokens_per_s"]:
                probe_tps.append(res["tokens_per_s"])

    # 2) full eval set concurrently (vLLM batches in flight)
    conc = max(1, int(cfg.get("concurrency", 16)))
    t0 = time.perf_counter()
    results = [None] * len(eval_rows)
    with ThreadPoolExecutor(max_workers=conc) as ex:
        futs = {ex.submit(call, r): i for i, r in enumerate(eval_rows)}
        for fut, i in futs.items():
            results[i] = fut.result()
    wall = time.perf_counter() - t0

    # 3) guided-decoding intervention probe: re-ask a subset WITHOUT the schema
    # constraint; how often the free-form output fails the schema = how often guided
    # decoding had to intervene to make the output usable.
    ecfg = cfg.get("evaluation", {})
    if ecfg.get("measure_intervention", False):
        pn = min(int(ecfg.get("intervention_probe_n", 200)), len(eval_rows))
        for i in range(pn):
            ff = one_call(cfg, client, name, eval_rows[i], schema, system_prompt,
                          user_tmpl, {}, run_idx)   # {} => no guided decoding
            results[i]["freeform_schema_ok"] = bool(ff["schema_ok"])

    stats = {
        "latency_p50_s": float(np.percentile(probe_lat, 50)) if probe_lat else None,
        "latency_p95_s": float(np.percentile(probe_lat, 95)) if probe_lat else None,
        "tokens_per_s": float(np.mean(probe_tps)) if probe_tps else None,
        "throughput_req_s": (len(eval_rows) / wall) if wall > 0 else None,
        "concurrency": conc, "probe_n": probe_n, "bulk_wall_s": wall,
    }
    return results, stats


def main():
    ap = argparse.ArgumentParser(description="Run inference for all enabled models.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--only", default=None, help="comma-separated model names")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"[env] HF cache -> {setup_local_cache(cfg)}")
    only = set(args.only.split(",")) if args.only else None
    schema = load_schema(cfg)
    system_prompt = read_text(cfg["paths"]["prompt_system"])
    user_tmpl = read_text(cfg["paths"]["prompt_user_template"])
    eval_rows = load_eval(cfg)

    env = {"vllm_version": vllm_version(), **gpu_info(),
           "seed": cfg["seed"], "temperature": cfg["temperature"],
           "guided_mode": cfg["vllm"].get("guided_mode"),
           "guided_backend": cfg["vllm"].get("guided_backend"),
           "n_eval": len(eval_rows)}
    print(f"[env] {json.dumps(env)}")

    raw_dir = abspath(cfg["paths"]["raw_outputs_dir"])
    models = [m for m in cfg["models"] if m.get("enabled", False)
              and (only is None or m["name"] in only)]
    if not models:
        print("[run] no enabled models match; edit config.yaml or pass --only")
        return
    print(f"[run] {len(models)} model(s): {[m['name'] for m in models]}")

    for model_cfg in models:
        name = model_cfg["name"]
        for run_idx in range(cfg["runs_per_model"]):
            print(f"\n===== {name} | run {run_idx + 1}/{cfg['runs_per_model']} =====")
            t_start = time.time()
            try:
                with ModelServer(cfg, model_cfg) as server, GPUPeakMonitor() as mon:
                    client = server.client()
                    results, stats = run_one(cfg, client, model_cfg, eval_rows, schema,
                                             system_prompt, user_tmpl, run_idx)
                peak = mon.peak
            except Exception as e:
                print(f"[run] FAILED {name} run {run_idx}: {e!r}")
                continue
            wall = time.time() - t_start
            meta = {**env, "model": name, "hf_id": model_cfg["hf_id"],
                    "quant": model_cfg.get("quant", "none"), "run": run_idx,
                    "peak_gpu_mem_mib": peak, "wall_time_s": wall,
                    "n_examples": len(results), **stats}
            out_path = raw_dir / f"{name}__run{run_idx}.jsonl"
            write_jsonl(out_path, results)
            with open(str(out_path).replace(".jsonl", ".meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            ok = sum(r["schema_ok"] for r in results)
            print(f"[run] saved {out_path} | schema_ok {ok}/{len(results)} "
                  f"| p50 {stats['latency_p50_s']} s | {stats['throughput_req_s']:.1f} req/s "
                  f"| peak_gpu {peak:.0f} MiB")

    print("\n[run] done. Next: python score.py && python report.py")


if __name__ == "__main__":
    main()

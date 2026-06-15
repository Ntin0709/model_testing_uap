# Policy-Faithfulness Evaluation Harness

Evaluate small instruction-tuned LLMs (1B–8B) as a **verification gate** in a
payments system: given a user's spending policy in plain English plus a structured
JSON translation, the model must decide whether the JSON **faithfully and completely**
represents the English, and emit a strict-schema verdict (not prose).

Because this gate controls real money, the harness treats **catching unfaithful JSON**
as the positive class and weights **false-accepts** (unfaithful JSON wrongly signed)
heaviest in the ranking.

## Layout

| file | role |
|---|---|
| `config.yaml` | models, paths, dataset mix, thresholds, ranking weights — **edit this** |
| `prompts/system_prompt.txt`, `prompts/user_template.txt` | exact prompt template |
| `schema/verdict_schema.json` | the strict verdict schema enforced via guided decoding |
| `data/seed_pairs.jsonl` | faithful English↔JSON seeds (input to the synthesizer) |
| `dataset.py` | load labeled `eval.jsonl` **or** synthesize + corrupt (tagged per type) |
| `serve.py` | vLLM lifecycle: launch, health-wait, teardown, confirm GPU freed |
| `run.py` | inference loop: serve → infer (schema-enforced) → save raw verdicts → teardown |
| `score.py` | all metrics, **re-runnable on saved outputs** (no re-inference) |
| `report.py` | ranking table (CSV + markdown) + per-model plots |
| `download.py` | front-load HF checkpoints |
| `run_all.sh` | the whole pipeline |

## The verdict schema (what the model must emit)

```json
{ "verdict": "FAITHFUL | NOT_FAITHFUL | AMBIGUOUS",
  "confidence": 0.0,
  "interpretable": true,
  "mismatches": [ { "field": "", "issue": "" } ] }
```

Enforced server-side with vLLM guided decoding via OpenAI `response_format`
`{"type":"json_schema", ..., "strict": true}` (switch to `guided_json` in
`config.yaml → vllm.guided_mode`). `temperature=0`, `seed=42` are pinned.

## Quick start (GPU machine)

```bash
# 0. Python 3.10+ with a CUDA-capable GPU
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # vllm>=0.8.5 (serves the 2026 families)

# 1. (gated models) authenticate + accept licenses on HF first
huggingface-cli login          # needed for Llama, Gemma, Ministral

# 2. ONE COMMAND end-to-end (dataset -> download -> serve+infer -> score -> report).
#    Default = smoke config (1 open model, 60 examples, ~5 min). Validates serving.
bash run_all.sh                       # smoke first

# 3. then 2-3 models, then the full sweep:
bash run_all.sh config.yaml "qwen2.5-3b,qwen3-4b-2507,gemma4-e4b,smollm3-3b"
bash run_all.sh config.yaml           # full sweep (all enabled models)
```

`run_all.sh` keeps **model downloads local to the script** (`./hf_home`, not `~/.cache`),
writes a full **timestamped log** to `./logs/run_<ts>.log` (also streamed to your
terminal), and on success freezes exact versions to `requirements.lock.txt`. You can
still run the steps individually with `--config` if you prefer.

Or step by step:

```bash
python dataset.py          # writes data/eval.jsonl (synthesized) or loads yours
python download.py         # pre-pull enabled checkpoints
python run.py              # serve each enabled model, 3 runs each, save raw verdicts
python run.py --only qwen2.5-3b,llama3.2-3b    # subset
python score.py            # re-scoreable on outputs/raw/ without re-running models
python report.py           # outputs/results/summary.{csv,md} + outputs/plots/*.png
```

## Choosing / editing models

Edit the `models:` list in `config.yaml`. Toggle `enabled`. `quant` (each maps to
**distinct** vLLM args — 4-bit and 8-bit are not the same):

| quant | bits | needs | notes |
|---|---|---|---|
| `none` | 16 | — | full precision (bf16/fp16) |
| `bnb-4bit` | 4 | any checkpoint | on-the-fly bitsandbytes NF4 |
| `awq` | 4 | `*-AWQ` checkpoint | e.g. `Qwen/Qwen2.5-7B-Instruct-AWQ` |
| `gptq` | 4 **or** 8 | `*-GPTQ-Int4`/`*-GPTQ-Int8` | bit-width read from checkpoint |

`bnb-8bit` is intentionally rejected — vLLM's on-the-fly bitsandbytes is 4-bit only;
for real **8-bit** use a `*-GPTQ-Int8` checkpoint with `quant: gptq`. The default config
covers both: Qwen ships official AWQ (4-bit) and GPTQ-Int8 (8-bit); other families use
`bnb-4bit` for 4-bit with an 8-bit slot for a community `*-GPTQ-Int8`.

The **full requested candidate list** (Qwen2.5 1.5B/3B/7B, Llama-3.2 1B/3B,
Llama-3.1-8B, Gemma-2 2B/9B, Phi-3.5-mini, Ministral 8B) is **enabled by default** with
fp16 + quantized variants, plus the **2026 block** (Qwen3 1.7B/4B/8B, Gemma-3 1B/4B,
Phi-4-mini). This is a **large sweep** — validate with the smoke config first (below),
then trim rows you don't need. The 2026 families need `vllm>=0.8.5` (already the pin);
verify each `hf_id` on HF. Ministral-3B is disabled (weights may be API-only).

## Dataset: scale, generation, and what it can prove

Two artifacts:

1. **`data/policies_faithful.jsonl`** — a large, diverse corpus of **correct
   policy→JSON pairs** produced by `generate_pairs.py` (a "policy factory"). The
   structured policy is sampled first, the canonical JSON is derived from it, and the
   natural-language side is rendered with varied phrasing. **Language is configurable**
   (`dataset.language`): `hinglish` (default — romanized Hindi mixed with English, real
   Indian merchants, `hazaar`/`lakh`/`₹` amounts, `subah`/`raat` times), `english`, or
   `mixed`. Phrasing includes non-literal-but-equivalent forms ("kabhi bhi" / "din
   bhar" == `00:00–23:59`, "kahin bhi" == any merchant, "5 hazaar" == 5000). This *is*
   the policy-to-correct-JSON dataset, and it doubles as the faithful seed bank.

   Example Hinglish row (faithful, equivalence-hard):
   `"Mujhe har din groceries ke liye 5 hazaar tak kharch karne do, kahin bhi, kabhi bhi."`
   → `{"max_amount":5000,"currency":"INR","period":"daily","categories":["groceries"],"merchants":"any","time_window":{"start":"00:00","end":"23:59"}}`
2. **`data/eval.jsonl`** — the labeled eval set (`dataset.py`), built from the corpus:
   `target_size` examples (default **2000**), `faithful_fraction` faithful (default
   0.40), the rest corrupted with **balanced counts per corruption type**.

Corruption types (each tagged, detection measured per type): `wrong_amount`,
`dropped_category`, `added_merchant`, `dropped_merchant`, `overpermissive_merchants`
(allow-list → "any", the dangerous one), `widened_time_window`, `narrowed_time_window`,
`swapped_currency`, `wrong_period`, and `combined` (two single-field drifts at once).
Each has **subtle single-field micro-drift** variants (hard negatives). Faithful pairs
that use non-literal-but-equivalent phrasing are flagged `equivalence_hard`, and the
false-reject rate on that subset (`FRR_equiv`) is reported separately — it's where
naive models wrongly reject real users.

**Is this production-grade?** At 2000 examples (~1200 unfaithful) the false-accept
rate is estimable to ≈ ±2.8% (95% CI, worst case; tighter for low true rates), and
each corruption type has ~120 examples (~±9% CI per type). To tighten the
safety-critical false-accept bound to ±1%, raise `dataset.target_size` to ~6000 — the
corpus auto-grows to cover it. **Synthetic data gives you statistical power and
coverage, but it is not a substitute for a held-out, human-labeled set of real user
policies.** Before going to production, validate the top 1–2 models on real labeled
data — drop it in as `data/eval.jsonl` and set `dataset.synthesize: false`:

```json
{"id":"...","english":"...","json":{...},
 "gold_verdict":"FAITHFUL|NOT_FAITHFUL","gold_mismatches":[{"field":"","issue":""}],
 "corruption_type":"none|wrong_amount|...","hard_negative":false,"equivalence_hard":false}
```

### Scaling the inference cost
`run.py` sends `concurrency` (default 32) requests in flight so vLLM batches them —
2000 examples is seconds-to-low-minutes per run on a small model. A separate
sequential `latency_probe_n` pass gives clean isolated p50/p95 (the interactive-gate
SLA), while `throughput_req_s` reports batched throughput.

## Metrics (per model, mean ± std over 3 runs)

- **Precision / recall / F1** with positive = unfaithful.
- **False-reject rate** — faithful policies wrongly rejected (user annoyance), plus
  **FRR_equiv** on equivalence-hard ("any time" == full-window) faithful pairs.
- **False-accept rate** — unfaithful policies wrongly signed (**safety-critical**, weighted heaviest).
- **Detection rate per corruption type.**
- **Mismatch-field recall** — did it point at the right field.
- **Confidence calibration** — reliability curve + ECE.
- **Threshold sweep** — precision & false-accept vs the confidence floor (0.5→0.99);
  directly informs `thresholds.production_floor`.
- **Schema-conformance rate** — fraction of *guided* outputs that parsed + validated
  (non-conforming output is treated as a FLAG, never an accept).
- **Guided-decoding intervention rate** — a subset is re-asked **without** the schema
  constraint; the fraction whose **free-form** output fails the schema = how often
  guided decoding had to intervene to make the output usable (`measure_intervention`).
  Plus **unusable_rate** = guided output still invalid (should be ~0).
- **p50/p95 latency, tokens/sec, peak GPU memory.**

A non-conforming or `AMBIGUOUS` output is counted as *flagged* (not signed), the safe
interpretation for a gate.

## Statistical rigor (so you can actually decide)

Point estimates aren't enough to pick a production model. The harness adds:

- **95% confidence intervals** on F1, precision, recall, FAR, FRR, ROC-AUC, PR-AUC via
  a **cluster bootstrap** — it resamples *examples* (keeping their repeated runs
  together), so the interval reflects sampling uncertainty, not just run-to-run noise.
  Run-to-run noise is reported separately (`FAR_run_std`, `flip_rate`).
- **Threshold-independent quality**: **ROC-AUC** and **PR-AUC** on a continuous
  unfaithfulness score derived from `verdict + confidence` — ranks models regardless of
  where you set the decision threshold.
- **Significance testing**: **McNemar's** paired test of each model vs the best one on
  per-example correctness, **Holm-corrected** for multiple comparisons (`significance.md`).
  Tells you whether a FAR gap is real or noise.
- **FAR-budget operating point**: the lowest confidence floor whose false-accept rate
  meets `thresholds.far_budget`, and the **coverage** (share of faithful policies still
  auto-approved) you pay for it — this is the number you ship (`operating_points.csv`).
- **Calibration**: ECE, **MCE**, and **Brier score** plus the reliability curve.
- **Self-consistency**: verdict **flip-rate** across repeated runs (temp 0 isn't
  bit-identical) — a reliability signal for a gate.
- **Subgroup slices** (`slices.csv`): FAR/FRR/accuracy disaggregated by `lang`,
  `equivalence_hard`, `period`, `currency`, `n_categories`, `merchants_type` — finds
  where a model fails systematically (e.g. higher FRR on `equivalence_hard` or `any`-merchant).
- **Error analysis** (`errors_<model>.csv`): every consensus mistake, **false-accepts
  first**, with the Hinglish text, gold vs predicted fields, and how many runs flagged it.

## Reproducibility

Fixed seeds (`42`), `temperature=0`, pinned versions in `requirements.txt`, prompt in
separate files, the **exact vLLM version + GPU recorded into every result**, and all
**raw verdicts saved** under `outputs/raw/` so you can re-score (`score.py`) without
re-running inference. Each model runs 3× and the report shows variance — even at temp 0,
vLLM is not bit-identical across batches.

## Outputs

- `outputs/raw/<model>__run<k>.jsonl` + `.meta.json` — raw verdicts + run metadata
- `outputs/results/metrics.json` — full scored metrics
- `outputs/results/summary.csv` / `summary.md` — ranked table
- `outputs/results/detection_by_type.csv`
- `outputs/results/operating_points.csv` — FAR-budget confidence floor + coverage per model
- `outputs/results/significance.md` — McNemar vs best model (Holm-corrected)
- `outputs/results/slices.csv` — disaggregated subgroup metrics
- `outputs/results/errors_<model>.csv` — error analysis (false-accepts first)
- `outputs/plots/<model>_{reliability,threshold_sweep,detection}.png`

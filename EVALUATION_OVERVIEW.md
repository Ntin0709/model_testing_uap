# Policy-Faithfulness Model Evaluation — One-Pager

## What we're doing & why
In our payments system a user writes a spending policy in plain English (often **Hinglish**),
and a component translates it into structured JSON. Before that JSON is trusted to control real
money, a small LLM acts as a **verification gate**: it reads the *English policy + the JSON* and
returns a strict-schema verdict — **FAITHFUL / NOT_FAITHFUL / AMBIGUOUS** — with a confidence and
the specific mismatched fields. We are evaluating which small model is good enough to be that gate.

Because this gates money, the metric that matters most is **catching unfaithful JSON**. A
*false-accept* (signing off a wrong JSON) is the safety-critical error; a *false-reject* (rejecting
a correct policy) only annoys users. We weight false-accepts heaviest.

## How we evaluate
- **Serving:** each model is brought up one at a time on **vLLM** (OpenAI-compatible endpoint),
  `temperature=0`, `seed=42`, the verdict **JSON schema enforced** via guided decoding. GPU is freed
  between models; vLLM version + GPU are recorded into every result.
- **Data:** a labelled set of `{english, json, gold_verdict}` examples. We generate a large, diverse
  **Hinglish** corpus of *correct* policy→JSON pairs, then programmatically corrupt the JSON one
  tagged way at a time (wrong amount, dropped/added/over-permissive merchants, widened/narrowed time
  window, swapped currency, wrong period, plus combined two-field drifts). Includes **hard negatives**
  (subtle single-field drifts) and **hard positives** (e.g. "kabhi bhi" == all-day) to stress both
  error types. Default ~2,000 examples. *(Synthetic gives statistical power; a human-labelled set is
  still required before production sign-off.)*
- **Each model runs 3×** to measure variance (temp 0 isn't bit-identical on vLLM).

## What we measure (per model)
- **Precision / Recall / F1** on catching unfaithful JSON, with **95% confidence intervals**
  (cluster bootstrap), and **McNemar significance** vs the best model.
- **False-accept rate** (safety-critical) and **false-reject rate** (incl. on hard-equivalent phrasings).
- **Detection rate per corruption type**; **mismatch-field accuracy**.
- **ROC-AUC / PR-AUC** (threshold-independent), **calibration** (ECE / MCE / Brier + reliability curve).
- **FAR-budget operating point:** the confidence floor that meets our false-accept budget, and the
  coverage we keep at it → the number we'd ship.
- **Schema-conformance / guided-decoding intervention rate**, **self-consistency** (verdict flip rate).
- **p50/p95 latency, tokens/sec, peak GPU memory.**

Output: a ranked CSV + rendered table, per-model plots, subgroup slices, and an error-analysis CSV
(false-accepts first). All raw verdicts are saved so results can be re-scored without re-running.

## Models under evaluation (1B–8B, instruction-tuned)
Each tested at **fp16** and, where available, **4-bit** and **8-bit** quantized variants.

| Family | Sizes | Quant coverage |
|---|---|---|
| **Qwen2.5-Instruct** | 1.5B, 3B, 7B | fp16 + AWQ (4-bit) + GPTQ-Int8 (8-bit) |
| **Qwen3** | 1.7B, 4B, **4B-Instruct-2507** | fp16 |
| **Llama-3.2 / 3.1-Instruct** | 1B, 3B, 8B | fp16 + bnb-4bit |
| **Gemma-2-it** | 2B, 9B | fp16 + bnb-4bit |
| **Gemma-3-it** | 1B, 4B | fp16 |
| **Gemma-4-it** (2026) | E2B, E4B | fp16 |
| **Phi-3.5-mini / Phi-4-mini** | 3.8B | fp16 + bnb-4bit |
| **Ministral** | 8B (3B optional) | fp16 + bnb-4bit |
| **SmolLM3** | 3B | fp16 |

Validated first on a 1-model **smoke test**, then a 3–4 model subset, then the full sweep.
Model list is editable in `config.yaml` (no code changes needed).

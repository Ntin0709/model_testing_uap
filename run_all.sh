#!/usr/bin/env bash
# ONE COMMAND: dataset -> download -> serve+infer -> score -> report.
#
# Usage:
#   bash run_all.sh                                   # smoke config (1 model) — safe first run
#   bash run_all.sh config.yaml                       # full sweep (all enabled models)
#   bash run_all.sh config.yaml "qwen2.5-3b,qwen3-4b-2507,gemma4-e4b,smollm3-3b"   # subset
#
# - Model checkpoints download into ./hf_home (next to this script), NOT ~/.cache.
# - Everything is logged to ./logs/run_<timestamp>.log (and streamed to your terminal).
# - After a successful run, exact versions are frozen to requirements.lock.txt.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

CONFIG="${1:-config.smoke.yaml}"
ONLY="${2:-}"

# keep all model downloads + caches local to this directory
export HF_HOME="$DIR/hf_home"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME"
export VLLM_CACHE_ROOT="$DIR/.vllm_cache"
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" logs

TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/run_${TS}.log"

ONLY_ARG=()
[ -n "$ONLY" ] && ONLY_ARG=(--only "$ONLY")

run_step () { echo; echo "==== $* ===="; }

{
  echo "=== pipeline start $TS | config=$CONFIG | models=${ONLY:-<all enabled>} ==="
  echo "=== HF_HOME=$HF_HOME ==="
  python --version || true
  python -c "import vllm; print('vllm', vllm.__version__)" 2>/dev/null || echo "vllm not importable yet"

  run_step "1/5 build dataset"
  python dataset.py  --config "$CONFIG"
  run_step "2/5 download checkpoints"
  python download.py --config "$CONFIG" "${ONLY_ARG[@]}"
  run_step "3/5 serve + infer"
  python run.py      --config "$CONFIG" "${ONLY_ARG[@]}"
  run_step "4/5 score"
  python score.py    --config "$CONFIG"
  run_step "5/5 report"
  python report.py   --config "$CONFIG"

  run_step "freeze exact versions"
  if pip freeze > requirements.lock.txt 2>/dev/null; then
    echo "wrote requirements.lock.txt ($(wc -l < requirements.lock.txt) pkgs)"
  else
    echo "pip freeze skipped"
  fi
  echo "=== pipeline done $(date +%H:%M:%S) ==="
} 2>&1 | tee "$LOG"

echo "Full log: $DIR/$LOG"

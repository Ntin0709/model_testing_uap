"""Shared helpers: config loading, paths, JSON IO, prompt rendering."""
import json
import os
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


def load_config(path: str = "config.yaml") -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with open(p) as f:
        return yaml.safe_load(f)


def abspath(rel: str) -> Path:
    """Resolve a config-relative path against the repo root."""
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


def read_jsonl(path) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_text(path) -> str:
    with open(abspath(str(path))) as f:
        return f.read()


def load_schema(cfg: dict) -> dict:
    with open(abspath(cfg["paths"]["verdict_schema"])) as f:
        return json.load(f)


def setup_local_cache(cfg: dict) -> str:
    """Point HF + vLLM downloads/caches at a directory next to the script (default
    ./hf_home), so checkpoints land where the harness runs rather than in ~/.cache.
    Uses setdefault so an env var exported by run_all.sh still wins."""
    root = abspath(cfg.get("paths", {}).get("hf_home", "hf_home"))
    root.mkdir(parents=True, exist_ok=True)
    hub = root / "hub"
    os.environ.setdefault("HF_HOME", str(root))
    os.environ.setdefault("HF_HUB_CACHE", str(hub))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(root))
    os.environ.setdefault("VLLM_CACHE_ROOT", str(abspath(".vllm_cache")))
    return str(root)


def render_user_prompt(template: str, english: str, json_obj) -> str:
    json_str = json.dumps(json_obj, ensure_ascii=False, indent=2) if not isinstance(json_obj, str) else json_obj
    return template.replace("{english}", english).replace("{json}", json_str)


def gpu_info() -> dict:
    """Best-effort GPU + driver snapshot via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"], text=True, timeout=15).strip()
        name, mem, drv = [x.strip() for x in out.splitlines()[0].split(",")]
        return {"gpu": name, "gpu_memory_total": mem, "driver_version": drv}
    except Exception as e:
        return {"gpu": "unknown", "gpu_memory_total": "unknown",
                "driver_version": "unknown", "error": str(e)}


def gpu_mem_used_mib() -> float:
    """GPU memory used (MiB) on the GPUs VISIBLE to this process. Respects
    CUDA_VISIBLE_DEVICES so on a shared node we measure only our GPU(s), not the
    whole box (otherwise the teardown 'freed' check never passes)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"], text=True, timeout=15).strip()
        used = {}
        for line in out.splitlines():
            idx, mem = line.split(",")
            used[int(idx)] = int(mem)
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        toks = [t.strip() for t in cvd.split(",")] if cvd else []
        if toks and all(t.isdigit() for t in toks):     # numeric device list -> filter
            return float(sum(used.get(int(t), 0) for t in toks))
        return float(sum(used.values()))                # UUIDs or unset -> all visible
    except Exception:
        return -1.0


def vllm_version() -> str:
    try:
        import vllm
        return vllm.__version__
    except Exception:
        try:
            return subprocess.check_output(["vllm", "--version"], text=True).strip()
        except Exception:
            return "unknown"

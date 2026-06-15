"""Pre-download HF checkpoints for the enabled models so the first vLLM serve
doesn't stall on a cold download. (vLLM would download on demand too; this just
front-loads it and surfaces gated-repo / auth errors early.)

Gated repos (Llama, Gemma, Ministral): run `huggingface-cli login` and accept the
license on each model's HF page first.
"""
import argparse

from utils import load_config, setup_local_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--all", action="store_true",
                    help="download every model in config, not just enabled ones")
    ap.add_argument("--only", default=None, help="comma-separated model names to fetch")
    args = ap.parse_args()
    cfg = load_config(args.config)
    print(f"[download] HF cache -> {setup_local_cache(cfg)}")

    from huggingface_hub import snapshot_download
    seen = set()
    only = set(args.only.split(",")) if args.only else None
    models = cfg["models"] if args.all else [m for m in cfg["models"] if m.get("enabled")]
    if only:
        models = [m for m in models if m["name"] in only]
    ids = [m["hf_id"] for m in models]
    ids = [i for i in ids if not (i in seen or seen.add(i))]
    print(f"[download] {len(ids)} unique checkpoint(s): {ids}")
    for hf_id in ids:
        print(f"[download] {hf_id} ...")
        try:
            path = snapshot_download(repo_id=hf_id)
            print(f"[download]   -> {path}")
        except Exception as e:
            print(f"[download]   FAILED {hf_id}: {e!r}\n"
                  f"            (gated? run `huggingface-cli login` and accept the license)")


if __name__ == "__main__":
    main()

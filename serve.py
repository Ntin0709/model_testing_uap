"""vLLM lifecycle: bring one model up on an OpenAI-compatible endpoint, wait for
health, hand back a client, then tear it down and confirm GPU memory is freed.

Usage as a context manager:

    with ModelServer(cfg, model_cfg) as server:
        client = server.client()       # openai.OpenAI pointed at the local endpoint
        ... run inference ...
    # GPU freed here
"""
import os
import signal
import subprocess
import time
import urllib.request

from utils import gpu_mem_used_mib

# Each quant maps to DISTINCT vLLM args. 4-bit vs 8-bit are not the same thing:
#   bnb-4bit : on-the-fly bitsandbytes NF4 (4-bit), works on any HF checkpoint.
#   awq      : 4-bit, needs an AWQ checkpoint (e.g. *-AWQ).
#   gptq     : INT4 *or* INT8 — vLLM reads the bit-width from the checkpoint's
#              quantize_config, so point hf_id at *-GPTQ-Int4 (4-bit) or *-GPTQ-Int8 (8-bit).
# There is no reliable on-the-fly bitsandbytes 8-bit path in vLLM, so bnb-8bit is
# rejected with guidance rather than silently behaving like 4-bit.
QUANT_ARGS = {
    "none": [],
    "bnb-4bit": ["--quantization", "bitsandbytes", "--load-format", "bitsandbytes"],
    "awq": ["--quantization", "awq_marlin"],
    "gptq": ["--quantization", "gptq_marlin"],
}


class ModelServer:
    def __init__(self, cfg: dict, model_cfg: dict):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.vcfg = cfg["vllm"]
        self.host = self.vcfg["host"]
        self.port = self.vcfg["port"]
        self.served_name = model_cfg["name"]
        self.proc = None
        self.log_file = None
        self.base_url = f"http://{self.host}:{self.port}/v1"

    # -- build the `vllm serve ...` command ---------------------------------
    def _command(self) -> list:
        quant = self.model_cfg.get("quant", "none")
        if quant == "bnb-8bit":
            raise ValueError(
                f"{self.served_name}: quant 'bnb-8bit' is not supported (vLLM on-the-fly "
                "bitsandbytes is 4-bit only). For 8-bit, set quant: gptq and point hf_id at "
                "an Int8 checkpoint, e.g. Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8.")
        if quant not in QUANT_ARGS:
            raise ValueError(
                f"unknown quant '{quant}' for {self.served_name}; "
                f"choose one of {sorted(QUANT_ARGS)} or gptq-Int8 checkpoint.")
        cmd = [
            "vllm", "serve", self.model_cfg["hf_id"],
            "--served-model-name", self.served_name,
            "--host", self.host, "--port", str(self.port),
            "--seed", str(self.cfg["seed"]),
            "--max-model-len", str(self.vcfg["max_model_len"]),
            "--gpu-memory-utilization", str(self.vcfg["gpu_memory_utilization"]),
            "--dtype", str(self.vcfg["dtype"]),
        ]
        cmd += QUANT_ARGS[quant]
        # guided_backend is opt-in: the flag was removed in vLLM 0.18 (xgrammar is the
        # default). Leave config blank on >=0.18; set it only on older vLLM that needs it.
        backend = self.vcfg.get("guided_backend")
        if backend:
            cmd += ["--guided-decoding-backend", backend]
        cmd += [str(x) for x in self.vcfg.get("extra_serve_args", [])]
        # per-model extra flags (e.g. Ministral needs --tokenizer-mode mistral ...)
        cmd += [str(x) for x in self.model_cfg.get("serve_args", [])]
        return cmd

    def _is_healthy(self) -> bool:
        # Verify OUR model is actually served (not just that /health is 200 — a
        # foreign service on the same port can answer /health and 404 the chat route).
        try:
            url = f"http://{self.host}:{self.port}/v1/models"
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status != 200:
                    return False
                return self.served_name in r.read().decode("utf-8", "ignore")
        except Exception:
            return False

    # -- lifecycle ----------------------------------------------------------
    def start(self):
        cmd = self._command()
        print(f"[serve] launching: {' '.join(cmd)}")
        os.makedirs("outputs", exist_ok=True)
        self.log_file = open(f"outputs/vllm_{self.served_name}.log", "w")
        self.proc = subprocess.Popen(
            cmd, stdout=self.log_file, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,  # own process group, so we can kill children
        )
        deadline = time.time() + self.vcfg["startup_timeout_s"]
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"[serve] vLLM exited early (code {self.proc.returncode}); "
                    f"see outputs/vllm_{self.served_name}.log")
            if self._is_healthy():
                print(f"[serve] {self.served_name} healthy at {self.base_url}")
                return self
            time.sleep(3)
        self.stop()
        raise TimeoutError(f"[serve] {self.served_name} not healthy within "
                           f"{self.vcfg['startup_timeout_s']}s")

    def client(self):
        from openai import OpenAI
        return OpenAI(base_url=self.base_url, api_key="EMPTY", timeout=120, max_retries=2)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            print(f"[serve] stopping {self.served_name} ...")
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=self.vcfg["shutdown_grace_s"])
            except subprocess.TimeoutExpired:
                print("[serve] SIGTERM grace expired; sending SIGKILL")
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.proc.wait(timeout=30)
        if self.log_file:
            self.log_file.close()
            self.log_file = None
        self._wait_gpu_freed()

    def _wait_gpu_freed(self, timeout_s: int = 120):
        target = self.vcfg.get("gpu_free_target_mib", 2000)
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            used = gpu_mem_used_mib()
            if used < 0 or used <= target:
                print(f"[serve] GPU memory freed (used={used:.0f} MiB)")
                return
            time.sleep(3)
        print(f"[serve] WARNING: GPU still using {gpu_mem_used_mib():.0f} MiB "
              f"after teardown (target {target}); check for orphan processes")

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False

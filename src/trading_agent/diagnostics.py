"""Honest runtime probes and bounded CPU/CUDA workload comparisons."""

from __future__ import annotations

from datetime import datetime, timezone
import platform
import subprocess
import time

import torch

from .config import make_env
from .evaluation import evaluate_policy
from .workflow import build_trainer, load_market, save_json, versions


def doctor() -> dict:
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "kernel": platform.release(),
        "platform": platform.platform(), "dependencies": versions(),
        "torch_cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "cuda_verified": False, "device": "cpu", "paper_only": True,
    }
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=15, check=False)
        report["nvidia_smi"] = {"exit_code": result.returncode, "output": result.stdout.strip() or result.stderr.strip()}
    except (FileNotFoundError, subprocess.TimeoutExpired):
        report["nvidia_smi"] = {"exit_code": None, "output": "nvidia-smi unavailable or timed out"}
    report["cpu_tensor_check"] = float(torch.arange(4.0).square().sum())
    if report["cuda_available"]:
        try:
            torch.cuda.reset_peak_memory_stats()
            tensor = torch.arange(256.0, device="cuda", requires_grad=True)
            loss = tensor.square().mean()
            loss.backward()
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            report.update(
                cuda_verified=True, device=torch.cuda.get_device_name(0),
                compute_capability=torch.cuda.get_device_capability(0),
                cuda_tensor_result=float(loss.item()), memory_free_bytes=free,
                memory_total_bytes=total, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            )
        except RuntimeError as error:
            report["cuda_error"] = str(error)
    if not report["cuda_verified"]:
        report["blocker"] = "CUDA tensor execution unavailable; see docs/gpu_repair.md. No GPU training is claimed."
    return report


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if doctor()["cuda_verified"] else "cpu"
    if requested == "cuda" and not doctor()["cuda_verified"]:
        raise ValueError("CUDA explicitly requested but not verified. Repair the driver; see docs/gpu_repair.md")
    if requested not in ("cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    return requested


def benchmark(data, config, output, *, steps=500) -> dict:
    if not 1 <= steps <= 10000:
        raise ValueError("benchmark steps must be in [1,10000]")
    bars = load_market(data, config)
    boundary = len(bars) // 2
    train, test = bars[:boundary], bars[-max(2, len(bars) // 4):]
    probe = doctor()
    results = []
    for device in (["cpu", "cuda"] if probe["cuda_verified"] else ["cpu"]):
        trainer, env = build_trainer(train, config, seed=0, device=device)
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.perf_counter()
        trainer.train(env, steps)
        if device == "cuda":
            torch.cuda.synchronize()
        duration = time.perf_counter() - start
        evaluation = evaluate_policy(make_env(test, config), trainer.act)
        results.append({
            "device": device, "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "seconds": duration, "steps": steps, "seed": 0,
            "configuration": trainer.config.__dict__ if hasattr(trainer.config, "__dict__") else {
                "hidden_sizes": trainer.config.hidden_sizes, "batch_size": trainer.config.batch_size,
                "replay_capacity": trainer.config.replay_capacity, "train_frequency": trainer.config.train_frequency,
            },
            "peak_vram_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
            "evaluation": evaluation.metrics.to_dict(),
        })
    report = {"runtime": probe, "results": results, "cuda_comparison": "measured" if probe["cuda_verified"] else "BLOCKED: no usable driver/device", "provenance": sorted({bar.source for bar in bars})}
    save_json(output, report)
    return report

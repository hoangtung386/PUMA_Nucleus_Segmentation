from __future__ import annotations

from dataclasses import dataclass, replace
import copy
import os
from typing import Any, Iterable

from puma.config import RuntimeConfig, Stage2ModelConfig
from puma.training.stage2_v132 import train_stage2_experiment_v132
from puma.utils import release_cuda_memory


@dataclass(frozen=True, slots=True)
class Stage2V132Job:
    config: Stage2ModelConfig
    seed: int


def build_v132_jobs(
    runtime: RuntimeConfig,
    configs: Iterable[Stage2ModelConfig],
) -> tuple[Stage2V132Job, ...]:
    return tuple(
        Stage2V132Job(config, int(seed))
        for config in configs
        for seed in runtime.training.seeds
    )


def _is_cuda_oom(error: BaseException | str) -> bool:
    text = str(error).lower()
    return any(token in text for token in (
        "cuda out of memory", "outofmemoryerror", "cuda error: out of memory"
    ))


def _run_once(
    runtime: RuntimeConfig,
    job: Stage2V132Job,
    hf_token: str | None,
) -> dict[str, Any]:
    try:
        return train_stage2_experiment_v132(
            runtime, job.config, job.seed, hf_token=hf_token
        )
    except Exception as exc:
        import traceback
        print(
            f"V13.2 failed [{job.config.name}/seed{job.seed}]: "
            f"{type(exc).__name__}: {exc}"
        )
        return {
            "status": "failed",
            "experiment": job.config.name,
            "seed": job.seed,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "cuda_oom": _is_cuda_oom(exc),
            "traceback": traceback.format_exc()[-8000:],
        }
    finally:
        release_cuda_memory()


def run_v132_job_with_oom_fallback(
    runtime: RuntimeConfig,
    job: Stage2V132Job,
    hf_token: str | None,
) -> dict[str, Any]:
    output = _run_once(runtime, job, hf_token)
    if output.get("status") != "failed" or not output.get("cuda_oom"):
        return output
    if os.environ.get("PUMA_V132_AUTO_OOM_FALLBACK", "1").strip().lower() in {
        "0", "false", "no", "off"
    }:
        return output

    effective = int(runtime.training.stage2_effective_batch_size)
    initial = int(runtime.training.stage2_micro_batch_size)
    candidates = [
        size for size in (128, 64, 32, 16)
        if size < initial and effective % size == 0
    ]
    last = output
    for micro in candidates:
        release_cuda_memory()
        retry_runtime = copy.deepcopy(runtime)
        retry_runtime.training.stage2_micro_batch_size = micro
        retry_cfg = replace(
            job.config,
            encoder_micro_batch_size=min(int(job.config.encoder_micro_batch_size), micro),
        )
        print(
            f"V13.2 OOM fallback [{job.config.name}/seed{job.seed}]: "
            f"Stage2/UNI2 micro={micro}/{retry_cfg.encoder_micro_batch_size}; "
            f"effective batch remains {effective}."
        )
        last = _run_once(
            retry_runtime, Stage2V132Job(retry_cfg, job.seed), hf_token
        )
        if last.get("status") != "failed" or not last.get("cuda_oom"):
            if last.get("status") in {"completed", "skipped"}:
                last = {
                    **last,
                    "oom_fallback_used": True,
                    "fallback_stage2_micro_batch_size": micro,
                    "fallback_encoder_micro_batch_size": retry_cfg.encoder_micro_batch_size,
                }
            return last
    return last


def run_v132_jobs(
    runtime: RuntimeConfig,
    configs: Iterable[Stage2ModelConfig],
    *,
    hf_token: str | None = None,
) -> list[dict[str, Any]]:
    """Run jobs sequentially on one accelerator.

    Parallel model training on a single Colab GPU is slower and far more likely to OOM.
    This queue keeps the GPU saturated per experiment and releases all CUDA memory between
    experiments. Multi-GPU orchestration should launch one notebook/process per GPU.
    """
    outputs: list[dict[str, Any]] = []
    jobs = build_v132_jobs(runtime, tuple(configs))
    for index, job in enumerate(jobs, 1):
        print(f"V13.2 job {index}/{len(jobs)}: {job.config.name}, seed={job.seed}")
        outputs.append(run_v132_job_with_oom_fallback(runtime, job, hf_token))
    return outputs

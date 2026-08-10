from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import copy
import multiprocessing as mp
import os
from typing import Any, Iterable

from puma.config import RuntimeConfig, Stage2ModelConfig
from puma.training.stage2_v13 import train_stage2_experiment_v13
from puma.utils import release_cuda_memory


@dataclass(frozen=True, slots=True)
class Stage2V13Job:
    config: Stage2ModelConfig
    seed: int


def build_v13_jobs(
    runtime: RuntimeConfig,
    configs: Iterable[Stage2ModelConfig],
) -> tuple[Stage2V13Job, ...]:
    return tuple(
        Stage2V13Job(config, int(seed))
        for config in configs
        for seed in runtime.training.seeds
    )


def _is_cuda_oom(error: BaseException | str) -> bool:
    text = str(error).lower()
    return (
        "cuda out of memory" in text
        or "outofmemoryerror" in text
        or "cuda error: out of memory" in text
    )


def _run_job(
    runtime: RuntimeConfig,
    job: Stage2V13Job,
    hf_token: str | None,
) -> dict[str, Any]:
    try:
        return train_stage2_experiment_v13(
            runtime,
            job.config,
            job.seed,
            hf_token=hf_token,
        )
    except Exception as exc:
        import traceback

        print(
            f"V13 failed [{job.config.name}/seed{job.seed}]: "
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


def _run_job_with_oom_fallback(
    runtime: RuntimeConfig,
    job: Stage2V13Job,
    hf_token: str | None,
) -> dict[str, Any]:
    output = _run_job(runtime, job, hf_token)
    if output.get("status") != "failed" or not output.get("cuda_oom"):
        return output
    if os.environ.get("PUMA_V13_AUTO_OOM_FALLBACK", "1").strip().lower() in {
        "0", "false", "no", "off"
    }:
        return output

    effective_batch = int(runtime.training.effective_batch_size)
    initial_micro_batch = int(runtime.training.stage2_micro_batch_size)
    fallbacks = [
        value
        for value in (128, 64, 32)
        if value < initial_micro_batch and effective_batch % value == 0
    ]
    last = output
    for micro_batch in fallbacks:
        release_cuda_memory()
        retry_runtime = copy.deepcopy(runtime)
        retry_runtime.training.stage2_micro_batch_size = micro_batch
        retry_config = replace(
            job.config,
            encoder_micro_batch_size=min(job.config.encoder_micro_batch_size, micro_batch),
        )
        print(
            f"V13 OOM fallback [{job.config.name}/seed{job.seed}]: "
            f"Stage2/UNI2 micro-batch {micro_batch}/{retry_config.encoder_micro_batch_size}; "
            f"effective batch remains {effective_batch}."
        )
        last = _run_job(
            retry_runtime,
            Stage2V13Job(retry_config, job.seed),
            hf_token,
        )
        if last.get("status") != "failed" or not last.get("cuda_oom"):
            if last.get("status") in {"completed", "skipped"}:
                last = {
                    **last,
                    "oom_fallback_used": True,
                    "fallback_stage2_micro_batch_size": micro_batch,
                    "fallback_encoder_micro_batch_size": retry_config.encoder_micro_batch_size,
                }
            return last
    return last


def _worker(payload: tuple[RuntimeConfig, Stage2V13Job, str | None, int]) -> dict[str, Any]:
    runtime, job, hf_token, cpu_threads = payload
    cpu_threads = max(1, int(cpu_threads))
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    os.environ["PUMA_STAGE2_WORKER"] = f"{job.config.name}:seed{job.seed}"
    try:
        import torch

        torch.set_num_threads(cpu_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except Exception:
        pass
    runtime.training.number_of_workers = min(
        int(runtime.training.number_of_workers), max(0, cpu_threads - 1)
    )
    return _run_job(runtime, job, hf_token)


def _run_pool(
    runtime: RuntimeConfig,
    jobs: tuple[Stage2V13Job, ...],
    *,
    hf_token: str | None,
    workers: int,
) -> list[dict[str, Any]]:
    if not jobs:
        return []
    workers = min(max(1, int(workers)), len(jobs))
    if workers == 1:
        return [_run_job_with_oom_fallback(runtime, job, hf_token) for job in jobs]

    cpu_threads = max(1, (os.cpu_count() or workers) // workers)
    payloads = tuple((runtime, job, hf_token, cpu_threads) for job in jobs)
    print(
        f"V13 queue: {len(jobs)} job(s), {workers} process(es), "
        f"CPU threads/job={cpu_threads}, CUDA_VISIBLE_DEVICES="
        f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<all>')}."
    )
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        outputs = list(executor.map(_worker, payloads))

    failed_oom = [
        index
        for index, output in enumerate(outputs)
        if output.get("status") == "failed" and output.get("cuda_oom")
    ]
    auto_fallback = os.environ.get("PUMA_V13_AUTO_OOM_FALLBACK", "1").strip().lower() not in {
        "0", "false", "no", "off"
    }
    if failed_oom and auto_fallback:
        print(f"Retrying {len(failed_oom)} OOM job(s) sequentially.")
        release_cuda_memory()
        for index in failed_oom:
            outputs[index] = _run_job_with_oom_fallback(runtime, jobs[index], hf_token)
    return outputs


def run_v13_jobs(
    runtime: RuntimeConfig,
    configs: Iterable[Stage2ModelConfig],
    *,
    hf_token: str | None,
    parallel_runs: int = 1,
    lora_parallel_runs: int = 1,
) -> list[dict[str, Any]]:
    configs = tuple(configs)
    frozen = tuple(config for config in configs if not config.use_lora)
    lora = tuple(config for config in configs if config.use_lora)
    outputs: list[dict[str, Any]] = []
    if frozen:
        outputs.extend(
            _run_pool(
                runtime,
                build_v13_jobs(runtime, frozen),
                hf_token=hf_token,
                workers=parallel_runs,
            )
        )
    if lora:
        outputs.extend(
            _run_pool(
                runtime,
                build_v13_jobs(runtime, lora),
                hf_token=hf_token,
                workers=lora_parallel_runs,
            )
        )
    return outputs

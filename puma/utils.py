from __future__ import annotations

import contextlib
import gc
import csv
import hashlib
import json
import os
import random
import shutil
import tempfile
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch


def configure_torch_runtime(*, deterministic: bool) -> None:
    """Apply hardware-level speed options without changing experiment hyperparameters.

    TF32 affects only eligible float32 CUDA matrix/conv kernels; AMP paths remain governed
    by the existing ``amp`` and ``prefer_bfloat16`` settings. Deterministic mode still
    disables cuDNN autotuning and requests deterministic algorithms.
    """
    with contextlib.suppress(Exception):
        torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        with contextlib.suppress(Exception):
            torch.backends.cuda.matmul.allow_tf32 = True
        with contextlib.suppress(Exception):
            torch.backends.cudnn.allow_tf32 = True
        # Let PyTorch select FlashAttention / memory-efficient SDPA kernels for the
        # transformer decoders and fusion encoder whenever the installed GPU supports them.
        with contextlib.suppress(Exception):
            torch.backends.cuda.enable_flash_sdp(True)
        with contextlib.suppress(Exception):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        with contextlib.suppress(Exception):
            torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = bool(deterministic)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    configure_torch_runtime(deterministic=deterministic)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _normalize_cuda_device_index(
    device: int | str | torch.device | None = None,
) -> int | None:
    """Resolve a CUDA device specification without assuming GPU 0."""
    if not torch.cuda.is_available():
        return None
    if device is None:
        return int(torch.cuda.current_device())
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        if device.type != "cuda":
            return None
        return int(
            torch.cuda.current_device() if device.index is None else device.index
        )
    return int(device)


def cuda_supports_native_bfloat16(
    device: int | str | torch.device | None = None,
) -> bool:
    """Return whether the selected CUDA GPU safely supports native BF16 training.

    Selection is based on the actual target device, not a GPU model-name allowlist. GPUs
    with CUDA compute capability below 8.0 use FP16. On capability 8.0 or newer, PyTorch
    must also report native BF16 support. This handles heterogeneous or non-default GPU
    indices correctly and avoids treating emulated BF16 as a safe convolution dtype.
    """
    try:
        index = _normalize_cuda_device_index(device)
        if index is None:
            return False
        major, _minor = torch.cuda.get_device_capability(index)
        if major < 8:
            return False
        with torch.cuda.device(index):
            try:
                supported = torch.cuda.is_bf16_supported(
                    including_emulation=False
                )
            except TypeError:
                # Compatibility with PyTorch versions that predate this keyword.
                supported = torch.cuda.is_bf16_supported()
        return bool(supported)
    except Exception:
        return False


def resolve_amp_dtype(
    prefer_bfloat16: bool = True,
    device: int | str | torch.device | None = None,
) -> torch.dtype:
    """Select FP32/FP16/BF16 for the actual execution device.

    CPU execution remains FP32. CUDA uses BF16 only when the selected GPU has native
    support and the user prefers it; every other CUDA GPU uses FP16.
    """
    index = _normalize_cuda_device_index(device)
    if index is None:
        return torch.float32
    if prefer_bfloat16 and cuda_supports_native_bfloat16(index):
        return torch.bfloat16
    return torch.float16




def release_cuda_memory(*, synchronize: bool = True) -> None:
    """Release Python objects and unused CUDA cache between independent runs.

    This does not change any training hyperparameter. It is intentionally called only
    at experiment/run boundaries, where allocator reuse is less valuable than avoiding
    fragmentation across different model architectures.
    """
    gc.collect()
    if not torch.cuda.is_available():
        return
    if synchronize:
        with contextlib.suppress(Exception):
            torch.cuda.synchronize()
    torch.cuda.empty_cache()
    # Helps Colab processes return inter-process CUDA allocations after a model is gone.
    with contextlib.suppress(Exception):
        torch.cuda.ipc_collect()


def build_adamw(
    parameters: Iterable[torch.nn.Parameter],
    *,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> torch.optim.AdamW:
    """Use the fused CUDA AdamW kernel when available, with safe fallbacks."""
    params = list(parameters)
    if device.type == "cuda":
        with contextlib.suppress(TypeError, RuntimeError, ValueError):
            return torch.optim.AdamW(
                params, lr=lr, weight_decay=weight_decay, fused=True
            )
        with contextlib.suppress(TypeError, RuntimeError, ValueError):
            return torch.optim.AdamW(
                params, lr=lr, weight_decay=weight_decay, foreach=True
            )
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def build_adamw_parameter_groups(
    parameter_groups: list[dict[str, Any]],
    *,
    device: torch.device,
) -> torch.optim.AdamW:
    """Build AdamW for explicit parameter groups using the fastest safe CUDA kernel."""
    if device.type == "cuda":
        with contextlib.suppress(TypeError, RuntimeError, ValueError):
            return torch.optim.AdamW(parameter_groups, fused=True)
        with contextlib.suppress(TypeError, RuntimeError, ValueError):
            return torch.optim.AdamW(parameter_groups, foreach=True)
    return torch.optim.AdamW(parameter_groups)


def clip_grad_norm_fast(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
) -> torch.Tensor:
    """Clip gradients with the foreach CUDA path when PyTorch supports it."""
    params = list(parameters)
    with contextlib.suppress(TypeError, RuntimeError):
        return torch.nn.utils.clip_grad_norm_(params, max_norm, foreach=True)
    return torch.nn.utils.clip_grad_norm_(params, max_norm)


def dataloader_performance_kwargs(workers: int, *, pin_memory: bool, persistent_workers: bool = False) -> dict[str, Any]:
    """Safe DataLoader throughput options without changing batch/sampling semantics."""
    resolved_workers = max(0, int(workers))
    kwargs: dict[str, Any] = {
        "num_workers": resolved_workers,
        "pin_memory": bool(pin_memory),
    }
    if resolved_workers > 0:
        kwargs["prefetch_factor"] = 2
        kwargs["persistent_workers"] = bool(persistent_workers)
    return kwargs

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_artifact_reference(
    value: Any,
    artifact_dirs: Path | Iterable[Path],
) -> Path | None:
    """Resolve an artifact strictly inside the configured canonical output folders.

    Saved CSV/lock paths may be absolute paths from an older project location.  We use
    only their basename and rebase into the current canonical directories.  This prevents
    deleted current output folders from silently reusing a legacy checkpoint elsewhere.
    """
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return None
    saved = Path(text).expanduser()
    if isinstance(artifact_dirs, (str, Path)):
        search_dirs = (Path(artifact_dirs),)
    else:
        search_dirs = tuple(Path(directory) for directory in artifact_dirs)
    if not search_dirs:
        return saved if saved.exists() else saved
    for directory in search_dirs:
        rebased = directory / saved.name
        if rebased.exists():
            return rebased
    # Return the canonical expected location even when missing so callers can report it.
    return search_dirs[0] / saved.name


def canonical_json(value: Any) -> str:
    def normalize(obj: Any) -> Any:
        if is_dataclass(obj):
            return normalize(asdict(obj))
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {str(k): normalize(v) for k, v in sorted(obj.items(), key=lambda x: str(x[0]))}
        if isinstance(obj, (tuple, list)):
            return [normalize(v) for v in obj]
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    return json.dumps(normalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(value: Any, length: int = 16) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(json.loads(canonical_json(value)), indent=2, ensure_ascii=False)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as temp:
        temp.write(payload)
        temp_path = Path(temp.name)
    os.replace(temp_path, path)


def atomic_save_numpy(path: Path, array: np.ndarray, allow_pickle: bool = False) -> None:
    """Atomically save an NPY, using local sequential staging for Colab Drive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = path.parent
    use_local_staging = (
        str(path).startswith("/content/drive/")
        and os.environ.get("PUMA_DISABLE_LOCAL_OUTPUT_STAGING", "").lower()
            not in {"1", "true", "yes"}
    )
    if use_local_staging:
        try:
            candidate = Path("/content/puma_output_staging")
            candidate.mkdir(parents=True, exist_ok=True)
            approximate_bytes = int(getattr(array, "nbytes", 0))
            if shutil.disk_usage(candidate).free > approximate_bytes + 512 * 1024**2:
                staging_dir = candidate
        except OSError:
            staging_dir = path.parent
    local_temp: Path | None = None
    destination_temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=staging_dir, suffix=".npy", delete=False
        ) as temp:
            np.save(temp, array, allow_pickle=allow_pickle)
            local_temp = Path(temp.name)
        if staging_dir == path.parent:
            os.replace(local_temp, path)
            local_temp = None
        else:
            with tempfile.NamedTemporaryFile(
                "wb", dir=path.parent, suffix=".npy", delete=False
            ) as destination:
                destination_temp = Path(destination.name)
            shutil.copyfile(local_temp, destination_temp)
            os.replace(destination_temp, path)
            destination_temp = None
    finally:
        for temporary in (local_temp, destination_temp):
            if temporary is not None:
                temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def simple_file_lock(lock_path: Path, timeout_seconds: float = 120.0) -> Iterator[None]:
    """Portable lock based on exclusive lock-file creation."""
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"pid={os.getpid()} time={utc_now_iso()}".encode("utf-8"))
        except FileExistsError:
            # CSV writes are short. Recover a lock left behind by a killed Colab process.
            with contextlib.suppress(FileNotFoundError, OSError):
                age_seconds = time.time() - lock_path.stat().st_mtime
                if age_seconds > max(600.0, 2.0 * timeout_seconds):
                    lock_path.unlink()
                    continue
            if time.time() - start > timeout_seconds:
                raise TimeoutError(f"Timed out waiting for lock: {lock_path}")
            time.sleep(0.2)
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        with contextlib.suppress(FileNotFoundError):
            lock_path.unlink()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def matching_csv_rows(path: Path, key_fields: dict[str, Any]) -> list[dict[str, str]]:
    """Return every row matching the experiment key, preserving CSV order."""
    expected = {k: str(v) for k, v in key_fields.items()}
    return [
        row
        for row in read_csv_rows(path)
        if all(row.get(key) == value for key, value in expected.items())
    ]


def latest_completed_csv_row(path: Path, key_fields: dict[str, Any]) -> dict[str, str] | None:
    """Return the most recent completed row for a resumable experiment."""
    completed = [row for row in matching_csv_rows(path, key_fields) if row.get("status") == "completed"]
    return completed[-1] if completed else None




def append_csv_row_atomic(path: Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    serialized = {
        key: (
            json.dumps(value, sort_keys=True, ensure_ascii=False)
            if isinstance(value, (dict, list, tuple))
            else value
        )
        for key, value in row.items()
    }
    with simple_file_lock(lock):
        existing = read_csv_rows(path)
        fieldnames: list[str] = []
        for current in existing + [{k: str(v) for k, v in serialized.items()}]:
            for key in current:
                if key not in fieldnames:
                    fieldnames.append(key)
        with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, delete=False) as temp:
            writer = csv.DictWriter(temp, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for current in existing:
                writer.writerow(current)
            writer.writerow(serialized)
            temp_path = Path(temp.name)
        os.replace(temp_path, path)




def save_best_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    score: float,
    config: Any,
    extra: dict[str, Any] | None = None,
    trainable_only: bool = False,
    include_training_state: bool = False,
) -> None:
    """Atomically save either a full or trainable-only checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if trainable_only:
        trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        state = {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
            if name in trainable_names
        }
        state_kind = "trainable_only"
    else:
        state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
        state_kind = "full"
    payload = {
        "model_state": state,
        "model_state_kind": state_kind,
        "optimizer_state": optimizer.state_dict() if include_training_state and optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if include_training_state and scheduler is not None else None,
        "scaler_state": scaler.state_dict() if include_training_state and scaler is not None else None,
        "epoch": int(epoch),
        "score": float(score),
        "config": json.loads(canonical_json(config)),
        "extra": extra or {},
        "saved_at": utc_now_iso(),
    }
    # Stage Colab checkpoints locally before one atomic copy to Drive.
    staging_dir = path.parent
    use_local_staging = (
        str(path).startswith("/content/drive/")
        and os.environ.get("PUMA_DISABLE_LOCAL_CHECKPOINT_STAGING", "").lower()
            not in {"1", "true", "yes"}
    )
    if use_local_staging:
        try:
            candidate = Path("/content/puma_checkpoint_staging")
            candidate.mkdir(parents=True, exist_ok=True)
            approximate_bytes = sum(
                tensor.numel() * tensor.element_size() for tensor in state.values()
            )
            if shutil.disk_usage(candidate).free > approximate_bytes + 512 * 1024**2:
                staging_dir = candidate
        except OSError:
            staging_dir = path.parent
    local_temp: Path | None = None
    destination_temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=staging_dir, suffix=".pt", delete=False
        ) as temp:
            torch.save(payload, temp.name, pickle_protocol=5)
            local_temp = Path(temp.name)
        if staging_dir == path.parent:
            os.replace(local_temp, path)
            local_temp = None
        else:
            with tempfile.NamedTemporaryFile(
                "wb", dir=path.parent, suffix=".pt", delete=False
            ) as destination:
                destination_temp = Path(destination.name)
            shutil.copyfile(local_temp, destination_temp)
            os.replace(destination_temp, path)
            destination_temp = None
    finally:
        for temporary in (local_temp, destination_temp):
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def restore_checkpoint_payload(
    payload: dict[str, Any], model: torch.nn.Module
) -> dict[str, Any]:
    """Restore an already CPU-loaded payload without another checkpoint disk read."""
    state_kind = payload.get("model_state_kind", "full")
    incompatible = model.load_state_dict(
        payload["model_state"], strict=state_kind != "trainable_only"
    )
    if state_kind == "trainable_only":
        trainable_names = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        missing_trainable = sorted(
            trainable_names.intersection(incompatible.missing_keys)
        )
        if missing_trainable or incompatible.unexpected_keys:
            raise RuntimeError(
                "Compact checkpoint is incompatible with this model: "
                f"missing trainable={missing_trainable}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
    return payload


def load_checkpoint(path: Path, model: torch.nn.Module, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    # Stage through CPU to avoid duplicating checkpoint tensors in VRAM.
    requested = torch.device(map_location) if not isinstance(map_location, str) or map_location != "cpu" else torch.device("cpu")
    load_location: str | torch.device = "cpu" if requested.type == "cuda" else map_location
    payload = torch.load(Path(path), map_location=load_location, weights_only=False)
    return restore_checkpoint_payload(payload, model)


def worker_seed_init(worker_id: int) -> None:
    # Keep each DataLoader worker single-threaded to limit host-memory pressure.
    with contextlib.suppress(Exception):
        torch.set_num_threads(1)
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def count_trainable_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def peak_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024**2))


def reset_peak_vram() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()




def rescale_partial_accumulation_gradients(
    parameters,
    *,
    accumulation_steps: int,
    microbatches_in_group: int,
) -> float:
    """Rescale the final partial accumulation group after AMP unscaling."""
    accumulation_steps = int(accumulation_steps)
    microbatches_in_group = int(microbatches_in_group)
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be >= 1")
    if microbatches_in_group < 1 or microbatches_in_group > accumulation_steps:
        raise ValueError(
            "microbatches_in_group must be in [1, accumulation_steps], got "
            f"{microbatches_in_group} for accumulation_steps={accumulation_steps}."
        )
    if microbatches_in_group == accumulation_steps:
        return 1.0
    factor = float(accumulation_steps) / float(microbatches_in_group)
    for parameter in parameters:
        gradient = getattr(parameter, "grad", None)
        if gradient is not None:
            gradient.mul_(factor)
    return factor

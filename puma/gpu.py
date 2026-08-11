"""GPU selection helpers.

This module must stay free of ``torch`` (and of anything importing it). ``CUDA_VISIBLE_DEVICES``
is only read when the CUDA driver is first initialised, so device selection has to happen
before ``torch`` is imported. Importing ``puma.gpu`` pulls in nothing heavier than
``subprocess``, which makes it safe to call from the first cell of a notebook.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Any

_TRUTHY = {"1", "true", "yes", "on"}

# nvidia-smi numbers GPUs by PCI bus id, while CUDA's default enumeration is
# CUDA_DEVICE_ORDER=FAST_FIRST, which can order them differently. Pinning the order makes
# "GPU 1" mean the same device in nvidia-smi, in CUDA_VISIBLE_DEVICES, and in torch.
DEVICE_ORDER = "PCI_BUS_ID"

_QUERY = ("index", "name", "memory.total")


def parse_gpu_inventory(text: str) -> list[dict[str, Any]]:
    """Parse ``nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader``."""
    inventory: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < len(_QUERY):
            continue
        index, name, memory = fields[0], fields[1], fields[2]
        if not index.isdigit():
            continue
        megabytes = "".join(character for character in memory if character.isdigit())
        inventory.append({
            "index": int(index),
            "name": name,
            "memory_mb": int(megabytes) if megabytes else None,
        })
    inventory.sort(key=lambda entry: entry["index"])
    return inventory


def query_gpu_inventory(*, timeout: float = 10.0) -> list[dict[str, Any]]:
    """List the GPUs nvidia-smi can see, or an empty list when it cannot be run.

    nvidia-smi is used rather than ``torch.cuda.device_count()`` on purpose: counting
    devices through torch initialises CUDA in this process and caches the count, after
    which setting ``CUDA_VISIBLE_DEVICES`` no longer changes anything.
    """
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return []
    try:
        completed = subprocess.run(
            [executable, f"--query-gpu={','.join(_QUERY)}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if completed.returncode != 0:
        return []
    return parse_gpu_inventory(completed.stdout)


def _cuda_is_initialised() -> bool:
    """True when torch has already brought up CUDA, making a device switch ineffective."""
    torch_module = sys.modules.get("torch")
    if torch_module is None:
        return False
    try:
        return bool(torch_module.cuda.is_initialized())
    except Exception:
        return False


def select_cuda_device(
    preferred_index: int = 1,
    *,
    inventory: list[dict[str, Any]] | None = None,
    environ: dict[str, str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Pin this process to one GPU, preferring ``preferred_index`` when it exists.

    On a workstation with two or more GPUs this selects ``preferred_index`` (GPU 1 by
    default), leaving GPU 0 free for the display and for other jobs. With a single GPU it
    falls back to GPU 0. After the call the selected device is the only one visible, so it
    is ``cuda:0`` to torch and every downstream ``resolve_device()`` uses it unchanged.

    An existing ``CUDA_VISIBLE_DEVICES`` is respected and left alone unless ``force`` is
    set, so launching with ``CUDA_VISIBLE_DEVICES=0 jupyter lab`` still overrides this.

    ``inventory`` and ``environ`` exist for testing; leave them unset in normal use.
    """
    environ = os.environ if environ is None else environ
    if preferred_index < 0:
        raise ValueError(f"preferred_index must be non-negative, got {preferred_index}")

    existing = environ.get("CUDA_VISIBLE_DEVICES")
    inventory = query_gpu_inventory() if inventory is None else inventory
    by_index = {int(entry["index"]): entry for entry in inventory}

    report: dict[str, Any] = {
        "detected": len(inventory),
        "devices": [
            f"{entry['index']}: {entry['name']}"
            + (f" ({entry['memory_mb'] / 1024:.0f} GB)" if entry.get("memory_mb") else "")
            for entry in inventory
        ],
        "preferred_index": preferred_index,
        "respected_existing": False,
        "cuda_already_initialised": _cuda_is_initialised(),
    }

    if existing is not None and existing.strip() != "" and not force:
        report["respected_existing"] = True
        report["cuda_visible_devices"] = existing
        first = existing.split(",")[0].strip()
        selected = int(first) if first.isdigit() else None
        report["selected_index"] = selected
        report["selected_name"] = (
            by_index.get(selected, {}).get("name") if selected is not None else None
        )
        report["reason"] = "CUDA_VISIBLE_DEVICES was already set; left unchanged"
        return report

    if not inventory:
        report["cuda_visible_devices"] = environ.get("CUDA_VISIBLE_DEVICES")
        report["selected_index"] = None
        report["selected_name"] = None
        report["reason"] = "no GPU reported by nvidia-smi; CUDA_VISIBLE_DEVICES left unset"
        return report

    if preferred_index in by_index:
        selected = preferred_index
        reason = f"{len(inventory)} GPUs detected; using preferred GPU {preferred_index}"
    else:
        selected = int(inventory[0]["index"])
        reason = (
            f"{len(inventory)} GPU(s) detected, so GPU {preferred_index} does not exist; "
            f"falling back to GPU {selected}"
        )

    environ["CUDA_DEVICE_ORDER"] = DEVICE_ORDER
    environ["CUDA_VISIBLE_DEVICES"] = str(selected)
    report["cuda_visible_devices"] = str(selected)
    report["cuda_device_order"] = DEVICE_ORDER
    report["selected_index"] = selected
    report["selected_name"] = by_index[selected]["name"]
    report["selected_memory_mb"] = by_index[selected].get("memory_mb")
    report["reason"] = reason
    return report


def describe_selection(report: dict[str, Any]) -> str:
    """One-line, printable summary of :func:`select_cuda_device`."""
    lines = [f"GPUs detected        : {report['detected']}"]
    for device in report["devices"]:
        lines.append(f"  {device}")
    lines.append(f"CUDA_VISIBLE_DEVICES : {report.get('cuda_visible_devices')}")
    if report.get("cuda_device_order"):
        lines.append(f"CUDA_DEVICE_ORDER    : {report['cuda_device_order']}")
    name = report.get("selected_name")
    index = report.get("selected_index")
    if index is not None:
        lines.append(
            f"training device      : physical GPU {index}"
            + (f" ({name})" if name else "")
            + "  -> cuda:0 inside torch"
        )
    lines.append(f"reason               : {report['reason']}")
    if report.get("cuda_already_initialised"):
        lines.append(
            "WARNING: torch had already initialised CUDA in this kernel, so the device "
            "selection above has NOT taken effect. Restart the kernel and Run All."
        )
    return "\n".join(lines)

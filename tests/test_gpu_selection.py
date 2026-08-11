"""Tests for GPU selection.

Run with the project virtualenv, from the project root:

    ./.venv/bin/python tests/test_gpu_selection.py

The multi-GPU cases use an injected inventory, so they run on a single-GPU machine.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puma.gpu import (  # noqa: E402
    DEVICE_ORDER,
    describe_selection,
    parse_gpu_inventory,
    query_gpu_inventory,
    select_cuda_device,
)

SMI_TWO = "0, NVIDIA GeForce RTX 3090 Ti, 24564 MiB\n1, NVIDIA GeForce RTX 3090 Ti, 24564 MiB\n"
SMI_ONE = "0, NVIDIA GeForce RTX 3080, 10240 MiB\n"
SMI_FOUR = "".join(f"{i}, NVIDIA A100-SXM4-40GB, 40960 MiB\n" for i in range(4))


def test_parses_nvidia_smi_csv() -> None:
    inventory = parse_gpu_inventory(SMI_TWO)
    assert [entry["index"] for entry in inventory] == [0, 1], inventory
    assert inventory[1]["name"] == "NVIDIA GeForce RTX 3090 Ti", inventory
    assert inventory[1]["memory_mb"] == 24564, inventory


def test_parser_ignores_noise_and_sorts() -> None:
    messy = "\n1, GPU B, 8192 MiB\n\nNo devices were found\n0, GPU A, 8192 MiB\n"
    inventory = parse_gpu_inventory(messy)
    assert [entry["index"] for entry in inventory] == [0, 1], inventory
    assert [entry["name"] for entry in inventory] == ["GPU A", "GPU B"], inventory


def test_two_gpus_selects_gpu_one() -> None:
    environ: dict[str, str] = {}
    report = select_cuda_device(1, inventory=parse_gpu_inventory(SMI_TWO), environ=environ)
    assert environ["CUDA_VISIBLE_DEVICES"] == "1", environ
    assert environ["CUDA_DEVICE_ORDER"] == DEVICE_ORDER, environ
    assert report["selected_index"] == 1, report
    assert report["detected"] == 2, report


def test_four_gpus_still_selects_gpu_one() -> None:
    environ: dict[str, str] = {}
    report = select_cuda_device(1, inventory=parse_gpu_inventory(SMI_FOUR), environ=environ)
    assert environ["CUDA_VISIBLE_DEVICES"] == "1", environ
    assert report["selected_index"] == 1, report


def test_single_gpu_falls_back_to_gpu_zero() -> None:
    environ: dict[str, str] = {}
    report = select_cuda_device(1, inventory=parse_gpu_inventory(SMI_ONE), environ=environ)
    assert environ["CUDA_VISIBLE_DEVICES"] == "0", environ
    assert report["selected_index"] == 0, report
    assert "falling back" in report["reason"], report


def test_no_gpu_leaves_environment_untouched() -> None:
    environ: dict[str, str] = {}
    report = select_cuda_device(1, inventory=[], environ=environ)
    assert "CUDA_VISIBLE_DEVICES" not in environ, environ
    assert report["selected_index"] is None, report
    assert report["detected"] == 0, report


def test_existing_selection_is_respected() -> None:
    environ = {"CUDA_VISIBLE_DEVICES": "0"}
    report = select_cuda_device(1, inventory=parse_gpu_inventory(SMI_TWO), environ=environ)
    assert environ["CUDA_VISIBLE_DEVICES"] == "0", environ
    assert "CUDA_DEVICE_ORDER" not in environ, environ
    assert report["respected_existing"] is True, report
    assert report["selected_index"] == 0, report


def test_empty_existing_selection_is_not_treated_as_a_choice() -> None:
    environ = {"CUDA_VISIBLE_DEVICES": "  "}
    select_cuda_device(1, inventory=parse_gpu_inventory(SMI_TWO), environ=environ)
    assert environ["CUDA_VISIBLE_DEVICES"] == "1", environ


def test_force_overrides_an_existing_selection() -> None:
    environ = {"CUDA_VISIBLE_DEVICES": "0"}
    report = select_cuda_device(
        1, inventory=parse_gpu_inventory(SMI_TWO), environ=environ, force=True
    )
    assert environ["CUDA_VISIBLE_DEVICES"] == "1", environ
    assert report["respected_existing"] is False, report


def test_preferred_index_zero_is_honoured() -> None:
    environ: dict[str, str] = {}
    select_cuda_device(0, inventory=parse_gpu_inventory(SMI_TWO), environ=environ)
    assert environ["CUDA_VISIBLE_DEVICES"] == "0", environ


def test_negative_preferred_index_is_rejected() -> None:
    try:
        select_cuda_device(-1, inventory=parse_gpu_inventory(SMI_TWO), environ={})
    except ValueError:
        return
    raise AssertionError("select_cuda_device accepted a negative preferred_index")


def test_describe_selection_mentions_the_physical_device() -> None:
    report = select_cuda_device(1, inventory=parse_gpu_inventory(SMI_TWO), environ={})
    text = describe_selection(report)
    assert "physical GPU 1" in text, text
    assert "cuda:0" in text, text


def test_module_does_not_import_torch() -> None:
    """Importing puma.gpu must not pull torch in, or the selection comes too late."""
    assert "torch" not in sys.modules, "torch was already imported before this test ran"


def test_live_machine_probe_matches_its_gpu_count() -> None:
    """Smoke test against the real nvidia-smi on whatever machine this runs on."""
    inventory = query_gpu_inventory()
    environ: dict[str, str] = {}
    report = select_cuda_device(1, inventory=inventory, environ=environ)
    assert report["detected"] == len(inventory), report
    if not inventory:
        assert report["selected_index"] is None, report
        return
    expected = 1 if len(inventory) >= 2 else 0
    assert report["selected_index"] == expected, (report, len(inventory))
    assert environ["CUDA_VISIBLE_DEVICES"] == str(expected), environ
    print(f"      (live probe: {len(inventory)} GPU(s) -> selected {expected})")


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {test.__name__}\n      {exc}")
        else:
            print(f"ok    {test.__name__}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

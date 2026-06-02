
import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_autocast_context(device: torch.device, dtype: torch.dtype | None = None):
    if dtype is None:
        if device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32)

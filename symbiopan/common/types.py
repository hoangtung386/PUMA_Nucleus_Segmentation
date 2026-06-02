from typing import TypedDict

import torch


class BatchDict(TypedDict, total=False):
    image: "torch.Tensor"
    tissue_sem: "torch.Tensor"
    nuclei_np: "torch.Tensor"
    nuclei_nc: "torch.Tensor"
    nuclei_hv: "torch.Tensor"
    site_id: "torch.Tensor"
    context_roi: "torch.Tensor"
    base_name: str
    source_name: str
    is_rare_augmented: bool

# Version 6: Ablation 1+2 with Smooth Stage 1

Architecture changes versus Version 5.2:
- Ablation 1: HoVer-NeXt-style NP/HV decoder heads.
- Ablation 2: ASPP tissue head.

Training change:
- Stage 1 now ramps FocalTversky, SC-DFA, and spatial prior smoothly instead of switching them on abruptly.

Schedule:
- FocalTversky: epoch 10 -> 16, max weight 0.5.
- SC-DFA: epoch 15 -> 22, max weight 0.3.
- Spatial prior: epoch 20 -> 28, max weight 0.2.
- Rare sampler is gentler: max sample weight 15.0, samples_per_epoch_multiplier 1.0.

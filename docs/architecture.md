# SymbioPan v9 (CellPath) — Model Architecture

## Legend

- `[B, C, H, W]` — tensor shape (batch, channels, height, width)
- `{ }` — dictionary / named tuple
- `---` — optional path

```mermaid
flowchart TD
    %% ─── INPUTS ───
    Input["Input Tensor<br/>[B, 3, 1024, 1024]"] --> Encoder

    subgraph Encoder["UnifiedPanopticEncoder"]
        direction TB
        CNN["ConvNeXt-Tiny<br/>timm features_only<br/>auto-built if cnn_model=None"] --> CNN_Feats["CNN Features<br/>C1: [B, 96, 256, 256]<br/>C2: [B, 192, 128, 128]<br/>C3: [B, 384, 64, 64]<br/>C4: [B, 768, 32, 32]"]

        ViT["Virchow2 ViT-H/14<br/>32 blocks, dim=1280<br/>patch_size from config<br/>fine-tune last 6 blocks"] --> ViT_Tokens["ViT Tokens<br/>[B, 5330, 1280]"]
        ViT --> Bridges

        Bridges["SpatialInjector ×4<br/>multi-head (h=8) Q/K/V proj<br/>target_grid=32<br/>after blocks 7,15,23,31"]
        CNN_Feats --> Bridges
        Bridges -->|"post-bridge tokens"| Intermediate["Intermediate (blocks 7,15,23,31)<br/>captured AFTER bridge<br/>reshape tokens → spatial<br/>[4, B, 1280, 73, 73]"]
    end

    %% ─── FPN ───
    Encoder --> FPN

    subgraph FPN["HierarchicalFPN"]
        direction TB
        Grid["Grid: H//patch_size × W//patch_size<br/>(not sqrt(n) guess)<br/>drops CLS/special tokens"] --> ViT_Proj["ViT Proj Conv2d 1×1<br/>1280 → 256"]
        Latent["Latent Conv2d 1×1<br/>CNN_dims → 256<br/>×4 levels → s1..s4"] --> P_CNN["[B, 256, H/4…H/32]"]
        Fusion["Top-down fusion<br/>ViT injected at p4 level"]
        P_CNN --> Fusion
        ViT_Proj --> Fusion
        Fusion --> FPN_Out["{p1–p5}<br/>[B, 256, H/2…H/32]<br/>+ low_level_feat<br/>[B, 96, H/4, H/4]"]

        P1_Fix["P1 = smooth1(s1↑ + p2↑)<br/>s1 from C1 upsampled 2×<br/>SKIP — preserves original encoder features"]
        P_CNN --> P1_Fix
        FPN_Out --> P1_Fix
    end

    %% ─── CONDITIONING ───
    Context["ContextROI<br/>[B, 3, 320, 320]"] -.-> CtxEnc["ContextEncoder<br/>EfficientNet-B0<br/>→ [B, 256] (global)"]
    CtxEnc -.-> CtxFusion["ContextFusion (FiLM)<br/>identity init (γ=0, β=0)<br/>(1+γ)·v + β"]
    SiteIDs["Site IDs<br/>[B]"] -.-> SiteEmb["nn.Embedding(9,256)<br/>+ site_proj (if dim mismatch)<br/>→ bias [B, FPNDim, 1, 1]"]
    CtxFusion -.-> FPN_Out
    SiteEmb -.-> FPN_Out

    %% ─── DECODERS ───
    FPN --> Decoders

    subgraph Decoders["ParallelDecoders"]
        direction TB
        Exch["MutualFeatureExchange<br/>depthwise gate on p3<br/>f_tissue, f_nuclei<br/>[B, 256, 128, 128]"]
        FPN_Out --> Exch

        subgraph Tissue["Tissue Branch"]
            TFuse["tissue_fuse Conv1×1<br/>cat[f_t, p4↑, p5↑, vit(7)↑, vit(15)↑, vit(23)↑, vit(31)↑]<br/>7×256 → 256<br/>← SKIP: vit_intermediate → tissue"]
            Intermediate -->|"4× proj 1280→256 + upsample"| TFuse
            Exch --> TFuse
            FPN_Out --> TFuse

            C2_Skip["C2 skip (additive)<br/>Conv1×1 192→256 + upsample<br/>→ ADD after tissue_fuse<br/>← SKIP: C2 → tissue decoder"]
            CNN_Feats -->|C2| C2_Skip
            TFuse --> C2_Skip

            ASPP["ASPP<br/>(rates 1,3,6,9 + pool)<br/>→ [B, 256, 128, 128]"]
            C2_Skip --> ASPP
            LowConv["low_level_conv<br/>[B,96]→[B,48]"]
            FPN_Out --> LowConv
            LowConv --> TissueHead["DeepLabV3PlusTissueHead<br/>upsample + fuse conv<br/>→ classifier Conv1×1"] --> TissueOut["Tissue Logits<br/>[B, 6, 512, 512]"]
            ASPP --> TissueHead
        end

        subgraph NucleiClass["Nuclei Classification Branch"]
            NC_Proj["ViT Projs ×4<br/>Conv2d 1×1, 1280→256"] --> NC_Fuse["Fuse Conv 3×3<br/>cat 4 levels → 1024→256"]
            Intermediate --> NC_Proj
            NC_Fuse --> NC_FPN_Fuse["FPN Fuse Conv 3×3<br/>cat[fused, p2, f_n↑]<br/>256+256+256 → 256<br/>← SKIP: p2 + exchanged p3 → NC"]
            FPN_Out -->|p2| NC_FPN_Fuse
            Exch -->|f_n| NC_FPN_Fuse
            NC_FPN_Fuse --> NC_Head["NC Head Conv 3×3 256→256 →<br/>BN+ReLU → Conv 1×1 256→10"] --> NCOut["NC Logits<br/>[B, 10, 128, 128]<br/>← Higher resolution (p2 level)"]
        end

        subgraph NP_HV["NP + HV Branches"]
            HighRes["cat[p1, p2↑, c1↑, c2↑]<br/>→ [B, 1024, 512, 512]<br/>← SKIPS: C1+C2 → np/hv decoder"]
            FPN_Out --> HighRes
            CNN_Feats -->|"C1 proj 96→256 + upsample"| HighRes
            CNN_Feats -->|"C2 proj 192→256 + upsample"| HighRes
            HighRes --> NP["HoVerNeXtNucleiHead<br/>Conv 1024→64→1"] --> NPOut["NP Logits<br/>[B, 1, 512, 512]"]
            HighRes --> HV["HoVerNeXtNucleiHead<br/>Conv 1024→64→2"] --> HVOut["HV Maps<br/>[B, 2, 512, 512]"]
        end
    end

    %% ─── UPSAMPLE ───
    TissueOut --> UpTissue["Upsample to input size<br/>→ [B, 6, H, W]"]
    NCOut --> UpNC["Upsample to input size<br/>→ [B, 10, H, W]"]
    NPOut --> UpNP["Upsample to input size<br/>→ [B, 1, H, W]"]
    HVOut --> UpHV["Upsample to input size<br/>→ [B, 2, H, W]"]

    SCDFA_W["SCDFA<br/>learned W [6×10]<br/>softmax(tissue) @ W"] -.-> NC_Residual["+ λ · SCDFA<br/>λ=0.3 from config (not 1.0)"]
    UpTissue --> SCDFA_W
    SCDFA_W -.-> NC_Residual

    %% ─── OUTPUTS ───
    UpTissue --> Final["Outputs"]
    NC_Residual --> Final
    UpNP --> Final
    UpHV --> Final

    Final --> Outputs["{
tissue:   [B, 6,  H, W]
nc:       [B, 10, H, W]
np:       [B, 1,  H, W]
hv:       [B, 2,  H, W]
}"]
```

## Class Hierarchy

```
UnifiedPanopticNet
├── encoder: UnifiedPanopticEncoder
│   ├── vit_model: Virchow2 ViT-H/14    # 32 blocks, dim=1280, patch_size from config
│   ├── cnn_model: ConvNeXt-Tiny        # 4 stages, features_only; auto-built if None
│   └── bridges: SpatialInjector ×4     # multi-head QKV cross-attn, target_grid=32
├── fpn: HierarchicalFPN
│   ├── latent_convs ×4                 # 1×1, CNN_dims→256
│   ├── vit_proj                        # 1×1, 1280→256; grid=H//patch_size
│   ├── smooth_convs ×4                 # 3×3, 256→256
│   └── low_level_fuse                  # 1×1, 256→96 + BN + ReLU
├── decoders: ParallelDecoders
│   ├── tissue_proj / nuclei_proj       # 1×1, 256→256
│   ├── exchange: MutualFeatureExchange # gated feature swap
│   ├── c2_proj_tissue                  # 1×1, 192→256 — SKIP C2→tissue
│   ├── vit_tissue_projs ×4             # 1×1, 1280→256 — SKIP vit→tissue
│   ├── tissue_fuse                     # 1×1, 7×256→256 + BN + ReLU
│   ├── c1_proj_np                      # 1×1, 96→256 — SKIP C1→np/hv
│   ├── c2_proj_np                      # 1×1, 192→256 — SKIP C2→np/hv
│   ├── tissue_decoder: DeepLabV3PlusTissueHead
│   │   └── aspp: ASPP                  # 4× atrous + pool
│   ├── nc_head: CellViTPlusPlusNucleiDecoder
│   │   ├── 4× ViT proj + fuse
│   │   └── fpn_fuse: cat[fused, p2, f_n]  # higher-res NC output at p2 level
│   ├── np_head: HoVerNeXtNucleiHead    # 1024→64→1
│   ├── hv_head: HoVerNeXtNucleiHead    # 1024→64→2
├── context_encoder: ContextEncoder     # optional, EfficientNet-B0
├── context_fusion: ContextFusionModule # optional, FiLM identity init
├── site_embed: nn.Embedding(9, 256)    # site conditioning
├── site_proj: nn.Linear or Identity    # projects embed to match FPN dim
└── sc_dfa: SCDFA                       # 6×10 weight matrix, λ=0.3 default
```

## Configuration (Stage1Config defaults)

| Parameter | Value | Description |
|-----------|-------|-------------|
| batch_size | auto-detected | 6 (H100), 4 (A100), 2 (V100), 1 (smaller) |
| grad_accum_steps | auto-adjusted | max(2, 12 // bs); effective batch ≈ 12 |
| epochs | 50 | Total training epochs |
| lr | 1e-4 | AdamW learning rate |
| weight_decay | 1e-4 | AdamW weight decay |
| warmup_epochs | 5 | Linear warmup epochs |
| virchow2_model_name | paige-ai/Virchow2 | HuggingFace ViT model |
| cnn_backbone | convnext_tiny | timm CNN backbone |
| fine_tune_last_n_blocks | 6 | ViT blocks to unfreeze |
| sc_dfa_lambda | 0.3 | SC-DFA strength (inference) |
| focal_start_epoch | 10 | FocalTversky ramp start |
| focal_full_epoch | 16 | FocalTversky ramp end |
| focal_max_weight | 0.5 | Max FocalTversky weight |
| sc_dfa_start_epoch | 15 | SC-DFA ramp start |
| sc_dfa_full_epoch | 22 | SC-DFA ramp end |
| sc_dfa_max_weight | 0.3 | Max SC-DFA λ during training |
| compile_model | True | torch.compile on GPU CC ≥ 7.0 |
| use_context_encoder | False | Enable context conditioning |
| use_stain_aug | False | Enable H&E stain augmentation |

## Key Design Decisions

1. **Background included as class 0** — Model predicts 6 tissue classes (0=background, 1=stroma, 2=blood_vessel, 3=tumor, 4=epidermis, 5=necrosis). Background is no longer `ignore_index=255`; it is learned like any other class. Inference output is already in PUMA format (no shift needed).

2. **No boundary branch** — The `BoundaryAttentionModule` was removed. `ParallelDecoders` returns 4 outputs (tissue, np, nc, hv). The boundary loss task has multiplier 0.0.

3. **Rare-class focused** — Weighted sampling (bonus ×8), class-weighted loss, 55% selection score weight on rare dice.

4. **Progressive training** — FocalTversky ramps epochs 10–16; SC-DFA ramps epochs 15–22.

5. **TTA ×8** — 8 geometric transforms (flip + rot) applied, inverse-transformed, averaged during inference.

6. **Encoder-decoder skip connections** — Four skip paths improve gradient flow and feature preservation: (1) FPN P1 ← s1 (C1 upsampled), (2) tissue decoder ← C2 additive, (3) tissue decoder ← 4× ViT intermediate features, (4) np/hv decoders ← C1 + C2 projections.

7. **NC high-resolution decoder** — `CellViTPlusPlusNucleiDecoder` fuses ViT features with FPN p2 and exchanged p3 to output nuclei classification logits at 2× higher resolution (p2 level instead of ViT grid).

8. **Multi-head SpatialInjector** — Cross-attention bridges use 8-head Q/K/V projection with `target_grid=32`, reducing CNN token count by 4× vs. `target_grid=64`.

9. **FiLM identity initialization** — Context conditioning starts as identity (γ=0, β=0 → `(1+0)·v+0 = v`), avoiding disruption of pretrained FPN features at warmup.

10. **Auto CNN backbone** — `cnn_model=None` triggers automatic `build_cnn_backbone()`, simplifying the constructor call in both training and inference.

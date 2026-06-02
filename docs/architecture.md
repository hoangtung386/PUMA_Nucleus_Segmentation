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
        CNN["ConvNeXt-Tiny<br/>timm features_only"] --> CNN_Feats["CNN Features<br/>C1: [B, 96, 256, 256]<br/>C2: [B, 192, 128, 128]<br/>C3: [B, 384, 64, 64]<br/>C4: [B, 768, 32, 32]"]

        ViT["Virchow2 ViT-H/14<br/>32 blocks, dim=1280<br/>fine-tune last 6 blocks"] --> ViT_Tokens["ViT Tokens<br/>[B, 5330, 1280]"]
        ViT --> Intermediate["Intermediate (blocks 8,16,24,31)<br/>reshape tokens → spatial<br/>[4, B, 1280, 73, 73]"]

        Bridges["SpatialInjector ×4<br/>cross-attention Q=ViT, K,V=CNN<br/>after blocks 7,15,23,31"]
        CNN_Feats --> Bridges
        ViT ---> Bridges
        Bridges --> ViT
    end

    %% ─── FPN ───
    Encoder --> FPN

    subgraph FPN["HierarchicalFPN"]
        direction TB
        Latent["Latent Conv2d 1×1<br/>CNN_dims → 256<br/>×4 levels → s1..s4"] --> P_CNN["[B, 256, H/4…H/32]"]
        ViT_Proj["ViT Proj Conv2d 1×1<br/>1280 → 256"] --> P_ViT["[B, 256, 73, 73]"]
        Fusion["Top-down fusion<br/>ViT injected at p4 level"]
        P_CNN --> Fusion
        P_ViT --> Fusion
        Fusion --> FPN_Out["{p1–p5}<br/>[B, 256, H/2…H/32]<br/>+ low_level_feat<br/>[B, 96, H/4, H/4]"]

        P1_Fix["P1 = smooth1(s1↑ + p2↑)<br/>s1 from C1 upsampled 2×<br/>SKIP — preserves original encoder features"]
        P_CNN --> P1_Fix
        FPN_Out --> P1_Fix
    end

    %% ─── CONDITIONING ───
    Context["ContextROI<br/>[B, 3, 320, 320]"] -.-> CtxEnc["ContextEncoder<br/>EfficientNet-B0<br/>→ [B, 256] (global)"]
    CtxEnc -.-> CtxFusion["ContextFusion<br/>FiLM: γ·feat + β"]
    SiteIDs["Site IDs<br/>[B]"] -.-> SiteEmb["nn.Embedding(9,256)<br/>→ bias [B,256,1,1]"]
    CtxFusion -.-> FPN_Out
    SiteEmb -.-> FPN_Out

    %% ─── DECODERS ───
    FPN --> Decoders

    subgraph Decoders["ParallelDecoders"]
        direction TB
        Exch["MutualFeatureExchange<br/>depthwise gate on p3<br/>f_tissue, f_nuclei<br/>[B, 256, 128, 128]"]
        FPN_Out --> Exch

        subgraph Tissue["Tissue Branch"]
            TFuse["tissue_fuse Conv1×1<br/>cat[f_t, p4↑, p5↑, vit(8)↑, vit(16)↑, vit(24)↑, vit(31)↑]<br/>7×256 → 256<br/>← SKIP: vit_intermediate → tissue"]
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
            NC_Fuse --> NC_Head["NC Head Conv 3×3 256→256 →<br/>BN+ReLU → Conv 1×1 256→10"] --> NCOut["NC Logits<br/>[B, 10, 73, 73]"]
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

    SCDFA_W["SCDFA<br/>learned W [6×10]<br/>softmax(tissue) @ W"] -.-> NC_Residual["+ λ · SCDFA"]
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
│   ├── vit_model: Virchow2 ViT-H/14    # 32 blocks, dim=1280
│   ├── cnn_model: ConvNeXt-Tiny        # 4 stages, features_only
│   └── bridges: SpatialInjector ×4     # cross-attention ViT←CNN
├── fpn: HierarchicalFPN
│   ├── latent_convs ×4                 # 1×1, CNN_dims→256
│   ├── vit_proj                        # 1×1, 1280→256
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
│   ├── nc_head: CellViTPlusPlusNucleiDecoder  # 4× ViT proj + fuse
│   ├── np_head: HoVerNeXtNucleiHead    # 1024→64→1
│   ├── hv_head: HoVerNeXtNucleiHead    # 1024→64→2
├── context_encoder: ContextEncoder     # optional, EfficientNet-B0
├── context_fusion: ContextFusionModule # optional, FiLM
├── site_embed: nn.Embedding(9, 256)    # site conditioning
└── sc_dfa: SCDFA                       # 6×10 weight matrix
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
| fine_tune_last_n_blocks | 6 | ViT blocks to unfreeze |
| focal_start_epoch | 10 | FocalTversky ramp start |
| focal_full_epoch | 16 | FocalTversky ramp end |
| focal_max_weight | 0.5 | Max FocalTversky weight |
| sc_dfa_start_epoch | 15 | SC-DFA ramp start |
| sc_dfa_full_epoch | 22 | SC-DFA ramp end |
| sc_dfa_max_weight | 0.3 | Max SC-DFA λ |
| compile_model | True | torch.compile on GPU CC ≥ 7.0 |
| use_context_encoder | False | Enable context conditioning |
| use_stain_aug | False | Enable H&E stain augmentation |

## Key Design Decisions

1. **Background included as class 0** — Model predicts 6 tissue classes (0=background, 1=stroma, 2=blood_vessel, 3=tumor, 4=epidermis, 5=necrosis). Background is no longer `ignore_index=255`; it is learned like any other class. Inference output is already in PUMA format (no shift needed).

2. **No boundary branch** — The `BoundaryAttentionModule` was removed. `ParallelDecoders` returns 4 outputs (tissue, np, nc, hv). The boundary loss task has multiplier 0.0.

3. **Rare-class focused** — Weighted sampling (bonus ×8), class-weighted loss, 55% selection score weight on rare dice.

4. **Progressive training** — FocalTversky ramps epochs 10–16; SC-DFA ramps epochs 15–22.

5. **TTA ×8** — 8 geometric transforms (flip + rot) applied, inverse-transformed, averaged during inference.

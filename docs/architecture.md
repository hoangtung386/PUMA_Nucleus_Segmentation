# SymbioPan v8 (CellPath) — Model Architecture

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
        CNN["ConvNeXt-Tiny<br/>timm features_only"] --> CNN_Feats["CNN Features<br/>[B, 96, 256, 256]<br/>[B, 192, 128, 128]<br/>[B, 384, 64, 64]<br/>[B, 768, 32, 32]"]

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
        Latent["Latent Conv2d 1×1<br/>CNN_dims → 256<br/>×4 levels"] --> P_CNN["[B, 256, H/4…H/32]"]
        ViT_Proj["ViT Proj Conv2d 1×1<br/>1280 → 256"] --> P_ViT["[B, 256, 73, 73]"]
        Fusion["Top-down fusion<br/>ViT injected at p4 level"]
        P_CNN --> Fusion
        P_ViT --> Fusion
        Fusion --> FPN_Out["{p1–p5}<br/>[B, 256, H/2…H/32]<br/>+ low_level_feat<br/>[B, 96, H/4, H/4]"]
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
            TFuse["tissue_fuse Conv1×1<br/>cat[f_t, p4↑, p5↑]<br/>768 → 256"] --> ASPP["ASPP<br/>(rates 1,3,6,9 + pool)<br/>→ [B, 256, 128, 128]"]
            ASPP --> LowConv["low_level_conv<br/>[B,96]→[B,48]"]
            FPN_Out --> LowConv
            LowConv --> TissueHead["DeepLabV3PlusTissueHead<br/>upsample + fuse conv<br/>→ classifier Conv1×1"] --> TissueOut["Tissue Logits<br/>[B, 5, 512, 512]"]
        end

        subgraph NucleiClass["Nuclei Classification Branch"]
            NC_Proj["ViT Projs ×4<br/>Conv2d 1×1, 1280→256"] --> NC_Fuse["Fuse Conv 3×3<br/>cat 4 levels → 1024→256"]
            Intermediate --> NC_Proj
            NC_Fuse --> NC_Head["NC Head Conv 3×3<br/>256→10"] --> NCOut["NC Logits<br/>[B, 10, 73, 73]"]
        end

        subgraph NP_HV["NP + HV Branches"]
            HighRes["cat[p1, p2↑]<br/>→ [B, 512, 512, 512]"]
            FPN_Out --> HighRes
            HighRes --> NP["HoVerNeXtNucleiHead<br/>Conv 512→64→1"] --> NPOut["NP Logits<br/>[B, 1, 512, 512]"]
            HighRes --> HV["HoVerNeXtNucleiHead<br/>Conv 512→64→2"] --> HVOut["HV Maps<br/>[B, 2, 512, 512]"]
        end

        subgraph Boundary["Boundary Branch"]
            FPN_Out --> P1["p1 [B,256,512,512]"]
            P1 --> BA["BoundaryAttentionModule<br/>Conv 256→64→1 + sigmoid"] --> BOut["Boundary Map<br/>[B, 1, 512, 512]"]
        end
    end

    %% ─── UPSAMPLE + SC-DFA ───
    TissueOut --> UpTissue["Upsample ×2<br/>→ [B, 5, 1024, 1024]"]
    NCOut --> UpNC["Upsample ×14<br/>→ [B, 10, 1024, 1024]"]
    NPOut --> UpNP["Upsample ×2<br/>→ [B, 1, 1024, 1024]"]
    HVOut --> UpHV["Upsample ×2<br/>→ [B, 2, 1024, 1024]"]
    BOut --> UpB["Upsample ×2<br/>→ [B, 1, 1024, 1024]"]

    SCDFA_W["SCDFA<br/>learned W [5×10]<br/>softmax(tissue) @ W"] -.-> NC_Residual["+ λ · SCDFA"]
    UpTissue --> SCDFA_W
    SCDFA_W -.-> NC_Residual

    %% ─── OUTPUTS ───
    UpTissue --> Final["Outputs"]
    NC_Residual --> Final
    UpNP --> Final
    UpHV --> Final
    UpB --> Final

    Final --> Outputs["{
tissue:   [B, 5,  H, W]
nc:       [B, 10, H, W]
np:       [B, 1,  H, W]
hv:       [B, 2,  H, W]
boundary: [B, 1,  H, W]
}"]

    %% ─── LOSS ───
    Outputs -.-> Loss

    subgraph Loss["MultiTaskUncertaintyLoss"]
        direction LR
        TCE["Tissue: CE + FocalTversky<br/>× 2.5"] --> LW1
        NCE["NC: CE + FocalTversky<br/>× 2.8"] --> LW2
        NP_L["NP: FocalBCE + SoftDice<br/>× 1.0"] --> LW3
        HV_L["HV: SmoothL1 (β=0.5)<br/>× 1.0"] --> LW4
        B_L["Boundary: not used<br/>× 0.0"] --> LW5
        LW1["∑ multiplier[i] · (exp(-log_var[i]) · loss[i] + log_var[i])"]
        LW2 --> LW1
        LW3 --> LW1
        LW4 --> LW1
        LW5 --> LW1
    end

    %% ─── INFERENCE ───
    subgraph Inference["Inference Pipeline"]
        direction TB
        WSI["Input WSI TIFF"] --> Tiler["Tiler<br/>1024×1024 tiles, 256 overlap<br/>768 stride"]
        Tiler --> TTA["TTA ×8<br/>(flip + rot) → average"]
        TTA --> Model_Fwd["Model forward<br/>+ site embedding"]
        Model_Fwd --> Tissue_Acc["Accumulate tissue logits<br/>weighted by overlap"]
        Model_Fwd --> HV_Inst["HV Instance Segmentation<br/>threshold → gradient → watershed"]
        HV_Inst --> Classify["Classify instances<br/>softmax majority vote"]
        Classify --> Poly["Instances → Polygons<br/>GeoJSON"]
        Tissue_Acc --> Tissue_Final["Tissue: argmax → PUMA<br/>TIFF mask"]
    end
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
│   ├── tissue_decoder: DeepLabV3PlusTissueHead
│   │   └── aspp: ASPP                  # 4× atrous + pool
│   ├── nc_head: CellViTPlusPlusNucleiDecoder  # 4× ViT proj + fuse
│   ├── np_head: HoVerNeXtNucleiHead    # 512→64→1
│   ├── hv_head: HoVerNeXtNucleiHead    # 512→64→2
│   ├── boundary_attn: BoundaryAttentionModule # 256→64→1 + sigmoid
│   └── tissue_fuse                     # 1×1, 768→256 + BN + ReLU
├── context_encoder: ContextEncoder     # optional, EfficientNet-B0
├── context_fusion: ContextFusionModule # optional, FiLM
├── site_embed: nn.Embedding(9, 256)    # site conditioning
└── sc_dfa: SCDFA                       # 5×10 weight matrix
```

## Configuration (Stage1Config defaults)

| Parameter | Value | Description |
|-----------|-------|-------------|
| batch_size | 8 | Per GPU |
| grad_accum_steps | 2 | Effective batch = 16 |
| epochs | 50 | Total training epochs |
| lr | 1e-4 | AdamW learning rate |
| weight_decay | 1e-4 | AdamW weight decay |
| warmup | 5 | Linear warmup epochs |
| fine_tune_last_n_blocks | 6 | ViT blocks to unfreeze |
| focal_ramp_start | 10 | FocalTversky ramp start |
| focal_ramp_end | 16 | FocalTversky ramp end |
| focal_max_weight | 0.5 | Max FocalTversky weight |
| scdfa_ramp_start | 15 | SC-DFA ramp start |
| scdfa_ramp_end | 22 | SC-DFA ramp end |
| scdfa_max_lambda | 0.3 | Max SC-DFA λ |
| context_encoder | False | Enable context conditioning |
| stain_aug | False | Enable H&E stain augmentation |

## Key Design Decisions

1. **No background tissue class** — Model predicts 5 foreground classes; background (PUMA ID 0) is `ignore_index=255` in loss. Inference shifts output +1 for PUMA compatibility.

2. **Rare-class focused** — Weighted sampling (bonus ×8), class-weighted loss, 55% selection score weight on rare dice.

3. **Progressive training** — FocalTversky ramps epochs 10–16; SC-DFA ramps epochs 15–22.

4. **TTA ×8** — 8 geometric transforms (flip + rot) applied, inverse-transformed, averaged during inference.

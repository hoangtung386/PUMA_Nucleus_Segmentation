# SymbioPan v8 — Code Review

## Điểm tốt

- ConvNeXt-Tiny backbone — đơn giản, đúng
- TTA logic trong `infer_wsi.py` — thiết kế tốt
- Warm-up cosine scheduler — chuẩn
- Gradient accumulation trong train loop — đúng hướng
- 9-class site type trong dataset — đúng hướng (dù model chưa dùng `site_ids`)
- DeepLabV3+ tissue head concept — tốt
- Context encoder/fusion module — cấu trúc ổn

---

## Bug nghiêm trọng (sẽ crash khi chạy thật)

### Bug 1 — `encoder.py`: Double forward pass

```python
# HIỆN TẠI — block(x) gọi 2 lần mỗi iteration
for i, block in enumerate(blocks):
    x = block(x)[0] if isinstance(block(x), tuple) else block(x)
```

`block(x)` được gọi một lần để check `isinstance`, một lần để gán kết quả. Mỗi vòng lặp chạy ViT block **2 lần** — sai kết quả, tốn gấp đôi VRAM.

```python
# SỬA
for i, block in enumerate(blocks):
    out = block(x)
    x = out[0] if isinstance(out, tuple) else out
```

### Bug 2 — `encoder.py`: Virchow2 API không chắc chắn

```python
x = self.vit_model.get_input_embeddings()(img)  # NLP-style API
```

Virchow2 dùng `AutoModel` với `trust_remote_code=True`. API thực tế phụ thuộc vào custom code của Paige AI.
Cách này dễ fail nếu model không có `get_input_embeddings()`. Các fallback `embeddings()`, `patch_embed()`, `_simple_patch_embed()` đều là phỏng đoán.

Cần kiểm tra Virchow2 thực tế trả về format gì khi gọi:
```python
outputs = model(pixel_values=img, output_hidden_states=True)
```

### Bug 3 — Channel mismatch: FPN `low_level_fuse` (48) vs `DeepLabV3PlusTissueHead` (96)

Trong `fpn_aggregator.py`:
```python
self.low_level_fuse = nn.Sequential(
    nn.Conv2d(fpn_dim, 48, ...)   # 256 → 48
)
low_level_feat = self.low_level_fuse(s1)  # output 48 channels
```

FPN trả về `low_level_feat` shape `(B, 48, H, W)`.

Trong `decoders.py`, `DeepLabV3PlusTissueHead` được init với `low_level_channels=96`:
```python
self.low_level_conv = nn.Conv2d(low_level_channels, 48, ...)  # expects 96 input channels
```

Khi `ParallelDecoders.forward()` gọi:
```python
tissue_logits = self.tissue_decoder(tissue_input, low_level_feat)
# low_level_feat = 48 channels, low_level_conv expects 96
```

→ **RuntimeError** ngay lần forward đầu tiên.

**Sửa**: Đồng bộ `low_level_channels` giữa FPN và tissue decoder, hoặc đổi FPN về 96 channels.

### Bug 4 — `vit_intermediate` shape mismatch: tokens vs spatial

Encoder trả về:
```python
vit_intermediate_tensor = torch.stack(intermediate_features, dim=0)
# shape: (4, B, seq_len, dim) = token sequences
```

FPN `forward()` NHẬN `vit_intermediate` nhưng KHÔNG dùng — parameter bị bỏ qua trong thân hàm.

Sau đó `ParallelDecoders.forward` truyền thẳng `vit_intermediate` vào `self.nc_head`:
```python
nc_logits = self.nc_head(vit_intermediate)
```

`CellViTPlusPlusNucleiDecoder` gọi `nn.Conv2d(1280, fpn_dim, 1)` trên tensor dạng `(B, seq_len, dim)`:
```python
for proj, feat in zip(self.vit_projs, vit_intermediate):
    x = proj(feat)  # Conv2d expects (B, C, H, W) but gets (B, seq_len, dim)
```

→ **RuntimeError**.

**Ghi chú**: Test `test_decoder_output_shapes` pass vì nó tự tạo `vit_intermediate` đúng spatial format `(4, 1, 1280, 64, 64)` — không đi qua encoder thật.

**Sửa**: Cần reshape token sequences `(B, seq_len, dim)` → spatial `(B, dim, H, W)` trước khi đưa vào decoder, hoặc tích hợp xử lý này trong FPN (hiện tại `vit_intermediate_projs` được định nghĩa nhưng không dùng).

---

## Bug vô hiệu — code chạy được nhưng không có tác dụng

### Bug 5 — `site_ids` nhận vào nhưng không dùng

```python
# panoptic_net.py
def forward(self, images, site_ids=None, context_roi=None):
    ...
    # site_ids không được dùng ở bất kỳ đâu
```

Dataset tính `site_id`, train loop truyền vào (`train_loop.py` line 28-30), model nhận nhưng bỏ qua. FiLM site conditioning (Phase 3) chưa được tích hợp.

### Bug 6 — `detect_gpu_setup()` không bao giờ được gọi

`gpu_setup.py` định nghĩa `detect_gpu_setup()` nhưng không module nào gọi nó:
- `scripts/run_stage1.py` chỉ gọi `training.stage1_trainer.main()`
- `stage1_trainer.py` không import hay gọi `detect_gpu_setup()`
- Notebook không gọi nó trong training flow

Ngoài ra, ngay cả khi gọi, `global STAGE1_DEFAULT_CONFIG = replace(...)` chỉ ảnh hưởng đến namespace của `gpu_setup`, không ảnh hưởng đến `stage1_trainer.cfg` (vì import từ `configs`). Config override hoàn toàn vô hiệu.

---

## Bug đã sửa

### ~Bug 5 (cũ)~ — Import sai trong `infer_wsi.py`

`infer_wsi.py` dùng `from inference.model_loader import load_stage1` — chính xác. ✅

### ~Bug 6 (cũ)~ — Import sai trong `tests/test_models.py`

`BoundaryAttentionModule` được re-export từ `models/decoders.py`:
```python
from models.components.boundary_attention import BoundaryAttentionModule
```
Nên import hoạt động bình thường. ✅

---

## Bảng tóm tắt

| Thành phần | Trạng thái | Mức độ |
|---|---|---|
| ConvNeXt-Tiny | ✅ Đúng | — |
| Virchow2 API | ⚠️ Fallback chain không chắc chắn | 🟠 Có thể crash |
| FPN channel mismatch | ❌ 48 vs 96 | 🔴 Crash |
| vit_intermediate shapes | ❌ Tokens vs spatial | 🔴 Crash |
| Double forward pass | ❌ 2x compute | 🔴 Sai kết quả |
| site_ids không dùng | ⚠️ Nhưng không dùng | 🟠 Vô hiệu |
| detect_gpu_setup không gọi | ❌ Không ai gọi | 🟠 Vô hiệu |
| Import infer_wsi | ✅ Đã sửa | — |
| Import BoundaryAttn test | ✅ Đã sửa | — |
| TTA logic | ✅ Đúng | — |
| Warm-up cosine LR | ✅ Đúng | — |
| Stain augmentation | ✅ Đúng | — |
| Grad accumulation | ✅ Đúng | — |

---

## Đánh giá tổng thể

**Kiến trúc thiết kế tốt**, nhưng có **2 bug crash** (channel mismatch + vit_intermediate shape) và **1 bug sai kết quả** (double forward pass) cần sửa trước khi chạy training thật. `site_ids` và `detect_gpu_setup` là các tính năng bị bỏ dở — cần tích hợp hoặc xoá.

# PUMA Version 13.2

V13.2 là bản pipeline PUMA hai giai đoạn (phát hiện + phân loại nhân tế bào) được tinh gọn
từ V13.1. Nó giữ nguyên thiết kế OOF năm fold an toàn rò rỉ dữ liệu ở Stage 1 và split
train/validation cố định đã tối ưu ở Stage 2, đồng thời cải thiện việc refit detector, mức
phơi nhiễm lớp hiếm, tối ưu hoá theo từng pha curriculum, tốc độ, và suy luận cho
Grand Challenge.

> Đây là bản tiếng Việt của [README_RUN_FIRST.md](README_RUN_FIRST.md). Hai file có cùng
> cấu trúc; khi sửa một file, hãy sửa cả file kia.

## Tình trạng hiện tại

| Stage | Notebook | Tình trạng |
|---|---|---|
| Tiền xử lý | `00_Preprocess.ipynb` | chạy được, đã chạy ở đây |
| Stage 1 | `01_Train_Stage1.ipynb` | code V13.2, chưa chạy trên máy trạm này |
| Stage 2 | `02_Train_Stage2.ipynb` | code V13.2, chưa chạy trên máy trạm này |
| Đánh giá / suy luận | `03_Evaluate_Infer.ipynb` | code V13.2, chưa chạy trên máy trạm này |

Lõi V13.2 đã thay thế toàn bộ code Stage-1/Stage-2 của V13.1 trong repo này. Chỉ
`puma/data/preprocess.py` là của riêng dự án này: V13.2 không sửa file đó, và nó đang giữ
phần chia fold theo hạn mức (capacity) của chúng ta. Xem mục *Những gì đã và chưa được
kiểm chứng*.

Phần 1 là cách chạy dự án trên máy trạm. Phần 2 là tài liệu tham chiếu pipeline V13.2.

---

# Phần 1 — Chạy trên máy trạm

Mục tiêu: một RTX 3090 Ti 24 GB, Linux, driver NVIDIA 525 hoặc mới hơn.

## 1. Gửi cả folder sang máy trạm

Gửi thẳng toàn bộ folder này là cách làm được, và là luồng mặc định của tài liệu này.
Không có file nào cần sửa sau khi chuyển: `setup_local.sh` dựng lại môi trường, còn
notebook tự tìm thư mục gốc bằng `Path.cwd()`.

**Dùng `rsync -a` hoặc `tar`, đừng dùng `scp -r`/`cp -r`/zip.** Lý do duy nhất nhưng quan
trọng: trong repo có symlink `Dataset -> dataset`. Các công cụ đi theo symlink sẽ nhân đôi
toàn bộ dataset (32 GB thành 64 GB), hoặc tạo ra một `Dataset` là thư mục thật rồi
`setup_local.sh` không sửa được nữa vì nó chỉ tạo symlink khi `Dataset` chưa tồn tại.

```bash
rsync -a --info=progress2 ./ user@workstation:/path/to/SymbioPan/
```

Hai thư mục trong đó là công chuyển vô ích, nhưng **không gây hỏng gì**:

| Thư mục | Dung lượng | Chuyện gì xảy ra trên máy trạm |
|---|---:|---|
| `.venv/` | 7.2 GB | Virtualenv nhúng cứng đường dẫn tuyệt đối nên sang máy khác là hỏng. `setup_local.sh` **xoá và dựng lại** nó ở bước 1, nên bản copy chỉ tốn băng thông. |
| `PUMA_outputs/` | 1.9 GB | Có thể được dùng lại, nhưng khoá cache gồm mtime của file nguồn mà phần lớn công cụ copy không giữ đúng, nên rất có thể `00_Preprocess.ipynb` vẫn rebuild — mất 25 giây. |

Muốn nhẹ đường truyền thì loại chúng ra cộng thêm hai thư mục dataset không được đọc ở bất
kỳ đâu trong `puma/` (`tif_context_ROIs` 21 GB — view V256 của Stage 2 được crop từ chính
ROI 1024×1024, và `geojson_tissue` 74 MB — V13.2 không train model tissue). Còn khoảng
**1.1 GB** thay vì 32 GB:

```bash
rsync -a --info=progress2 \
  --exclude '.venv' \
  --exclude 'PUMA_outputs' \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '.ipynb_checkpoints' \
  --exclude 'dataset/01_training_dataset_tif_context_ROIs' \
  --exclude 'dataset/01_training_dataset_geojson_tissue' \
  ./ user@workstation:/path/to/SymbioPan/
```

Sau khi chuyển, kiểm tra ba thứ trên máy trạm trước khi làm gì tiếp:

```bash
cd /path/to/SymbioPan
ls -l Dataset                                          # phải là: Dataset -> dataset
ls dataset/01_training_dataset_tif_ROIs/*.tif | wc -l   # phải là 205
ls dataset/01_training_dataset_geojson_nuclei/*.geojson | wc -l   # phải là 205
```

Nếu `Dataset` là thư mục thật chứ không phải symlink, xoá nó rồi tạo lại:
`rm -rf Dataset && ln -s dataset Dataset`.

Stage 2 cần thêm checkpoint UNI2-h trong `PUMA_pretrained_checkpoints/UNI2-h/`. Nếu bản
copy đã có sẵn thì Stage 2 chạy được offline; nếu không, `02_Train_Stage2.ipynb` tải về
một lần (cần `HF_TOKEN`, xem mục 3).

## 2. Dựng môi trường

Dependency được quản lý bằng [uv](https://docs.astral.sh/uv/), không phải bằng `pip`
trong notebook. Cài uv một lần nếu máy trạm chưa có:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Sau đó, từ thư mục gốc của dự án:

```bash
cd /path/to/SymbioPan
bash setup_local.sh
```

Script này idempotent (chạy lại bao nhiêu lần cũng được) và làm toàn bộ các việc sau:

1. xoá `.venv` nào mang từ máy khác sang;
2. tạo `.venv` mới trên CPython 3.11 do uv tự tải về (không cần Python hệ thống, không
   cần `apt`) — chọn 3.11 thay vì 3.13 vì `rasterio`/`timm` có wheel đầy đủ hơn;
3. cài PyTorch kèm CUDA (mặc định `cu128`; xem phần driver cũ bên dưới);
4. cài `requirements_colab.txt`, JupyterLab, ipykernel, ipywidgets;
5. gỡ kernel Jupyter `symbiopan` cũ nếu có và đăng ký lại trỏ vào `.venv` của *máy này*;
6. tạo symlink `Dataset -> dataset` nếu còn thiếu;
7. in ra danh sách đầy đủ GPU và GPU nào notebook sẽ chọn, tiếp đó là cách torch nhìn
   thấy các thiết bị đó, hỗ trợ bf16, toàn bộ dependency import được, và số file dataset.

`requirements_colab.txt` cố ý không có torch, vì Colab đã cài sẵn. Trên máy trạm torch
phải đến từ bước 3.

**Driver cũ.** Wheel `cu128` cần driver 525 hoặc mới hơn. Kiểm tra bằng `nvidia-smi`, nếu
driver cũ hơn thì build theo CUDA 12.6:

```bash
CUDA_BACKEND=cu126 bash setup_local.sh
```

Phần cuối output khi chạy thành công trên máy trạm hai GPU:

```
python           3.11.15  (/path/to/SymbioPan/.venv/bin/python)
gpus visible     2
  GPU 0          NVIDIA GeForce RTX 3090 Ti  24 GB
  GPU 1          NVIDIA GeForce RTX 3090 Ti  24 GB
notebooks will use GPU 1 (NVIDIA GeForce RTX 3090 Ti)
                 2 GPUs detected; using preferred GPU 1
torch            2.11.0+cu128
cuda available   True
  torch cuda:0    NVIDIA GeForce RTX 3090 Ti  23.6 GB  sm_86  bf16=True
  torch cuda:1    NVIDIA GeForce RTX 3090 Ti  23.6 GB  sm_86  bf16=True
deps             all 12 imports OK
tif ROIs         205 files  [OK]  .../Dataset/01_training_dataset_tif_ROIs
geojson nuclei   205 files  [OK]  .../Dataset/01_training_dataset_geojson_nuclei
```

Bản thân script không mask GPU nào — nó liệt kê tất cả để bạn kiểm tra được cách đánh
số. Việc mask diễn ra trong notebook.

## 3. Khởi động JupyterLab

```bash
cd /path/to/SymbioPan
./.venv/bin/jupyter lab
```

Phải khởi động **từ thư mục gốc của dự án**. Khi không chạy trên Colab, cả bốn notebook
lấy `PROJECT_DIR = Path.cwd()` và từ chối chạy tiếp nếu không thấy `puma/`, nên chạy sai
thư mục sẽ báo lỗi rõ ràng ngay lập tức thay vì import sai package.

Cả bốn notebook đều chọn sẵn kernel `SymbioPan (uv .venv)`. Notebook `00`, `01` và `02`
assert ở cell thứ hai rằng `sys.executable` đúng là `.venv/bin/python`, nên chạy sai kernel
sẽ lỗi ngay thay vì lỗi muộn ở một import thiếu.

Trong uv venv không có `pip`, nên `%pip install` **sẽ không chạy** trong các notebook này.
Đây là chủ ý. Muốn thêm package:

```bash
VIRTUAL_ENV=.venv uv pip install <package>
```

Tiền xử lý và Stage 1 không cần `HF_TOKEN`. Nó chỉ cần từ Stage 2 trở đi, nơi checkpoint
`MahmoodLab/UNI2-h` (repo gated) được tải về: chấp nhận điều khoản của repo trên Hugging
Face, rồi `export HF_TOKEN=hf_...` trước khi khởi động JupyterLab.

## 4. Notebook dùng GPU nào

**Trên máy có hai GPU hoặc nhiều hơn, mặc định huấn luyện chạy trên GPU 1**, để lại GPU 0
cho màn hình và các job khác. Máy một GPU thì tự động về GPU 0. Cell bootstrap in ra chính
xác lựa chọn đó:

```
GPUs detected        : 2
  0: NVIDIA GeForce RTX 3090 Ti (24 GB)
  1: NVIDIA GeForce RTX 3090 Ti (24 GB)
CUDA_VISIBLE_DEVICES : 1
CUDA_DEVICE_ORDER    : PCI_BUS_ID
training device      : physical GPU 1 (NVIDIA GeForce RTX 3090 Ti)  -> cuda:0 inside torch
reason               : 2 GPUs detected; using preferred GPU 1
```

Muốn đổi mặc định, sửa một dòng trong cell bootstrap của notebook:

```python
PREFERRED_GPU_INDEX = 1     # 0 cho GPU đầu tiên, 1 cho GPU thứ hai, v.v.
```

Muốn override cho cả session mà không sửa gì, đặt biến môi trường trước khi khởi động
JupyterLab — một `CUDA_VISIBLE_DEVICES` có sẵn luôn được tôn trọng:

```bash
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/jupyter lab
```

Ba chi tiết nên biết, vì đây là những chỗ hay gây nhầm lẫn:

- **`CUDA_DEVICE_ORDER=PCI_BUS_ID` được ghim cứng.** `nvidia-smi` đánh số GPU theo PCI bus
  id, còn mặc định của CUDA là `FAST_FIRST`, nên nếu không ghim thì "GPU 1" có thể là hai
  card khác nhau ở hai công cụ. Khi đã ghim, chỉ số trong notebook, trong
  `CUDA_VISIBLE_DEVICES` và trong `nvidia-smi` khớp nhau.
- **GPU được chọn trở thành `cuda:0`.** Việc mask ẩn hoàn toàn các GPU còn lại, nên
  `resolve_device()` và mọi `torch.device('cuda')` trong `puma/` đều lấy đúng card mà không
  cần sửa gì thêm. Thấy `cuda:0` trong log khi đang train trên GPU vật lý 1 là bình thường.
- **Việc chọn thiết bị phải xảy ra trước khi `torch` được import**, vì
  `CUDA_VISIBLE_DEVICES` chỉ được đọc lúc CUDA driver khởi tạo. Chính vì vậy `puma/gpu.py`
  không import torch, và cell bootstrap chạy trước mọi import torch. Nếu cell bootstrap
  được chạy lại trong một kernel đã train, việc đổi GPU sẽ âm thầm không có tác dụng — nên
  nó in cảnh báo trong trường hợp đó, và cell kiểm tra môi trường sẽ raise nếu torch thấy
  nhiều hơn một thiết bị. Tóm lại: **Restart Kernel rồi Run All** khi đổi GPU.

`setup_local.sh` in ra danh sách GPU đầy đủ và nói rõ notebook sẽ chọn GPU nào, nên có thể
xác nhận trước khi chạy bất cứ thứ gì.

Cell bootstrap của `01` và `02` đã được chạy thật với inventory GPU được inject, kết quả:

| Số GPU máy có | `CUDA_VISIBLE_DEVICES` | GPU dùng để train |
|---:|---|---|
| 2 | `1` | GPU vật lý 1 |
| 4 | `1` | GPU vật lý 1 |
| 1 | `0` | GPU vật lý 0 (fallback) |
| 0 | không đặt | CPU |

Cả hai notebook cho cùng kết quả vì dùng chung một cell bootstrap. Trong `puma/` không có
chỗ nào hard-code `cuda:1`, không có `set_device()`, không có `DataParallel` — mọi nơi đều
đi qua `resolve_device()` trả về `cuda` trần, nên GPU được mask chính là GPU được dùng.

## 5. Thứ tự chạy

1. `00_Preprocess.ipynb`
2. `01_Train_Stage1.ipynb`
3. `02_Train_Stage2.ipynb`
4. `03_Evaluate_Infer.ipynb`

Toàn bộ luồng từ lúc folder vừa sang tới lúc có model:

```bash
cd /path/to/SymbioPan
bash setup_local.sh          # dựng .venv + kernel, khoảng vài phút
export HF_TOKEN=hf_...       # chỉ cần cho Stage 2
./.venv/bin/jupyter lab
# Trong JupyterLab, kernel "SymbioPan (uv .venv)", Run All lần lượt:
#   00_Preprocess.ipynb    ~25 giây   -> PUMA_outputs/
#   01_Train_Stage1.ipynb             -> 5 checkpoint detector + OOF candidates
#   02_Train_Stage2.ipynb             -> model phân loại (50 epoch sàng lọc, 100 epoch model thắng)
#   03_Evaluate_Infer.ipynb           -> model final để submit (sau khi bật switch)
```

`00_Preprocess.ipynb` là **bắt buộc trước** `01_Train_Stage1.ipynb`, kể cả khi bản copy đã
mang theo `PUMA_outputs/`: khoá cache gồm mtime file nguồn nên rất có thể nó rebuild, và 25
giây rẻ hơn nhiều so với việc phát hiện thiếu artifact ở giữa lượt train. Stage 1 mở các
artifact `.npy` qua `PumaNpyStore.open()`, hàm này raise
`Missing preprocessed artifacts: [...]. Run 00_Preprocess.ipynb first.` khi chúng chưa có.
Riêng `puma_fold_assignments.npy` là file định nghĩa năm fold.

Mọi notebook mở đầu bằng cùng một cell bootstrap, chạy được cả ở đây lẫn trên Colab:

```python
try:
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT_DIR = Path('/content/drive/MyDrive/Research/PUMA')
except ImportError:
    PROJECT_DIR = Path.cwd().resolve()
```

### `00_Preprocess.ipynb` — Run All

Mất khoảng 25 giây trên 12 core (`preprocessing_workers=0` dùng toàn bộ logical core).
Ghi 1.9 GB vào `PUMA_outputs/`. Kết quả mong đợi:

```
205 GeoJSON / 205 TIFF / 205 matched pairs, 97193 annotated features
all ROIs 1024×1024
fold sizes [41, 41, 41, 41, 41]      size imbalance ratio 1.0
folds missing a class entirely: none
```

`FORCE_PREPROCESS = False` sẽ dùng lại cache còn hợp lệ. Chỉ đặt `True` khi muốn rebuild
có chủ đích — khoá cache đã bao gồm cấu hình dữ liệu, phiên bản schema artifact, và danh
mục file TIFF/GeoJSON nguồn, nên input thay đổi thật thì cache tự rebuild.

### `01_Train_Stage1.ipynb` — Run All, không cần sửa gì

Chọn kernel `SymbioPan (uv .venv)` → Run All. Bảy cell chạy theo thứ tự: bootstrap (chọn
GPU) → kiểm tra môi trường → runtime + preflight → **kiểm tra fold** → train → OOF + lock
triển khai Stage 1.

Cell kiểm tra fold in số ROI mỗi fold, loại melanoma, và cả mười class count, và raise nếu
split bị lệch. `run_stage1_a1()` chạy lại đúng kiểm tra đó bên trong, nên không thể bỏ qua
bằng cách chạy cell lộn thứ tự.

Train tuần tự năm outer fold `A1_IFCRN_PP`. **Mỗi outer fold train hai lượt**: một lượt
inner-validation để chọn epoch và post-processing, rồi reset và refit trên cả bốn fold
không phải outer. Tức là mười lượt train, không phải năm — cần tính đến khi ước lượng thời
gian.

Chạy xong, `PUMA_stage1_training_outputs/` có:

```
stage1_best_A1_IFCRN_PP_fold{0..4}_seed0.pt    5 checkpoint detector
stage1_results.csv                             metric từng fold
stage1_lock.json                               lock triển khai Stage 1
stage1_oof_candidates.npy                      input BẮT BUỘC của Stage 2
```

`stage1_oof_candidates.npy` là thứ nối Stage 1 sang Stage 2. Cell cuối gọi
`validate_full_oof(runtime)`, hàm này raise nếu còn ROI nào chưa có dự đoán out-of-fold —
nên **cứ để đủ `run_folds=(0, 1, 2, 3, 4)`**; thiếu một fold là Stage 2 không chạy được.

Muốn thử độ nặng trước khi cam kết cả năm fold, đặt `run_folds=(0,)` trong cell runtime,
chạy, xem VRAM và thời gian, rồi khôi phục đủ năm fold và chạy lại.

Nếu session bị ngắt: mở lại notebook và Run All. Mỗi epoch đều lưu resume checkpoint, các
fold đã xong bị bỏ qua khi hash cấu hình khớp, nên nó tiếp tục chứ không train lại từ đầu.

### `02_Train_Stage2.ipynb` — sàng lọc 50 epoch, rồi train model thắng 100 epoch

Cần Stage 1 xong trước, vì Stage 2 đọc `stage1_oof_candidates.npy`.

**Lượt 1 — sàng lọc.** Giữ nguyên mặc định `STAGE2_EPOCHS = 50` và Run All. Nó tạo split
80/20 cố định, tải checkpoint UNI2-h, rồi train tuần tự cả bốn thí nghiệm và in bảng xếp
hạng theo `macro_f1`.

**Lượt 2 — model thắng.** Nhìn bảng xếp hạng, rồi sửa đúng hai dòng trong cell runtime:

```python
STAGE2_EPOCHS = 100
WINNER_EXPERIMENT = 'V13_2_02_META_RARE_BS'   # đổi thành tên đứng đầu bảng
```

Restart Kernel rồi Run All. Ở `STAGE2_EPOCHS = 100`, cell chọn thí nghiệm tự chuyển từ "cả
bốn" sang "chỉ model thắng", nên không cần sửa gì thêm. `create_runtime()` chỉ nhận đúng 50
hoặc 100 — số khác bị từ chối ngay, kèm thông báo rõ ràng.

Model 100 epoch phải train lại **từ đầu**, không resume từ lượt 50 epoch: schedule
curriculum khác nhau (15/15/20 so với 30/30/40).

Chạy xong, `PUMA_stage2_training_outputs/` có checkpoint và metric của từng thí nghiệm. Đó
là **model phát triển**. Muốn ra model cuối để submit thì sang `03_Evaluate_Infer.ipynb`:
đặt `CREATE_DEVELOPMENT_LOCK = True` để lock model thắng, rồi `TRAIN_FINAL_MODEL = True`
với `FINAL_EPOCHS = 100` để train lại cấu hình đó trên **toàn bộ** ROI có nhãn. Kết quả:

```
stage2_v132_final_<experiment>_100ep_seed0_<hash>.pt    model phân loại cuối
stage2_v132_final_lock.json                            cấu hình + validity threshold
stage1_lock.json + 5 checkpoint A1                     ensemble detector khi triển khai
```

Bộ suy luận cần **cả hai**: năm checkpoint Stage 1 làm detector, và một model Stage 2 làm
bộ phân loại.

### `03_Evaluate_Infer.ipynb` — xem kết quả, train final, suy luận

Mọi bước sau bảng xếp hạng đều nằm sau một switch tường minh
(`CREATE_DEVELOPMENT_LOCK`, `TRAIN_FINAL_MODEL`, `RUN_LOCAL_INFERENCE`,
`RUN_GRAND_CHALLENGE_INFERENCE`), mặc định đều `False`, nên Run All chỉ in bảng xếp hạng.

## 6. Cấu hình Stage 1 và Stage 2 cho một RTX 3090 Ti

V13.2 cho mỗi stage một ngân sách batch và epoch riêng. Mặc định trong notebook:

```python
stage1_epochs=40,
stage2_epochs=50,                    # 50 sàng lọc, 100 cho model thắng; không nhận giá trị khác
stage1_effective_batch_size=16,
stage2_effective_batch_size=256,
stage1_micro_batch_size=16,          # không cần accumulation
stage2_micro_batch_size=256,         # không cần accumulation
stage1_early_stopping_enabled=True,
stage1_early_stopping_patience=10,
early_stopping_enabled=False,        # Stage 2: chạy đủ epoch để so sánh công bằng
```

Stage 1 train trên tile 512×512 (`tile_size=512`, `tile_overlap=128`), nhẹ hơn Stage 2 rất
nhiều.

Effective batch của mỗi stage phải chia hết cho micro-batch của stage đó, nếu không
`create_runtime()` sẽ từ chối cấu hình. Khi CUDA OOM, chỉ giảm micro-batch: Stage 1 tự
lùi `8 → 4 → 2 → 1` và Stage 2 `128 → 64 → 32 → 16`, cả hai đều giữ nguyên effective batch
bằng gradient accumulation, nên kết quả không đổi. Fallback này tự động khi
`PUMA_V132_AUTO_OOM_FALLBACK=1`, biến mà cell bootstrap đã đặt.

bf16 AMP được chọn tự động trên Ampere. Để load dữ liệu nhanh hơn trên máy nhiều core,
thêm vào cell runtime:

```python
runtime.training.number_of_workers = 8
```

## 7. Những gì đã và chưa được kiểm chứng

Đã kiểm chứng trên máy chuẩn bị bản này (RTX 3080 10 GB, driver 595, 12 core), với phần
code mà V13.2 **không** thay đổi:

- `setup_local.sh` từ đầu đến cuối: uv venv, torch 2.11.0+cu128, `cuda available: True`,
  cả 12 dependency import được;
- `00_Preprocess.ipynb` chạy hết, exit 0, 23 giây, 205/205 cặp file, 97193 nhân;
- cell kiểm tra fold, chạy trên artifact thật;
- `tests/test_fold_assignment.py`, 9/9 pass;
- `tests/test_gpu_selection.py`, 14/14 pass — các trường hợp hai, bốn, một và không GPU
  chạy trên inventory được inject, nên logic chọn multi-GPU vẫn được phủ trên máy một GPU;
- `import puma.gpu` không kéo theo `torch` lẫn `numpy`, và việc set
  `CUDA_VISIBLE_DEVICES` ở thời điểm đó thực sự quyết định những gì torch thấy sau này.

`puma/data/preprocess.py`, `puma/gpu.py` và cả hai file test đều không bị thay đổi khi merge
V13.2, nên các kết quả trên vẫn còn giá trị cho chúng.

Chưa kiểm chứng:

- **Chưa chạy bất cứ thứ gì thuộc Stage 1 hoặc Stage 2 của V13.2 ở đây.** Lõi sau khi
  merge compile được và mọi import đều giải quyết được, nhưng chưa có lượt train, sinh OOF,
  train final hay suy luận nào được thực thi trên máy trạm này.
- Chưa chạy gì trên RTX 3090 Ti. Các hướng dẫn 24 GB ở trên là suy luận từ kích thước tile
  và phép tính batch, không phải đo thực tế.
- Không có máy nào từ hai GPU vật lý trở lên. Việc chọn GPU 1 được kiểm chứng bằng unit
  test, bằng cơ chế mask, và bằng cách chạy thật cell bootstrap của `01`/`02` với inventory
  GPU được inject (bảng ở mục 4) — nhưng chưa phải trên phần cứng multi-GPU thật. Hãy đọc
  output của cell bootstrap ở lần chạy đầu tiên để xác nhận.

Hãy chạy thử một fold trước, bằng cách đặt `run_folds=(0,)` trong cell runtime, trước khi
cam kết chạy cả năm. Sau đó khôi phục `run_folds=(0, 1, 2, 3, 4)` — muốn OOF phủ hết dữ
liệu thì phải đủ năm fold.

## 8. Xử lý sự cố

| Hiện tượng | Nguyên nhân và cách sửa |
|---|---|
| `No module named pip` khi `%pip install` | Đúng như thiết kế: uv venv không có pip. Dùng `VIRTUAL_ENV=.venv uv pip install <package>`. |
| `Wrong kernel: /usr/bin/python3` | Kernel → Change Kernel → `SymbioPan (uv .venv)`. Nếu không có kernel này, chạy lại `bash setup_local.sh`. |
| `... is not the project root (no puma/ package here)` | JupyterLab được khởi động ở thư mục khác. Khởi động từ thư mục gốc dự án. |
| `GeoJSON directory does not exist: .../Dataset/...` | Thiếu symlink `Dataset -> dataset` (đường dẫn Linux phân biệt chữ hoa/thường). Chạy `ln -s dataset Dataset`. |
| `Missing preprocessed artifacts: [...]` | Chạy `00_Preprocess.ipynb` trước. |
| `Degenerate fold assignment: sizes [...]` | Split không đủ để làm nested validation. Rebuild với `FORCE_PREPROCESS = True`. |
| `V13.2 Stage-2 epochs must be exactly 50 or 100` | `stage2_epochs` chỉ nhận hai profile đó. Sàng lọc ở 50, train lại model thắng ở 100. |
| `Stage-1 sampling fractions must sum to 1.0` | Một `runtime.data.*_fraction` bị sửa mà không cân lại các giá trị còn lại. |
| `The wrong PUMA package is loaded` | Còn `puma` cũ trong `sys.modules` hoặc trên `sys.path`. Restart kernel rồi Run All; cell bootstrap dọn cả hai. |
| `torch.cuda.is_available()` là `False` | Driver quá cũ so với wheel. Kiểm tra `nvidia-smi` rồi build lại bằng `CUDA_BACKEND=cu126 bash setup_local.sh`. |
| CUDA OOM | Fallback tự động sẽ giảm micro-batch và giữ nguyên effective batch. Muốn ép thì đặt `stage1_micro_batch_size=8` hoặc `stage2_micro_batch_size=128`. |
| Thiếu `PUMA_pretrained_checkpoints/UNI2-h/uni2_h_model.bin` khi suy luận | Suy luận offline theo thiết kế. Chạy cell checkpoint của `02_Train_Stage2.ipynb` một lần, và đóng file đó vào container. |
| Train chạy sai GPU | Đọc output cell bootstrap. Nếu nó nói `respected existing`, tức là `CUDA_VISIBLE_DEVICES` từ shell đang thắng — unset nó rồi restart kernel. |
| `Expected exactly one visible GPU after selection` | `CUDA_VISIBLE_DEVICES` liệt kê nhiều thiết bị, hoặc torch đã khởi tạo CUDA trước khi cell bootstrap chạy. Restart Kernel rồi Run All. |
| Bootstrap cảnh báo CUDA đã được khởi tạo | Việc đổi GPU không có tác dụng trong kernel này. Restart Kernel rồi Run All. |

---

# Phần 2 — Tài liệu tham chiếu pipeline V13.2

## Stage 1: chỉ A1_IFCRN_PP

V13.2 chỉ giữ lại `A1_IFCRN_PP`.

Với mỗi fold trong năm outer fold:

1. giữ nguyên outer fold, không dùng để train;
2. train trên ba fold và dùng một fold không phải outer làm inner validation;
3. chọn epoch tốt nhất và post-processing tốt nhất trên inner validation;
4. reset A1;
5. refit A1 trên **cả bốn fold không phải outer** đúng bằng số epoch đã chọn;
6. dự đoán outer fold chưa từng được dùng bằng model đã refit.

Cách này vẫn giữ OOF an toàn rò rỉ dữ liệu, đồng thời tăng lượng dữ liệu mỗi detector OOF
được học từ 3/5 lên 4/5.

### Chọn post-processing cho Stage 1

V13.2 đánh giá đồng thời:

- heatmap threshold;
- local-max radius;
- suppression radius.

Trước tiên nó tìm vùng có oracle macro-F1 tốt nhất, sau đó trong các cấu hình nằm trong
khoảng `0.005` so với ngưỡng trần tốt nhất thì ưu tiên recall cao hơn ở lớp đuôi, rồi đến
recall tổng thể cao hơn. Nhờ vậy có thiên hướng recall nhẹ mà không phải chấp nhận một
detector có trần kém đi rõ rệt.

### Sampling ở Stage 1

Xác suất chọn gốc tile:

- tâm theo mật độ: 30%
- tâm theo nhân nhỏ: 20%
- tâm theo nhân nói chung: 30%
- tâm theo nhân hiếm: 15%
- background/ngẫu nhiên: 5%

Sampling theo nhân hiếm chỉ để detector (vốn không phân biệt lớp) nhìn thấy được nhân
hiếm; Stage 1 vẫn không được train như một bộ phân loại lớp. `create_runtime()` từ chối
các tỉ lệ không tổng bằng 1.0.

### Batch ở Stage 1

Mặc định:

```text
physical batch = 16
effective batch = 16
```

Lượt train inner-selection bật early stopping với patience 10; epoch được chọn sau đó được
dùng đúng như vậy cho lượt refit 4/5.

Fallback CUDA-OOM tự động thử `8 -> 4 -> 2 -> 1` trong khi vẫn giữ effective batch 16 bằng
gradient accumulation.

### Chia fold

Fold được tạo bởi `multilabel_greedy_folds()` trong `puma/data/preprocess.py`, nhóm theo
`case_id` để không bệnh nhân nào nằm ở hai fold, và phân tầng theo loại melanoma cùng cả
mười class. Kích thước fold là một hạn mức cứng bằng `total_rois / number_of_folds` (lệch
nhau nhiều nhất một ROI), tính theo ROI chứ không theo nhóm case; cân bằng lớp được tối ưu
*trong* hạn mức đó, rồi tinh chỉnh thêm bằng cách hoán đổi các nhóm có cùng số ROI giữa
các fold.

Trên PUMA kết quả là `[41, 41, 41, 41, 41]`.

Vì mỗi fold vừa là outer fold cho OOF vừa là inner validation của một fold khác, một split
lệch sẽ làm hỏng cả việc chọn threshold và độ phủ OOF mà vẫn train và báo cáo bình thường,
không lỗi. Do đó `validate_fold_assignments()` raise khi có fold nhỏ hơn một nửa kích thước
kỳ vọng, và nó được gọi từ `multilabel_greedy_folds()`, từ `run_stage1_a1()`, và từ cell
kiểm tra fold trong notebook `00` và `01`. `tests/test_fold_assignment.py` phủ phần cân
bằng, nhóm case, phân tầng, tính tiền định, và hành vi của validator:

```bash
./.venv/bin/python tests/test_fold_assignment.py
./.venv/bin/python tests/test_gpu_selection.py
```

## Chọn GPU

`puma/gpu.py` chứa phần chọn thiết bị mà cell bootstrap của notebook sử dụng:
`query_gpu_inventory()` đọc `nvidia-smi`, còn `select_cuda_device(preferred_index)` ghim
`CUDA_VISIBLE_DEVICES` và `CUDA_DEVICE_ORDER`. Module này cố ý không import `torch` — xem
mục *Notebook dùng GPU nào* ở Phần 1 để hiểu vì sao thời điểm lại quan trọng. Cả hai hàm
nhận inventory và environment được inject, nhờ đó các nhánh multi-GPU vẫn test được trên
máy một GPU.

## Stage 2: split cố định đã tối ưu

V13.2 tạo (hoặc dùng lại) một split phát triển 80/20 đã tối ưu, nhóm theo case, dùng chung
cho mọi thí nghiệm Stage 2. Không tạo lại split giữa các thí nghiệm.

Split cân bằng số ROI, số nhân, phân bố/sự hiện diện của lớp, lớp hiếm, nhóm case, và tỉ
lệ primary/metastatic.

```python
ensure_v132_split(runtime, force=False, val_fraction=0.20, seed=2026, check_sources=True)
```

## Hai profile epoch chính xác

Chỉ hai profile epoch cho Stage 2 được chấp nhận.

### Sàng lọc: 50 epoch

```text
Epoch  1-15: GT_POS
Epoch 16-30: OOF_POS
Epoch 31-50: OOF_ALL
```

### Final/model thắng: 100 epoch

```text
Epoch   1-30: GT_POS
Epoch  31-60: OOF_POS
Epoch  61-100: OOF_ALL
```

Model thắng ở 100 epoch phải được train lại từ đầu theo profile 100 epoch; không được
resume một lượt sàng lọc 50 epoch đã xong như thể nó cùng một schedule.

## Curriculum của Stage 2

Logic học của V13.1 được giữ lại có chủ đích:

1. **GT_POS (30%)** — centroid GT hoàn hảo; học phenotype nhân "sạch".
2. **OOF_POS (30%)** — các positive OOF của Stage 1 đã được match; học phenotype dưới sai
   số định vị thật của detector.
3. **OOF_ALL (40%)** — positive cộng thêm các candidate REJECT; học đồng thời phân loại và
   tính hợp lệ của candidate.

Centroid GT không bị jitter. Validity loss bị tắt ở hai pha chỉ-positive đầu tiên và chỉ
được bật ở OOF_ALL.

## Learning-rate schedule theo pha

Không dùng một cosine toàn cục. Mỗi pha curriculum có schedule riêng.

| Pha | LR type/fusion | LR validity |
|---|---:|---:|
| GT_POS | warmup lên `1e-4`, cosine về `5e-5` | `0` |
| OOF_POS | `7.5e-5` -> `3e-5` | `0` |
| OOF_ALL | `5e-5` -> `5e-6` | `1e-4` -> `1e-5` |

GT_POS dùng warmup ba epoch. Optimizer là AdamW với `weight_decay=1e-4`; gradient clipping
là `1.0`.

## Phơi nhiễm mạnh cho lớp hiếm

Các lớp đuôi:

- plasma cell
- neutrophil
- apoptosis
- melanophage
- endothelium

Mục tiêu của sampler chính trong V13.2 ở batch Stage-2 256:

| Pha | mức phơi nhiễm được đảm bảo, yêu cầu cho mỗi lớp đuôi |
|---|---:|
| GT_POS | 16 / batch |
| OOF_POS | 12 / batch |
| OOF_ALL | 8 / batch trong phần positive |

Sampler nhận biết case và ưu tiên mẫu chưa dùng (unique-first) trong giới hạn ngân sách lặp
lại. Candidate phổ biến bị chặn ở 4 lần lặp/epoch; candidate lớp đuôi ở 12 lần lặp/epoch.
Nếu một lớp đuôi quá ít mẫu để đạt quota mà không vượt trần lặp lại, sampler tự hạ quota
hiệu dụng và ghi lại trong thống kê sampler, thay vì âm thầm overfit cùng mấy nhân đó.

Augmentation dùng phép quay/lật D4 giữ nguyên hình thái cộng với nhiễu stain nhẹ. Mẫu hiếm
được xem nhiều lần hơn, chứ không bị biến dạng mạnh hơn.

## Hard mining

Trong OOF_ALL:

- hard mining bắt đầu từ epoch thứ 4 của pha;
- hard pool được làm mới mỗi 3 epoch;
- khoảng 50% quota reject có thể đến từ hard reject;
- khoảng 25% quota rare được đảm bảo có thể đến từ hard rare positive.

## Các thí nghiệm Stage 2 được giữ lại

V13.2 chủ động thu gọn từ sáu nhánh cũ xuống bốn lượt có kiểm soát:

| Thí nghiệm | Mục đích |
|---|---|
| `V13_2_01_META_CONTROL_BS` | Control META V64+V128, Balanced Softmax, sampler vừa phải |
| `V13_2_02_META_RARE_BS` | **Model chính**: phơi nhiễm lớp hiếm mạnh + hard mining + Balanced Softmax |
| `V13_2_03_META_RARE_CE` | Cùng sampler/chính sách train như model chính nhưng dùng CE thường, để kiểm tra Balanced Softmax còn giúp gì không sau khi đã sửa mạnh phần phơi nhiễm |
| `V13_2_04_META_CONTEXT_RARE_BS` | Cùng chính sách rare/BS như thí nghiệm 02, nhưng thêm context V256 (V64+V128+V256) để tách riêng giá trị của context tissue lớn hơn |

Đã bỏ khỏi vòng sàng lọc V13.2:

- nhánh CB-Focal riêng;
- nhánh CB-CE riêng;
- nhánh RareBoost riêng (phơi nhiễm lớp hiếm giờ là phần của chính sách train chính);
- LoRA.

Lượt sàng lọc 50 epoch thì train cả bốn. Lượt 100 epoch thì chỉ train model đã chọn.

## Batch và profile tốc độ của Stage 2

Mặc định:

```text
Stage-2 effective batch = 256
Stage-2 physical batch  = 256
UNI2-h encoder batch    = 256
```

Fallback CUDA-OOM thử các physical batch nhỏ hơn (`128, 64, 32, 16`) trong khi vẫn giữ
effective batch 256 bằng gradient accumulation.

Các tối ưu tốc độ gồm:

- BF16 autocast khi được hỗ trợ;
- TF32 / cuDNN benchmark ở chế độ nhanh không tiền định;
- AdamW fused/foreach khi được hỗ trợ;
- tensor CUDA channels-last khi có lợi;
- chạy các thí nghiệm tuần tự trên một GPU (không tranh chấp GPU song song);
- cache mảng "nóng" cho cả Colab và máy local;
- cache native-crop cho từng worker;
- worker Stage-2 bền (persistent) cho mỗi pha curriculum;
- cache trong epoch cho feature UNI2-h đã đóng băng của các candidate bị oversample;
- validation thưa: tại biên các pha cộng thêm mỗi hai epoch trong OOF_ALL;
- làm mới hard pool mỗi ba epoch thay vì mỗi epoch.

`02_Train_Stage2.ipynb` bật đường chạy nhanh không tiền định để tối đa throughput
(`FAST_NONDETERMINISTIC = True`). Chỉ dùng chế độ tiền định khi việc chạy lại y hệt quan
trọng hơn tốc độ.

## Chọn checkpoint

### Sàng lọc 50 epoch

Early stopping bị tắt. Mọi thí nghiệm được chạy đủ 50 epoch để so sánh công bằng. Việc chọn
checkpoint bắt đầu từ epoch thứ 6 của pha OOF_ALL và dùng macro-F1 pooled.

### Model thắng 100 epoch

Notebook hỗ trợ đúng profile 30/30/40 đó ở 100 epoch. Nếu bật early stopping bằng tay, chỉ
áp dụng trong OOF_ALL; patience cấu hình là 15 và minimum delta là `0.001`.

## Metric chính

V13.2 dùng F1 theo lớp tính trên TP/FP/FN pooled, rồi lấy trung bình trên mười lớp nhân để
ra macro-F1. F1 theo từng ảnh có thể được log như một chỉ số phụ nhưng không phải metric
chính để chọn checkpoint.

## Train final

Sau lượt sàng lọc 50 epoch:

1. lock một thí nghiệm Stage 2;
2. giữ năm checkpoint A1 đã refit làm ensemble detector khi triển khai;
3. train lại cấu hình Stage 2 đã chọn trên toàn bộ ROI có nhãn;
4. dùng `final_epochs=100` cho model final dự kiến;
5. giữ validity threshold đã chọn ở giai đoạn phát triển trong deployment lock.

Các artifact final chính:

- `stage1_lock.json`
- `stage2_v132_lock.json`
- `stage2_v132_final_<experiment>_100ep_seed0_<hash>.pt`
- `stage2_v132_final_lock.json`

## Suy luận cho Grand Challenge

**Yêu cầu submit offline:** baseline chính thức chạy container với networking bị tắt. Vì
vậy bản submit/container phải chứa file local
`PUMA_pretrained_checkpoints/UNI2-h/uni2_h_model.bin`. Suy luận của V13.2 báo lỗi rõ ràng
khi thiếu file này thay vì cố tải về online.

`puma/pipeline/inference.py` cung cấp cả suy luận ROI local và suy luận triển khai cho PUMA
Track 2.

Bài test baseline chính thức của PUMA mount đúng một file TIFF vào
`/input/images/melanoma-wsi/`. ROI của challenge/test là 1024×1024, đúng bằng kích thước
ROI có nhãn khi train. Vì vậy V13.2 nạp trực tiếp toàn bộ ROI; **không có lớp
macro-tile/WSI streaming**. Để tương thích, wrapper suy luận cũng nhận
`/input/images/melanoma-whole-slide-image/` nếu interface Grand-Challenge dùng alias đó.

Đường chạy Grand-Challenge:

- đọc đúng một file TIFF 1024×1024 từ mount input chính thức;
- chạy Stage 1 với logic tile/overlap 512px nội bộ trên ROI đó;
- gộp/suppress toàn bộ candidate Stage 1 cho cả ROI;
- tính geometry 7-D gốc của Stage 2 trên toàn bộ ROI 1024×1024;
- chạy Stage 2 với crop V64/V128/(V256 nếu có);
- ghi `/output/melanoma-10-class-nuclei-segmentation.json`;
- ghi `/output/images/melanoma-tissue-mask-segmentation/<uuid>.tif`.

Dự đoán centroid được serialize thành các polygon nhỏ đối xứng. Trung bình cộng các đỉnh
polygon đúng bằng centroid của model, và đó chính là toạ độ mà bộ đánh giá nuclei chính
thức sử dụng.

### Cảnh báo về output tissue

V13.2 là pipeline nhân tế bào và **không** train model segmentation tissue. Nếu được cung
cấp một tissue mask thật, wrapper suy luận sẽ kiểm tra và ghi nó. Ngược lại nó chỉ có thể
phát ra một mask toàn background để thoả hợp đồng file của Track 2. Fallback đó hợp lệ về
mặt cấu trúc nhưng không phải một dự đoán tissue có tính cạnh tranh.

## Các thư mục output chính

- tiền xử lý: `PUMA_outputs/`
- Stage 1: `PUMA_stage1_training_outputs/`
- Stage 2: `PUMA_stage2_training_outputs/`

Các artifact tiền xử lý chính:

- `puma_rgb_images.npy`
- `puma_instance_maps.npy`
- `puma_class_maps.npy`
- `puma_centroid_heatmaps.npy`
- `puma_centroid_match_disks_15px.npy`
- `puma_roi_manifest.npy`
- `puma_nuclei_centroids.npy`
- `puma_roi_centroid_offsets.npy`
- `puma_fold_assignments.npy`
- `puma_preprocessing_metadata.json`, trong đó có mục `fold_report` với số ROI, loại
  melanoma và class count của từng fold

Các artifact chính của Stage 1:

- năm checkpoint `stage1_best_A1_IFCRN_PP_fold*_seed0.pt`
- `stage1_results.csv`
- `stage1_lock.json`
- `stage1_oof_candidates.npy`
- `stage1_oof_candidates_metadata.json`

## Chính sách resume và thư mục output

Artifact khi train dùng đúng các thư mục chuẩn sau:

- `PUMA_stage1_training_outputs/`
- `PUMA_stage2_training_outputs/`

Mọi stage train đều lưu một resume checkpoint sau mỗi epoch hoàn tất. Nếu session
Colab/runtime dừng, chạy lại đúng notebook/cấu hình đó thì train tiếp từ epoch kế tiếp. Nếu
resume checkpoint hoặc thư mục output tương ứng bị xoá, phần đó bắt đầu lại từ đầu. Việc bỏ
qua một lượt đã hoàn tất chỉ được phép khi hash cấu hình hiện tại và toàn bộ artifact
checkpoint/prediction cần thiết đều có mặt.

Checkpoint sàng lọc/final của Stage 2 nhúng hash cấu hình mang tính ngữ nghĩa của chúng, và
bản triển khai final còn có một deployment hash riêng cho validity threshold. Chỉ đổi
threshold final thì deployment lock được cập nhật mà không cần train lại phần weight không
đổi. Micro-batch/worker do fallback OOM về mặt vật lý được coi là chi tiết thực thi, không
phải một danh tính thí nghiệm khác.

## Hợp đồng geometry cho ROI 1024

Ảnh train có nhãn của PUMA và input test của challenge đều là ROI 1024×1024. V13.2 giữ
nguyên geometry 7-D gốc của Stage 2:

- `log_nearest_distance`
- `local_density`
- `detector_confidence`
- `border_distance_normalized`
- `microns_per_pixel`
- `x_normalized`
- `y_normalized`

`nearest_distance` và `local_density` chỉ được tính sau khi toàn bộ dự đoán theo tile của
Stage 1 đã được gộp và suppress cho cả ROI. Do đó biên tile nội bộ của Stage 1 không làm
thay đổi geometry của Stage 2. Suy luận Grand-Challenge kiểm tra shape không gian của input
và từ chối input khác 1024×1024, thay vì âm thầm làm lệch phân bố geometry.

## Colab

Cả bốn notebook chạy nguyên trạng trên Colab: cell bootstrap mount Drive và dùng
`/content/drive/MyDrive/Research/PUMA` làm thư mục gốc dự án. Dependency ở đó đến từ
`pip install -r requirements_colab.txt` trong một cell tạm, còn torch do runtime Colab cung
cấp.

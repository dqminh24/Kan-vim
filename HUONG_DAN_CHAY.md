# Hướng dẫn chạy (đợt 7)

Một lệnh chạy toàn bộ đợt 7: `scripts/run_fix.py --dot7`. Hai tập chính CIFAR-100 và Food-101; Tiny-ImageNet và
ImageNet-100 chạy một phần. Bảng kết quả tự sinh: `results_dot7/KET_QUA_DOT7.md`.

**Đang dùng bản gọn** (`--dot7 --lite`, 43 job, xem mục "Bản gọn" bên dưới). Các bước 5 và 6 ghi sẵn lệnh của cả hai bản:
chép đúng dòng của bản mình chạy.

Hướng dẫn của các đợt trước (`scripts/run_plan.py`, Kaggle / Colab, đợt 1–6 của `run_fix.py`) không còn dùng; xem lại
trong lịch sử git: `git show 8e503ed:HUONG_DAN_CHAY.md` và `git show 8e503ed:HUONG_DAN_CHAY_FIX.md`.

## Đợt 7 làm gì

Nội dung (cùng cấu hình đợt 6: giảm ở đầu vào, lr backbone 5e-5, `--prune_norm`, patience 0, mixup 0,2, chưng cất):
- **Food-101** thành tập chính, đủ ma trận như CIFAR-100: baseline × 3 seed (seed 0 lưu làm teacher), KANPool, MLP Pool,
  Avg Pool, Max Pool, KAN Prune, MLP Prune, Static Drop × giữ 75 / 50 / 25% × 3 seed, 30 epoch (`--food_epochs`).
  Ảnh Food-101 giờ dùng quy trình của ImageNet-100: train `RandomResizedCrop(224)` + lật; eval `Resize(256)` + `CenterCrop(224)`.
- **Giảm độ phân giải** (`train.py --img_size`): ảnh 192 / 160 / 112 px = 144 / 100 / 49 token (≈ giữ 75 / 50 / 25%),
  Vim-Ti không có bộ giảm token, pos-embed nội suy từ lưới 14×14, có chưng cất như mọi phương pháp giảm token.
  Chạy trên CIFAR-100 và Food-101, 3 seed.
- **Static Drop** cho Tiny-ImageNet (giữ 50%, seed 0, 10 epoch, teacher của đợt 6).
- **ImageNet-100 làm lại** cả 8 job (baseline + 7 phương pháp kể cả Static Drop, giữ 50%, seed 0, 10 epoch) với val sạch:
  `--val_from test` chia đôi val chính thức (2.500 val / 2.500 test, cân bằng theo lớp). Đợt 6 lấy val từ ảnh train,
  vốn nằm trong tập pretrain ImageNet-1K, nên val bị "nhiễm" (cao hơn test 3–6 điểm, chọn sai epoch).
- Mỗi file json giờ ghi thêm `data` (biến đổi ảnh train / val / test đọc lại từ chính DataLoader, số ảnh), `model_input`
  (kích thước ảnh, số token, cách thu nhỏ), `train_recipe` (mixup, label smoothing, chưng cất, teacher), `env`.

- **Đối chứng Δ** (`--reducer delta_prune`): chọn token bằng Δ có sẵn của Mamba (không học), CIFAR-100, sau khối 6, giữ
  50% / 25%, 3 seed; so ghép cặp với KAN/MLP Pool, Static Drop, KAN Prune của đợt ctx / r25 (cùng cấu hình). Chạy bằng
  bản pure (backbone chính thức không cho đọc Δ), nên chậm hơn ~25%.
- JSON ghi thêm chẩn đoán: KAN/MLP Pool có thật sự ưu tiên token không (entropy trọng số, so với Avg Pool = 1), token
  Prune giữ lại nằm rải rác hay dồn cụm (so với Static Drop), phần thật của từng thành phần loss (CE / L1 / KD).

**99 job: pha 1 (6 job) ~2,5 giờ, pha 2 (93 job, chủ yếu Food-101 ~85–90 phút/job) ~62 giờ trên 2 × RTX 4090.**
Pha 2 xếp theo seed: seed 0 của mọi job Food-101 xong trước (~18 giờ), dừng giữa chừng vẫn có đủ một đường cong.

**Bản gọn `--dot7 --lite` (43 job, ~57 giờ-GPU: ~29 giờ trên 2 × 4090, ~12–15 giờ trên 4 × 4090):** bước 5 và 6 có sẵn
lệnh với `--lite`. **Quên `--lite` ở bất kỳ lệnh `run_fix` nào (kể cả khi chạy lại sau sự cố) là chạy bản đầy đủ 99 job.**
**Không tải ImageNet-100** (bỏ bước 3): có `data/imagenet-100` thì `--lite` chạy thêm 8 job ImageNet-100 (51 job).
- Giữ: Food-101 seed 0 đủ ma trận (baseline làm teacher + 7 phương pháp × 3 mức giữ + giảm độ phân giải × 3), giảm độ
  phân giải CIFAR-100 × 3 seed, Static Drop Tiny-ImageNet.
- Seed 1 / 2 của Food-101 chỉ ở giữ 25%: KANPool, MLP Pool, Static Drop, giảm độ phân giải 112 px (chỗ Pool so với
  Static Drop còn tranh cãi, cần 3 seed để tính KPI). Các ô Food-101 khác chỉ có 1 seed: bảng ghi "tham khảo, không
  tính KPI". Tập test 25.250 ảnh chỉ làm nhỏ nhiễu lấy mẫu ảnh test (~±0,5 điểm ở mức tin cậy 95%), không bỏ được dao
  động giữa các lần train (CIFAR-100: 0,1–0,4 điểm với Pool, tới 0,75–1,35 với Prune).
- Bỏ: baseline Food-101 seed 1 / 2, đối chứng Δ, ImageNet-100 làm lại.
- Giảm độ phân giải CIFAR-100 cả 3 seed chạy ở pha 1, cùng lúc với baseline Food-101 (không cần teacher Food-101).

## Chạy trên máy GPU mới (từng bước)

**0. Thuê máy:** vast.ai / RunPod, **2 × RTX 4090** (không chọn RTX 5090 / RTX PRO Blackwell), image
`pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel`, **disk ≥ 100 GB** (Food-101 ~10 GB, ImageNet-100 ~30 GB, checkpoint,
kết quả). Thời gian chạy ~60 giờ: chọn máy cho thuê dài, không bị ngắt (không dùng "interruptible").
Bản gọn: ~29 giờ trên 2 × 4090 hoặc ~12–15 giờ trên 4 × 4090; nên đặt disk 150 GB.

**1. Đưa code và kết quả đợt 6 lên `/workspace`:** code lấy bằng
`cd /workspace && git clone -b dot7 https://github.com/dqminh24/Kan-vim.git kan-vim` (khi đó bỏ qua lệnh giải nén
`kan-vim-v5.zip` ở bước 2), hoặc upload `kan-vim-v5.zip`; và upload `results_dot6.zip` (trên máy của bạn nằm ở
`result/results_dot6.zip`; cần teacher Tiny-ImageNet và json đợt 6 để so).

**2. Cài đặt (10–30 phút):**
```bash
cd /workspace && python -m zipfile -e kan-vim-v5.zip . && cd kan-vim       # image không có sẵn unzip
bash scripts/setup_rent.sh                     # cài kernel Vim chính thức, pytest, matplotlib, huggingface_hub
python -m zipfile -e /workspace/results_dot6.zip .                          # -> kan-vim/results_dot6/*.json, teacher_*.pt
ls results_dot6/teacher_tiny-imagenet.pt       # phải có (nếu thấy results_dot6/results_dot6/ là giải nén sai chỗ)
python -c "from huggingface_hub import hf_hub_download as d; import shutil; shutil.copy(d('hustvl/Vim-tiny-midclstok', 'vim_t_midclstok_76p1acc.pth'), 'vim_t.pth')"
```

**3. ImageNet-100 (tuỳ chọn, ~30–60 phút; bỏ qua thì đợt 7 tự bỏ 8 job ImageNet-100). Bản gọn: bỏ bước này.** Bản Kaggle của Ambesh
Shekhar, cần API key Kaggle (kaggle.com → Settings → API → Create New Token → file `kaggle.json`).
```bash
pip install -q kaggle && mkdir -p ~/.kaggle && cp /workspace/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d ambityga/imagenet100 -p /workspace/in100 --unzip   # lỗi 404: tìm "imagenet100" trên Kaggle
python scripts/prep_imagenet100.py /workspace/in100      # phải in "thấy 100 lớp train, 100 lớp val", train ~130000, val 5000
```

**4. Kiểm tra (bắt buộc, ~5 phút):**
```bash
python -m pytest -q                            # phải thấy "90 passed"
python scripts/check_impl.py vim_t.pth         # phải in 2 dòng (224 px, 160 px) và USE_OFFICIAL
```
Không thấy `USE_OFFICIAL` thì vẫn chạy được (bản pure, chậm hơn ~25%), nhưng tốc độ không so được với đợt 6: nên
kiểm lại `setup_rent.sh` trước.

**5. Chạy thử (~40–60 phút, gồm tải Food-101 ~5 GB lần đầu):** mỗi loại job một lần, 2 epoch × 20 bước.

Bản gọn (7 job; 6 nếu thiếu teacher Tiny-ImageNet):
```bash
bash scripts/run_fix.sh --dot7 --lite --quick && tail -f run_fix.log   # Ctrl+C chỉ thoát xem, không dừng chạy
```
Bản đầy đủ (10 job):
```bash
bash scripts/run_fix.sh --dot7 --quick && tail -f run_fix.log
```
Chờ log in `XONG CHẠY THỬ` / `XONG TOÀN BỘ` (hoặc `XONG, có job lỗi`: xem `results_quick_dot7/logs/<tên job>.log`). Kiểm:
- `results_quick_dot7/KET_QUA_DOT7.md`, mục "Dữ liệu và biến đổi ảnh": Food-101 và ImageNet-100 là `RandomResizedCrop`,
  CIFAR-100 / Tiny-ImageNet là `RandomCrop(... padding_mode=reflect)`, không có dòng **CẢNH BÁO**.
- `grep -h "distill=\|patch tokens" results_quick_dot7/logs/*res160*.log` phải có `100 patch tokens` và
  `distill=on (teacher 224 px)`.
- Chỉ bản đầy đủ (có tải ImageNet-100): `grep -h "data:" results_quick_dot7/logs/imagenet-100*.log` phải là
  `train ~130000 | val 2500 | test 2500`.
- Đầu `run_fix.log` không có dòng `CẢNH BÁO: thiếu ... teacher`.

**5b. Đo thời gian thật trước khi chi tiền (~2 phút):** ước tính 85–90 phút/job Food-101 lấy từ ImageNet-100, chưa đo.
```bash
python train.py --dataset food101 --data_root data --ckpt vim_t.pth --impl official --batch 64 --workers 12   --val_size 5000 --reducer kan_pool --ratio 0.5 --teacher_ckpt results_quick_dot7/teacher_food101.pt --timing
```
Dòng `TIMING ... 1 epoch ~= X min`: một job Food-101 ≈ 30 × X phút. Cả đợt ≈ (số job Food-101 × 30 × X phút + phần
còn lại) ÷ số GPU: bản đầy đủ 75 job Food-101, phần còn lại ~20 giờ-GPU (CIFAR-100 / ImageNet-100); bản gọn 33 job
Food-101, phần còn lại ~6 giờ-GPU (CIFAR-100 / Tiny-ImageNet). Nếu `data` chiếm phần lớn thời gian mỗi bước thì máy thiếu CPU: tăng `--workers`
hoặc chọn máy nhiều nhân hơn trước khi chạy thật.

**6. Chạy thật (nền, kết quả ở `results_dot7/`, tách riêng với chạy thử):**

Bản gọn (43 job; ~29 giờ trên 2 × 4090, ~12–15 giờ trên 4 × 4090):
```bash
OMP_NUM_THREADS=3 bash scripts/run_fix.sh --dot7 --lite && tail -f run_fix.log
```
Bản đầy đủ (99 job, ~58 giờ trên 2 × 4090):
```bash
OMP_NUM_THREADS=3 bash scripts/run_fix.sh --dot7 && tail -f run_fix.log
```
`OMP_NUM_THREADS=3`: không có nó, mỗi job mở số luồng CPU bằng số nhân của máy và các job tranh nhau CPU.
Bảng `results_dot7/KET_QUA_DOT7.md` cập nhật sau mỗi job. Mất kết nối / máy khởi động lại: chạy lại **đúng lệnh đã dùng
(giữ nguyên `--lite`)**, job xong được giữ, job dở chạy tiếp. Muốn dừng sớm: seed 0 của mọi job Food-101 xong sau ~20 giờ
đầu (bản đầy đủ, 2 GPU); bản gọn khoảng 10 giờ đầu trên 4 GPU, khoảng 21 giờ trên 2 GPU (ước tính).

**7. Sau khi xong (GPU rảnh, không job nào khác chạy):** đo tốc độ cho bài (batch 1 = độ trễ, batch 64 = thông lượng;
dòng đầu in rõ scan đang dùng), đánh giá giảm token **không huấn luyện lại** (vài phút / tập), rồi đóng gói và tải về:
```bash
python scripts/bench.py --ckpt vim_t.pth --impl official --img_sizes 192 160 112 --batches 1 64 --out results_dot7
python scripts/eval_noretrain.py --dataset cifar100 --teacher_ckpt teacher/teacher_cifar100.pt --impl official --out results_dot7
python scripts/eval_noretrain.py --dataset food101 --teacher_ckpt results_dot7/teacher_food101.pt --impl official --out results_dot7
python scripts/eval_noretrain.py --dataset cifar100 --teacher_ckpt teacher/teacher_cifar100.pt --impl pure --reduce_at 6 --img_sizes --out results_dot7
python -c "import shutil; shutil.make_archive('results_dot7', 'zip', '.', 'results_dot7')"   # đóng gói lại kèm bench
```
Kiểm tra mở được file zip trên máy mình rồi mới **Destroy** máy.

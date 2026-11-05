# Kết quả đo tốc độ — Kaggle Tesla T4

Thước đo: **thời gian train xong cùng 400 mẫu, 1 epoch.**
Phân phối độ dài: `p50=103, p95=137, max=184` token · `max_length=512`.
Benchmark trong notebook ghim `CUDA_VISIBLE_DEVICES=0` (single-GPU); mục 2×GPU chạy qua
subprocess `accelerate launch` với `CUDA_VISIBLE_DEVICES=0,1`.

## Bảng đầy đủ (run B — run hoàn chỉnh)

| Cấu hình | Giây | VRAM đỉnh | Nhận định |
|---|---|---|---|
| **2×T4 `accelerate launch --multi_gpu`** | **13.4** | — | ⭐ nhanh nhất |
| pad động | 22.4 | 7.61 GB | — |
| batch 8 | 22.5 | 7.61 GB | trong nhiễu |
| sort theo `length` | 22.6 | 7.61 GB | trong nhiễu |
| `attn=sdpa` | 22.7 | 7.61 GB | trong nhiễu |
| **fp16** | 22.8 | 7.61 GB | trong nhiễu |
| 1×T4 `accelerate launch` | 23.0 | — | mốc so sánh cho 2×GPU |
| batch 16 | 23.0 | **13.86 GB** | không nhanh hơn, VRAM 1.8x |
| `attn=eager` | 23.4 | 8.15 GB | ranh giới nhiễu |
| TRL không packing | 24.1 | **4.98 GB** | chậm ~7%, nhưng **ít VRAM nhất** |
| TRL `packing=True` | 31.6 | 11.88 GB | **1.41x chậm hơn** |
| **bf16** | **77.7** | 8.41 GB | **3.41x chậm hơn fp16** |
| batch 32 | — | OOM | `Tried to allocate 2.43 GiB` |

## Nhiễu đo được từ 2 run độc lập

Notebook chạy 2 lần (run A, run B) với cùng cấu hình → ước lượng được nhiễu thật:

| Cấu hình | run A | run B | lệch |
|---|---|---|---|
| fp16 | 22.8 | 22.8 | 0.2% |
| pad động | 22.1 | 22.4 | 1.5% |
| TRL không packing | 23.8 | 24.1 | 1.4% |
| batch 8 | 22.0 | 22.5 | 2.2% |
| `attn=sdpa` | 22.2 | 22.7 | 2.2% |
| `attn=eager` | 22.8 | 23.4 | 2.6% |
| sort theo length | 22.0 | 22.6 | 2.8% |
| batch 16 | 22.3 | 23.0 | 3.3% |
| TRL `packing=True` | 30.2 | 31.6 | 4.4% |
| bf16 | 72.8 | 77.7 | 6.8% |

**Nhiễu run-to-run trong nhóm cụm: trung bình 2.0%, max 3.3%.**

### Đọc bảng cho đúng — tôi đã nói quá ở bản trước

Bản trước tôi viết *"7 cấu hình đầu nằm trong nhiễu, không phân biệt được"*. Có 2 run rồi thì
thấy **câu đó hơi quá mạnh**. Chính xác hơn:

| Nhóm | Kết luận |
|---|---|
| pad động · batch 8 · sort · `sdpa` · fp16 (22.4–22.8, spread **1.8%**) | **Không phân biệt được** — nhỏ hơn nhiễu 2.0% |
| `attn=eager` (23.4) vs `sdpa` (22.7) — 3.1% | **Ranh giới.** Nhưng VRAM 8.15 vs 7.61 GB lặp lại ở **cả 2 run** → chênh VRAM là thật. Vẫn chọn `sdpa` |
| TRL không packing (24.1) — 7.6% chậm hơn pad động | **Thật nhưng nhỏ**, và đổi lại VRAM thấp nhất (4.98 GB) |
| batch 16 (23.0) | Trong nhiễu về tốc độ, nhưng **VRAM 1.8x** → rõ ràng không đáng |
| packing (31.6) · bf16 (77.7) · 2×GPU (13.4) | **Rõ ràng khỏi nhiễu** |

## 1. ⭐ bf16 emulation vs fp16: chậm hơn 3.41x

```
fp16   22.8s
bf16   77.7s     -> 3.41x cham hon   (run A: 3.19x)
```

T4 là sm_75, **không có bf16 native** → PyTorch chạy emulation, không chạm tensor core.
🔴 `torch.cuda.is_bf16_supported()` vẫn **trả về `True`**. Log của chính notebook:
```
torch.cuda.is_bf16_supported() bao: True
bf16 NATIVE (cap>=8.0)            : False
```
Cách đúng: `torch.cuda.get_device_capability(0)[0] >= 8`.

**Một dòng code quyết định 3.2–3.4x** — lớn hơn mọi tối ưu single-GPU khác cộng lại.

## 2. ⭐ 2×T4 data parallel: nhanh 1.72x

```
-> 1 GPU: 23.0s
-> 2 GPU: 13.4s
2 GPU nhanh hon 1.72x (ly tuong 2.00x)
```

**86% hiệu suất scaling** — tốt hơn tôi dự đoán nhiều.

🔴 **Lời khuyên của tôi đã sai.** Tôi viết: *"2 GPU chỉ lợi khi model/dataset đủ lớn để overhead
đồng bộ không nuốt hết lợi ích"* và *"với model ≤4B thì 1×T4 là đủ; đừng phức tạp hoá"*.
Thực tế với model **0.6B** và chỉ **400 mẫu** — tức trường hợp nhỏ nhất có thể — vẫn được **1.72x**.

⚠️ Nhưng phải làm **đúng cách**:
- `device_map="auto"` **KHÔNG** phải data parallel (nó chia layer = model parallel, không tăng tốc)
- `Trainer` thấy 2 GPU sẽ tự dùng `nn.DataParallel` — **chậm và làm batch nhân đôi âm thầm**
- Phải `accelerate launch --multi_gpu --num_processes=2` với một **script riêng**, không chạy trong cell notebook

## 3. 🔴 BỐN lời khuyên của tôi bị số liệu bác bỏ

| Tôi đã viết | Đo thật | Vì sao tôi sai |
|---|---|---|
| `packing=True` *"đòn mạnh nhất, 2-4x"* | **1.41x CHẬM HƠN**, VRAM 1.6x | Packing nhồi sequence đầy tới `max_length=512` → mỗi step nặng gấp ~4 lần |
| sort theo length *"win gần như free"* | **trong nhiễu** (22.6 vs 22.4) | Collator đã pad theo **batch-max**, không phải `max_length`; phân phối hẹp → hết chỗ tiết kiệm |
| *"tăng batch tới sát trần VRAM"* | batch 16 **không nhanh hơn**, VRAM **1.8x** | GPU bão hoà ở batch 8 với sequence ngắn |
| *"2 GPU chỉ lợi khi model/data đủ lớn"* | **1.72x ngay cả với 0.6B / 400 mẫu** | Đánh giá quá cao overhead đồng bộ |

**Mẫu số chung:** suy luận đúng về hiện tượng, sai về hệ thống thật. Giữ cả bốn trong tài liệu
kèm số bác bỏ — sai ở đâu cũng là thông tin.

> **Đo một lần đáng hơn ba lần suy luận. Đo hai lần mới biết lần đầu có nghĩa gì.**

## 4. `sdpa` vs `eager` + FlashAttention-2

Tốc độ ranh giới nhiễu, nhưng **VRAM 7.61 vs 8.15 GB lặp lại ở cả 2 run** → thật. Dùng `sdpa`.

🔴 **FA2 không dùng được trên T4.** Notebook tự kiểm tra:
```
flash-attn chua cai. Tren sm_75 (T4) thi KHONG cai duoc - can sm_80+.
Card nay: compute cap (7, 5) -> KHONG dung duoc FA2
```

## 5. Tham số còn tồn tại ở transformers 5.22.1 (113 tham số)

| Tham số | Có? |
|---|---|
| `group_by_length` | ❌ đã bỏ |
| `length_column_name` | ✅ còn (flag dùng nó thì mất — API không nhất quán) |
| `warmup_ratio` | ❌ đã bỏ |
| `warmup_steps` · `gradient_checkpointing` · `optim` · `torch_compile` | ✅ |

## 6. Hai bug của tôi, đã sửa và xác nhận hoạt động

**`except` handler tự crash:** `f"batch {bs:<24s}"` với `bs` là `int` → `ValueError`.
Sau fix, run này in đúng: `batch 32: OOM (CUDA out of memory. Tried to allocate 2.43 GiB...)`.
**Handler lỗi cũng phải được test.**

**`train_dp.py` SyntaxError:** `\n` trong f-string thành newline thật khi generator sinh cell —
**cùng loại lỗi, lần thứ ba trong session**. Đã thêm bước notebook tự `ast.parse()` script ngay
sau khi viết file, và lần này log in `train_dp.py: syntax OK` trước khi chạy.

**Lỗ hổng đã để lọt:** `tools/validate_nb.py` kiểm tra cell notebook nhưng **không** kiểm tra
chuỗi script nhúng trong cell. Code sinh ra code thì phải validate code sinh ra.

## 7. Điểm yếu còn lại của chính notebook này

Cảnh báo nhiễu trong ô tổng hợp chỉ gom các cấu hình nằm trong 10% của **cấu hình nhanh nhất**.
Khi 2×GPU (13.4s) nhanh vượt trội thì cụm 22–24s **không được flag** dù nó mới là chỗ cần cảnh báo.
Muốn đúng thì phải gom cụm (clustering) chứ không so với một mốc duy nhất.

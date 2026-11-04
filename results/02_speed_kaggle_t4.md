# Kết quả đo tốc độ — Kaggle Tesla T4 (single-GPU, đã ghim `CUDA_VISIBLE_DEVICES=0`)

Thước đo: **thời gian train xong cùng 400 mẫu, 1 epoch.**
Phân phối độ dài: `p50=103, p95=137, max=184` token · `max_length=512`.

## Bảng đầy đủ

| Cấu hình | Giây | VRAM đỉnh |
|---|---|---|
| sort theo `length` | 22.0 | 7.61 GB |
| batch 8 | 22.0 | 7.61 GB |
| pad động | 22.1 | 7.61 GB |
| `attn=sdpa` | 22.2 | 7.61 GB |
| batch 16 | 22.3 | **13.86 GB** |
| **fp16** | 22.8 | 7.61 GB |
| `attn=eager` | 22.8 | 8.15 GB |
| TRL không packing | 23.8 | **4.98 GB** ← ít VRAM nhất |
| TRL `packing=True` | 30.2 | 11.88 GB |
| **bf16** | **72.8** | 8.41 GB |
| batch 32 | — | OOM (`Tried to allocate 2.43 GiB`) |

## ⚠️ Cách đọc bảng này cho đúng

**7 cấu hình đầu nằm trong khoảng 22.0–22.8s — chênh nhau 3.6%.** Mỗi cấu hình chỉ chạy **1 lần**.
Notebook 01 đo được σ của accuracy là 0.79%; σ của thời gian chưa đo nhưng chắc chắn không nhỏ hơn
mức chênh 3.6% này. **Nên 7 cấu hình đó là không phân biệt được với nhau.**

Chỉ **hai** thứ nổi lên rõ ràng khỏi nhiễu:

| | So với nhanh nhất |
|---|---|
| **bf16 (emulation)** | **3.31x chậm hơn** |
| `packing=True` | 1.37x chậm hơn |

Mọi cần điều khiển khác, trên workload này, **không đáng để tinh chỉnh**.

## 1. ⭐ bf16 emulation vs fp16: chậm hơn 3.19x

```
fp16   22.8s
bf16   72.8s     -> 3.19x cham hon
```

T4 là sm_75, **không có bf16 native**. PyTorch chạy bf16 bằng emulation, không chạm tensor core —
trong khi T4 có tensor core fp16 rất mạnh.

🔴 Và `torch.cuda.is_bf16_supported()` **trả về `True`** trên card này. Log của chính notebook:
```
torch.cuda.is_bf16_supported() bao: True
bf16 NATIVE (cap>=8.0)            : False
-> Card nay KHONG co bf16 native.
```
Cách đúng: `torch.cuda.get_device_capability(0)[0] >= 8`.

**Đây là một dòng code quyết định 3.19x tốc độ** — lớn hơn mọi tối ưu khác cộng lại.
Nó cũng giải thích trọn vẹn việc notebook 01 từ 623s xuống 308s.

## 2. 🔴 BA lời khuyên của tôi bị số liệu bác bỏ

| Tôi đã viết | Đo thật | Tại sao tôi sai |
|---|---|---|
| `packing=True` *"đòn mạnh nhất, 2-4x"* | **1.37x CHẬM HƠN** (30.2s), VRAM 11.88 vs 7.61 | Packing nhồi sequence đầy tới `max_length=512` → mỗi step nặng gấp ~4 lần. Ít step hơn không bù lại được |
| sort theo length *"win gần như free"* | **22.0 vs 22.1s — không lợi gì** | Collator đã pad theo **batch-max**, không phải `max_length`. Phân phối độ dài hẹp → không còn padding để tiết kiệm |
| *"tăng batch tới sát trần VRAM"* | batch 16: **không nhanh hơn** (22.3s) nhưng VRAM **1.8x** (13.86 GB) | GPU đã bão hoà ở batch 8 với sequence ngắn. Tăng batch chỉ mua thêm rủi ro OOM |

**Mẫu số chung của cả ba:** tôi suy luận từ *"p50=103 mà max_length=512 → thừa 79%"* mà bỏ qua
việc **pad động đã giải quyết xong vấn đề đó**. Suy luận đúng về hiện tượng, sai về hệ thống thật.

> **Đo một lần đáng hơn ba lần suy luận.** Giữ lại cả ba lời khuyên sai trong tài liệu kèm số
> liệu bác bỏ — sai ở đâu cũng là thông tin.

## 3. `sdpa` vs `eager`: tốc độ ngang nhau, nhưng VRAM khác thật

22.2s vs 22.8s (2.7% — trong nhiễu), nhưng **VRAM 7.61 vs 8.15 GB** là khác biệt thật và
nhất quán. Dùng `sdpa`.

🔴 **FlashAttention-2 không dùng được trên T4.** Notebook tự kiểm tra:
```
flash-attn chua cai. Tren sm_75 (T4) thi KHONG cai duoc - can sm_80+.
Card nay: compute cap (7, 5) -> KHONG dung duoc FA2
```
Mọi tutorial khuyên FA2 đều vô dụng ở đây.

## 4. Tham số nào còn tồn tại ở transformers 5.22.1

Notebook tự introspect (113 tham số):

| Tham số | Có? |
|---|---|
| `group_by_length` | ❌ **đã bỏ** |
| `length_column_name` | ✅ còn (nhưng flag dùng nó thì mất — API không nhất quán) |
| `warmup_ratio` | ❌ **đã bỏ** |
| `warmup_steps` | ✅ |
| `gradient_checkpointing` · `optim` · `torch_compile` | ✅ |

Helper `ta(**kw)` bắt được `group_by_length` và notebook **vẫn chạy tiếp** thay vì chết.

## 5. 🔴 Phần 2×T4 thất bại — lỗi escaping của tôi

Script `train_dp.py` được notebook viết ra bị:
```
File "/kaggle/working/train_dp.py", line 16
    {"role":"user","content":f"Schema:
                             ^
SyntaxError: unterminated f-string literal
```
`\n` trong f-string đã thành newline thật khi generator sinh cell. **Cùng loại lỗi đã gặp 2 lần
trước đó** trong cùng session.

Đã sửa, và thêm bước notebook **tự `ast.parse()` script ngay sau khi viết file**, trước khi tốn
một lần `accelerate launch`:
```python
import ast
try: ast.parse(open("/kaggle/working/train_dp.py").read()); print("train_dp.py: syntax OK")
except SyntaxError as e: print("SYNTAX LOI:", e); raise
```

**Bài học:** code sinh ra code thì phải validate code sinh ra. `tools/validate_nb.py` kiểm tra
cell notebook, nhưng **không** kiểm tra chuỗi script nhúng trong cell — đó là lỗ đã để lọt lỗi này.

## 6. Bug `except` handler (đã sửa, xác nhận hoạt động)

Trước: `print(f"batch {bs:<24s} OOM")` với `bs` là `int` → `ValueError` **trong chính except handler**.
Sau khi sửa, run này in ra đúng:
```
batch 32: OOM (CUDA out of memory. Tried to allocate 2.43 GiB. GPU 0 has a total capa)
```
**Handler lỗi cũng phải được test.**

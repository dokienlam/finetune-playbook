# Kết quả đo tốc độ — Kaggle Tesla T4 (single-GPU, đã ghim `CUDA_VISIBLE_DEVICES=0`)

Thước đo: **thời gian train xong cùng 400 mẫu, 1 epoch**. Phân phối độ dài: `p50=103, p95=137, max=184` token.

Run này ERROR ở mục batch-size (bug của tôi, xem cuối), nhưng **mọi số trước đó đã ra**.

## 1. ⭐ bf16 emulation vs fp16 — con số lớn nhất

```
-> Tren Tesla T4 (khong co bf16 native): bf16 cham hon fp16 2.83x.
```

**bf16 chậm hơn fp16 2.83 lần** trên T4. Đây là phép đo cô lập đúng biến dtype, và nó
**giải thích trọn vẹn** việc notebook 01 từ 623s xuống 308s (2.02x) — tức dtype là nguyên nhân
chính, không phải việc bỏ `nn.DataParallel`.

Nguyên nhân: T4 là sm_75, **không có bf16 native**. PyTorch chạy bf16 bằng emulation, không
chạm tensor core — trong khi T4 lại có tensor core fp16 rất mạnh.

🔴 Và `torch.cuda.is_bf16_supported()` **trả về `True`** trên card này. Phải dùng
`torch.cuda.get_device_capability(0)[0] >= 8`.

## 2. ⚠️ Chiến lược padding — kết quả NGƯỢC với dự đoán của tôi

| Chiến lược | Giây | vs pad động | VRAM đỉnh |
|---|---|---|---|
| **pad động** (collator pad theo batch-max) | **22.8** | **1.00x** | 7.61 GB |
| sort theo `length` | 22.8 | 1.00x | 7.63 GB |
| TRL không packing | 26.8 | 0.85x | 4.99 GB |
| TRL `packing=True` | 31.1 | **0.73x** | 12.06 GB |

### Hai dự đoán của tôi đều sai

**Sai 1: "`group_by_length` / sort theo độ dài là win gần như free".**
Thực tế **không lợi gì** (22.8s vs 22.8s). Lý do: pad động đã pad theo *batch-max* chứ không
phải `max_length`, và phân phối độ dài ở đây rất hẹp (p50=103, p95=137, max=184) — nên **không
còn padding nào để tiết kiệm**. Sort chỉ có ý nghĩa khi độ dài **phân tán rộng**.

**Sai 2: "`packing=True` là đòn mạnh nhất về thuật toán, có thể 2-4x".**
Thực tế **chậm hơn 0.73x** và tốn **gấp 1.6x VRAM**. Lý do: packing nhồi mỗi sequence đầy tới
`max_length=512`, nên mỗi step xử lý 512 token thay vì ~130. Ít step hơn nhưng mỗi step nặng
hơn nhiều — với mẫu ngắn và chỉ 1 epoch thì không bù lại được.

### Rút ra: packing có lợi khi nào?

Packing tăng **throughput tính theo token**, không phải tăng tốc "train xong tập dữ liệu".
Nó chỉ thắng khi:
- độ dài mẫu **phân tán rộng** (padding động vẫn còn thừa nhiều), **hoặc**
- `max_length` gần `p95` (sequence đầy sẵn, packing không làm nặng thêm), **hoặc**
- bạn tính chi phí theo **token/giây** (ví dụ pretrain), không theo epoch

Với dữ liệu instruction ngắn và đều như ở đây, **pad động đã gần tối ưu** — không cần làm gì thêm.

🔴 **Bài học phương pháp:** tôi suy luận "p50=103 mà max_length=512 → thừa 79% → packing phải
thắng lớn". Suy luận đó bỏ qua việc collator pad theo batch-max, không phải max_length. **Đo một
lần đáng hơn ba lần suy luận.**

## 3. `group_by_length` đã bị bỏ ở transformers v5

Helper `ta(**kw)` bắt được và notebook **vẫn chạy tiếp**, in ra 6 lần:
```
[bo tham so khong ton tai]: ['group_by_length']
```
Đó là mục đích của helper: khi API đổi, code bỏ cần điều khiển không có và **nói cho bạn biết**,
thay vì chết.

## 4. 🔴 Bug làm run này ERROR — của tôi

```python
except torch.cuda.OutOfMemoryError:
    print(f"batch {bs:<24s} OOM")      # bs la int, :s -> ValueError
```
→ `ValueError: Unknown format code 's' for object of type 'int'`

Batch 16 hoặc 32 **thật sự OOM** (`Tried to allocate 1.67 GiB, 1.57 GiB free`), nhưng chính
cái except handler lại crash. **Handler lỗi cũng phải được test.**

Phụ: OOM xảy ra ở 14.56GB dù VRAM đỉnh của pad-động chỉ 7.61GB → có phân mảnh bộ nhớ giữa các
lần bench. Đã thêm `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

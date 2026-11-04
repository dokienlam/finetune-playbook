# Chạy trên Kaggle

## Vì sao Kaggle

| | Kaggle T4 | RTX 5060 Ti |
|---|---|---|
| VRAM | 16GB (hoặc 2×16GB) | 16GB |
| bf16 | ❌ **không** (sm_75) | ✅ (sm_120) |
| Tốc độ | chậm hơn ~2-3x | nhanh hơn |
| Quota | 30h GPU/tuần, 12h/session | không giới hạn |
| Chiếm máy bạn | không | **có** |

Dùng Kaggle khi: máy đang làm việc khác, muốn chạy song song nhiều thí nghiệm, hoặc cần 2×T4.

## Bật GPU + Internet

Notebook → **Settings** (bên phải):
- **Accelerator** → `GPU T4 x2` (hoặc P100)
- **Internet** → `On` ⚠️ **bắt buộc**, không bật thì `load_dataset`/`from_pretrained` fail

## Bẫy đã gặp thật

### 1. T4 không có bf16 — và `is_bf16_supported()` **nói dối** ⚠️ lỗi thật đã gặp

`torch.cuda.is_bf16_supported()` **trả về `True` trên T4 (sm_75)**, vì nó tính cả bf16
*emulation*. Chạy được, nhưng không dùng tensor core → chậm hơn fp16 nhiều, trong khi T4 lại
có tensor core fp16 rất mạnh.

Bằng chứng từ log run đầu tiên của `notebooks/01_llm_sft.ipynb`:
```
compute cap : (7, 5)          <- sm_75, KHONG co bf16 native
bf16=True  =>  dung bf16      <- is_bf16_supported() bao True
Train xong trong 623s
```

**Cách kiểm tra đúng** — dùng compute capability, bf16 native chỉ có từ sm_80 (Ampere):
```python
BF16  = torch.cuda.get_device_capability(0)[0] >= 8
DTYPE = torch.bfloat16 if BF16 else torch.float16
# roi: bf16=BF16, fp16=not BF16
```
(Bản torch mới có `is_bf16_supported(including_emulation=False)` — nhưng check capability
thì chạy được trên mọi bản.)

### 2. torchao cũ phá peft ⚠️ lỗi thật đã gặp
Kaggle cài sẵn `torchao 0.10.0`. `peft 0.21.2` yêu cầu `>=0.16.0` và `is_torchao_available()`
**raise ImportError** ngay khi `get_peft_model()`. Thông báo lỗi không hề nhắc tới LoRA nên rất khó đoán:

```
ImportError: Found an incompatible version of torchao.
Found version 0.10.0, but only versions above 0.16.0 are supported
```

Fix (không dùng torchao thì gỡ hẳn):
```bash
!pip uninstall -q -y torchao
```

### 3. Kaggle dùng Python 3.13
Một số lib ML chưa có wheel cho 3.13 → phải build từ source, rất lâu hoặc fail.

### 4. Đường dẫn
- `/kaggle/input/` — read-only, dataset bạn add vào
- `/kaggle/working/` — ghi được, **là output của notebook**, giới hạn 20GB
- Mọi thứ ngoài `/kaggle/working/` **mất** khi session kết thúc

### 5. Session 12h, nhưng commit chạy lại từ đầu
Bấm "Save Version" → Kaggle chạy lại **toàn bộ** notebook từ trên xuống trong môi trường sạch.
Nên mọi ô phải chạy được theo thứ tự, không phụ thuộc trạng thái bạn tạo tay.

## Dùng Kaggle API để tự động hoá

```bash
uv tool install --with six --with python-slugify "kaggle>=1.8.0"

# Token: kaggle.com/settings -> API -> Generate New Token
mkdir -p ~/.kaggle && echo "KGAT_..." > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token

kaggle kernels push   -p .                       # push va chay
kaggle kernels status <user>/<slug>              # xem trang thai
kaggle kernels output <user>/<slug> -p ./log     # tai log + output ve
```

`kernel-metadata.json`:
```json
{
  "id": "<user>/<slug>",
  "title": "Tieu de",
  "code_file": "notebook.ipynb",
  "language": "python", "kernel_type": "notebook",
  "is_private": "true", "enable_gpu": "true", "enable_internet": "true"
}
```

⚠️ `title` phải "slugify" ra đúng `id`, không thì Kaggle tạo slug khác và bạn push vào nhầm chỗ.

## Dùng cả 2 GPU T4

`device_map="cuda"` chỉ dùng **1 GPU**. Muốn dùng cả hai:
- `device_map="auto"` — chia layer (model parallel). Giúp khi model không vừa 1 card, **không** tăng tốc.
- `accelerate launch --multi_gpu` — data parallel, mới thật sự nhanh hơn.

Với model ≤4B thì 1×T4 là đủ; đừng phức tạp hoá.

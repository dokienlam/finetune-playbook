# Script chạy local (đã verify trên RTX 5060 Ti)

Các script này **đã chạy thật** trên máy local với torch 2.11.0+cu128, transformers 5.22.1.
Chúng dùng `bf16` (Blackwell/Ampere/Ada). Lên Kaggle T4 phải đổi sang `fp16` — dùng
`notebooks/` thay thế, đã xử lý sẵn.

| File | Nội dung | Kết quả |
|---|---|---|
| `00_check.py` | Kiểm tra GPU/torch/bf16 + matmul benchmark | ✅ chạy |
| `01_data.py` | Dữ liệu + label masking (đọc kỹ file này nhất) | ✅ chạy |
| `02_train_lora.py` | LoRA SFT text→SQL | ✅ 108 giây |
| `03_compare.py` | So sánh gốc vs fine-tune | ✅ **0/60 → 44/60 (73.3%)** |
| `train_ocr_lora.py` | VLM/OCR: Qwen3-VL-2B + CORD-v2 → JSON | ⚠️ pipeline verify chạy (4 step, exit 0), **chưa đo chất lượng** |

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch torchvision --torch-backend=cu128      # cu128 BAT BUOC cho RTX 50xx
uv pip install transformers peft trl datasets accelerate
python 00_check.py && python 01_data.py && python 02_train_lora.py && python 03_compare.py
```

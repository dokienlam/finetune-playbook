# VRAM và tối ưu

## Bảng VRAM (nguồn: Unsloth, tối thiểu tuyệt đối)

| Params | QLoRA 4-bit | LoRA 16-bit |
|---|---|---|
| 3B | 3.5 GB | 8 GB |
| 7B | 5 GB | 19 GB |
| 8B | 6 GB | 22 GB |
| 9B | 6.5 GB | 24 GB |
| 11B | 7.5 GB | 29 GB |
| 14B | 8.5 GB | 33 GB |
| 27B | 22 GB | 64 GB |
| 32B | 26 GB | 76 GB |
| 70B | 41 GB | 164 GB |

**16GB (RTX 5060 Ti / Kaggle T4):** QLoRA tới **14B** thoải mái. LoRA 16-bit chỉ tới **~3B**.
**Kaggle 2×T4 = 32GB** nhưng cần model parallel mới dùng được cả hai — không tự động.

## Các cần điều khiển, theo thứ tự nên thử

1. **`per_device_train_batch_size`** ↓ + **`gradient_accumulation_steps`** ↑
   Giữ batch hiệu dụng không đổi. Rẻ nhất, gần như không mất chất lượng.
2. **`gradient_checkpointing=True`** — tiết kiệm nhiều, chậm ~20-30%.
   Unsloth có `use_gradient_checkpointing="unsloth"` tiết kiệm thêm ~30% nữa.
3. **QLoRA 4-bit** — giảm ~4x. Unsloth nói "dynamic 4-bit" gần như không mất chất lượng.
4. **`max_length`** ↓ — nhưng xem phân phối độ dài trước (xem `02-du-lieu.md`), cắt bừa là hỏng nhãn.
5. **`max_pixels`** ↓ (chỉ VLM) — **cần điều khiển mạnh nhất cho VLM**, xem dưới.
6. **`optim="adamw_8bit"`** (cần bitsandbytes) — optimizer state giảm 4x.
7. **`r`** ↓ — ảnh hưởng VRAM ít nhất, đừng sờ tới trước các cái trên.

## VLM: `max_pixels` là vua

Số vision token ≈ `pixels / (28×28)`:

| Ảnh | Vision token |
|---|---|
| 512×512 | ~335 |
| 768×768 | ~750 |
| 1024×1024 | ~1.330 |
| 1280×1280 | ~2.090 |
| 2000×2000 | ~5.100 |

Ảnh gấp **2.6x** chiều → gấp **~7x** token → VRAM nổ. Luôn resize trước khi vào processor:

```python
processor = AutoProcessor.from_pretrained(MODEL, max_pixels=768*768)
img.thumbnail((768, 768))     # va resize that su truoc do
```

⚠️ Với OCR có chữ nhỏ, resize quá mạnh làm mất chữ → model không đọc được. Phải cân bằng:
thử 768 → nếu CER tệ thì tăng lên 1024 và giảm batch.

## Chẩn đoán OOM

```python
print(f"dang dung {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"dinh      {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
torch.cuda.reset_peak_memory_stats()
```

OOM **ngay step đầu** → model + optimizer không vừa: dùng QLoRA hoặc model nhỏ hơn.
OOM **sau vài step** → activation: giảm batch, bật gradient checkpointing, giảm `max_length`.
OOM **lúc eval** → `per_device_eval_batch_size` thấp hơn train, và `max_new_tokens` nhỏ hơn.

## Inference

| Engine | Khi nào dùng |
|---|---|
| `transformers.generate` | Thử nghiệm, notebook. Chậm nhất |
| **vLLM** | Production, nhiều request. PagedAttention + continuous batching |
| SGLang | Tương tự vLLM, mạnh ở structured output |
| llama.cpp / GGUF | CPU hoặc máy yếu, qua Ollama. Đã làm ở `smartkids-finetune` |

Lượng hoá khi deploy (số từ `smartkids-finetune` của bạn — bằng chứng thật):

| Quant | Điểm |
|---|---|
| Q8_0 | 68.4% |
| Q4_K_M | 63.2% |
| **Q2_K** | **18.4%** ⚠️ và 71.1% câu bị cắt |

**Q2_K phá model.** Q4_K_M là sàn thực dụng. Đừng nén dưới 4-bit để lấy vài trăm MB.

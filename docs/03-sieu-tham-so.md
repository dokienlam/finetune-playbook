# Bảng siêu tham số theo modality

Giá trị khởi đầu, không phải chân lý. Nhưng đừng tự nghĩ ra số khác trước khi chạy xong baseline này.

## LLM text (SFT)

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| `learning_rate` | **2e-4** | LoRA chịu lr cao hơn full FT ~100x |
| `num_train_epochs` | **1–3** | >3 epoch trên instruction data: lợi giảm, overfit tăng |
| LoRA `r` | **16** hoặc 32 | Thang: 8/16/32/64/128. Underfit → tăng r |
| `lora_alpha` | **r** hoặc **r×2** | Giữ `alpha/r >= 1` |
| `lora_dropout` | **0** | Lên 0.1 nếu nghi overfit |
| `target_modules` | `q,k,v,o,gate,up,down` | Target hết. Chỉ attention là không đủ |
| batch hiệu dụng | **16** | VD: `batch=2` × `accum=8` |

### ⚠️ Về `packing` và `group_by_length` — đã đo, kết quả ngược trực giác

Trên dữ liệu instruction ngắn và đều (`p50=103, p95=137, max=184` token), đo thật trên T4:

| Chiến lược | vs pad động | VRAM |
|---|---|---|
| pad động (pad theo batch-max) | **1.00x** (nhanh nhất) | 7.61 GB |
| sort theo `length` | 1.00x — **không lợi gì** | 7.63 GB |
| TRL `packing=True` | **0.73x — CHẬM HƠN** | 12.06 GB |

Lý do: collator pad theo **batch-max**, không phải `max_length`. Phân phối độ dài hẹp → không
còn padding để tiết kiệm. Còn `packing` nhồi sequence đầy tới `max_length=512` nên mỗi step
nặng gấp ~4 lần.

**Packing chỉ thắng khi** độ dài phân tán rộng, hoặc `max_length` gần `p95`, hoặc bạn tính
chi phí theo token/giây (pretrain) chứ không theo epoch. Số liệu: [`results/02_speed_kaggle_t4.md`](../results/02_speed_kaggle_t4.md)
| `warmup_steps` | 5–10% tổng step | v5 bỏ `warmup_ratio` |
| `lr_scheduler_type` | `cosine` hoặc `linear` | |
| `weight_decay` | 0.01 | |

## LLM preference (DPO / GRPO)

| Tham số | DPO | GRPO |
|---|---|---|
| `learning_rate` | **1e-6** (default TRL) | **1e-6** |
| `beta` | 0.1 | 0.0 (KL tắt mặc định → không load ref model) |
| `loss_type` | `["sigmoid"]` | `dapo` (default v1.17.1) |
| `max_length` | 1024 | — |
| `num_generations` | — | 8 (group size G) |
| `max_completion_length` | — | 512 |

⚠️ lr cho DPO/GRPO **thấp hơn SFT ~200 lần**. Dùng 2e-4 cho DPO là phá model.
⚠️ `DPOTrainer` yêu cầu `processing_class` có `padding_side="left"`.

## VLM / OCR

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| `max_length` | **None** ⚠️ | BẮT BUỘC. Truncate cắt image placeholder token → crash |
| `learning_rate` | **1e-4** | Thấp hơn LLM-only vì có projector |
| `per_device_train_batch_size` | **1** | VLM ăn VRAM gấp nhiều lần. Bù bằng `accum` |
| `gradient_accumulation_steps` | 8 | |
| `max_pixels` / resize ảnh | **768×768** | Cần điều khiển VRAM số 1 — xem dưới |
| `gradient_checkpointing` | True | |
| LoRA target | chỉ phần ngôn ngữ | Không train vision encoder |

**`max_pixels` là cần điều khiển VRAM quan trọng nhất của VLM.** Số vision token ≈ pixels/(28×28):
- 768×768 → ~750 token
- 1280×1280 → ~2.000 token
- 2000×2000 → ~5.100 token

Ảnh gấp 2.6 lần chiều → gấp ~7 lần token → VRAM nổ. Resize ảnh trước khi vào processor.

## Embedding / Reranker

| Tham số | Giá trị |
|---|---|
| `learning_rate` | 2e-5 (full FT) · 1e-4 (LoRA) |
| epoch | 1–3 |
| batch | **càng to càng tốt** với `MultipleNegativesRankingLoss` (negative lấy trong batch) |

Chọn loss theo hình dạng dữ liệu:

| Dữ liệu bạn có | Loss |
|---|---|
| (anchor, positive) | `MultipleNegativesRankingLoss` ← mạnh nhất, dùng mặc định |
| (anchor, positive, negative) | `TripletLoss` / `CachedMultipleNegativesRankingLoss` |
| (câu A, câu B, điểm 0–1) | `CoSENTLoss` |
| Reranker: (query, doc, nhãn 0/1) | `BinaryCrossEntropyLoss` |

⚠️ Danh sách loss trên lấy từ TOC của sbert.net; **phần body trang docs chưa đọc được đầy đủ** — kiểm tra lại ở https://sbert.net/docs/sentence_transformer/loss_overview.html

## ASR (Whisper)

| Tham số | Giá trị |
|---|---|
| Cách train | **Full fine-tune** (cách chính thống HF), không LoRA |
| `gradient_checkpointing` | True |
| precision | fp16 |
| Model | `openai/whisper-small` (244M) vừa 16GB thoải mái |

`whisper-large-v3` (1.5B) full FT trên 16GB là "có thể nhưng chật" — chưa xác nhận, tự đo.

## Diffusion LoRA

| Tham số | Giá trị |
|---|---|
| Base model khả thi 16GB | `stable-diffusion-v1-5/stable-diffusion-v1-5` (đã xác nhận) |
| Flag bắt buộc cho 16GB | `--gradient_checkpointing --use_8bit_adam` |
| Thêm cho 12GB | `--enable_xformers_memory_efficient_attention --set_grads_to_none` |
| Train text encoder | ❌ cần ≥24GB |

⚠️ SDXL / Flux / Qwen-Image LoRA trên 16GB: community nói được (cần quantize NF4/fp8) nhưng **không xác nhận được từ nguồn chính thức** — trang docs flux/qwen_image đều 404 khi fetch. Kiểm tra `examples/dreambooth/README_flux.md` trên GitHub trước khi tin.

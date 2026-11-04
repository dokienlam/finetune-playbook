# Lộ trình học

> Không mở được WebSearch trong session dựng repo này, nên link khóa học là từ kiến thức model (cutoff 5/2026) — **tự mở xác nhận còn sống**. Ngược lại, toàn bộ phần leaderboard và phiên bản thư viện thì đã fetch trực tiếp từ trang chính thức 11/2026.

## Bước 1 — Hiểu bản chất (1–2 ngày)
**HuggingFace LLM Course** — https://huggingface.co/learn/llm-course
Chương về Supervised Fine-tuning + thư viện TRL. Chính thống nhất, API cập nhật theo lib.

## Bước 2 — Thực hành nhanh (1 tuần)
**Unsloth docs** — https://unsloth.ai/docs *(lưu ý: `docs.unsloth.ai` đã 301 sang `unsloth.ai/docs`)*
Thực dụng nhất. ~40 notebook Colab/Kaggle chạy free, nhanh ~2x, ít VRAM hơn HF thuần.
Notebook đáng chạy nhất cho mục tiêu của repo này:
- Vision: `Qwen3_VL_(8B)-Vision.ipynb`, `Gemma3_(4B)-Vision.ipynb`
- **OCR**: `Deepseek_OCR_(3B).ipynb`, `Deepseek_OCR_2_(3B).ipynb`, `Paddle_OCR_(1B)_Vision.ipynb`
- Speech: `Whisper.ipynb`
- Embedding: xem trang `unsloth.ai/docs/basics/embedding-finetuning`

**HuggingFace smol-course** — https://github.com/huggingface/smol-course
Thiết kế riêng cho model nhỏ trên máy cá nhân.

## Bước 3 — Có hệ thống (2–3 tuần)
**mlabonne/llm-course** — https://github.com/mlabonne/llm-course
LoRA, QLoRA, DPO, ORPO, model merging, quantization.

## Bước 4 — Hiểu sâu
**Raschka — LLMs from Scratch** — https://github.com/rasbt/LLMs-from-scratch
Có chương tự code LoRA từ 0. Đọc mới thật hiểu LoRA là gì, không chỉ gọi API.

**Stanford CS336** — lecture free trên YouTube. Nặng lý thuyết, học sau khi đã làm được.

## Tài liệu tham chiếu khi làm (không phải để "học")
- TRL API: https://huggingface.co/docs/trl/en/index — hiện **v1.17.1**
- sentence-transformers: https://sbert.net/docs/sentence_transformer/training_overview.html — hiện **6.2.0**, có cả trang "Training with PEFT Adapters" và "Training with Unsloth"
- diffusers LoRA: https://github.com/huggingface/diffusers/tree/main/examples/dreambooth

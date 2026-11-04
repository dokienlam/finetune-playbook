# Framework shootout — Kaggle Tesla T4 (single-GPU)

Run này kernel chết ở ô **cuối cùng** (probe `vlmeval`), nhưng **mọi benchmark đã xong trước đó** —
đó chính là mục đích của việc đảo thứ tự "benchmark trước, cài phá môi trường sau".

## 1. ⭐ HF Trainer + PEFT vs TRL SFTTrainer

Cùng model, data, LoRA config, 20 step:

| Framework | s/step | VRAM đỉnh | loss |
|---|---|---|---|
| HF Trainer + PEFT | 1.492 | 4.50 GB | 0.3122 |
| **TRL SFTTrainer** | **1.240** | **3.18 GB** | 0.3121 |

**TRL nhanh hơn 1.20x và tốn ít VRAM hơn 29%, với loss gần như y hệt** (0.3121 vs 0.3122).

Tôi vốn nghĩ TRL chỉ là lớp bọc tiện lợi quanh `Trainer`. Số liệu nói khác: nó có collator và
đường xử lý padding hiệu quả hơn code tôi tự viết. **Dùng `Trainer` để học cơ chế, dùng TRL để làm thật.**

## 2. Inference: transformers vs vLLM

| Engine | tok/s | Chi tiết | VRAM |
|---|---|---|---|
| `transformers.generate` | 108.7 | 1295 tok / 11.9 s | 1.68 GB |
| **vLLM 0.33.0** | **175.8** | 1267 tok / 7.2 s | — |

**vLLM nhanh hơn 1.62x** ở batch 32 prompt. Khoảng cách sẽ rộng hơn nhiều khi số request tăng
(continuous batching mới phát huy). Nhưng cài vLLM mất **4-5 phút** và **ghi đè torch**.

## 3. `lm-eval` chạy được

```
|arc_easy|  1|none|  0|acc|↑ |0.625|± |0.0775|
```
Qwen3-0.6B gốc, `--limit 40`. ⚠️ Sai số ±0.0775 rất lớn vì chỉ 40 mẫu — khi làm thật phải bỏ `--limit`.

## 4. Nhóm an toàn: 15/16 dùng được

| Nhóm | Framework | Bản | Cài (phút) |
|---|---|---|---|
| finetune | transformers | 5.22.1 | 0.3 |
| finetune | peft | 0.21.2 | 0.0 |
| finetune | trl | 1.17.1 | 0.0 |
| finetune | accelerate | 1.15.0 | 0.0 |
| finetune | bitsandbytes | 0.50.2 | 0.1 |
| inference | optimum | ? | 0.0 |
| eval | lm-eval | 0.4.13 | 0.2 |
| eval | lighteval | 0.13.0 | 0.3 |
| eval | mteb | 2.26.33 | 0.1 |
| eval | deepeval | 4.3.3 | 0.1 |
| eval | **ragas** | — | **import lỗi** |
| eval | evaluate | 0.4.6 | 0.0 |
| eval | sentence-transformers | 5.7.0 | 0.0 |
| tracking | wandb | 0.28.1 | 0.0 |
| tracking | mlflow | 3.18.0 | 0.2 |
| tracking | langfuse | 5.1.0 | 0.1 |

⚠️ `sentence-transformers` trên Kaggle resolve về **5.7.0**, local là **6.2.0**. Thêm một lý do
phải `pip freeze` khi báo cáo.

## 5. Nhóm phá môi trường

| Framework | Bản | Cài (phút) | Kết quả |
|---|---|---|---|
| **unsloth** | 2026.11.1 | 0.5 | ✅ |
| **llamafactory** | 0.9.5 | 0.6 | ✅ |
| torchtune | — | 0.0 | ❌ import: `No module named 'torchao.dtypes.nf4tensor'` |
| axolotl | — | 0.3 | ❌ cài: `Failed to build 'zstandard'` |
| sglang | — | 0.5 | ❌ cài: `outlines_core` |
| **llama-cpp-python** | 0.3.36 | 4.7 | ✅ |
| vlmeval | — | — | 🔴 **kernel chết** (`DeadKernelError`) |

**`vlmeval` không có trên PyPI** — phải cài từ GitHub:
```bash
pip install git+https://github.com/open-compass/VLMEvalKit.git
```
Lần trước nó báo `No matching distribution found`; lần này làm **chết hẳn kernel**. Đã bỏ khỏi
danh sách probe.

🔴 **Bài học:** ô lưu báo cáo phải đặt **TRƯỚC** phần probe nhóm phá môi trường, không phải sau.
Kernel chết ở ô cuối là mất toàn bộ `json` tổng hợp — may là log vẫn còn nên cứu được số.

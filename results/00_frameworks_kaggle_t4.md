# Framework shootout — Kaggle Tesla T4 (single-GPU, `CUDA_VISIBLE_DEVICES=0`)

Notebook chạy **2 lần** (run A, run B) → so được số nào tin được, số nào không.

✅ **Lần này `framework_report.json` đã được ghi** dù kernel vẫn chết ở cuối — vì ô lưu báo cáo
đã dời **lên trước** phần probe nhóm phá môi trường. Run trước mất sạch JSON, phải vớt số từ log.

## 1. ⭐ HF Trainer + PEFT vs TRL SFTTrainer

Cùng model, data, LoRA config, 20 step:

| Framework | s/step (A) | s/step (B) | VRAM đỉnh | loss |
|---|---|---|---|---|
| HF Trainer + PEFT (collator tự viết) | 1.492 | 1.419 | **4.500 GB** | 0.3122 / 0.3132 |
| **TRL SFTTrainer** | **1.240** | **1.200** | **3.183 GB** | 0.3121 / 0.3122 |
| | **1.20x** | **1.18x** | **−29%** | y hệt |

**TRL nhanh hơn 1.18–1.20x và tốn ít VRAM hơn 29%, loss y hệt.**

Độ tin cậy:
- Chênh **thời gian** giữa 2 run: HF 4.9%, TRL 3.2% → lợi thế 18–20% **vượt rõ khỏi nhiễu**
- Chênh **VRAM** giữa 2 run: **0.000 GB — y hệt từng byte**, cả hai framework

🔵 **VRAM là thước đo tin cậy hơn thời gian rất nhiều.** Nó tất định; thời gian thì không.
Khi so sánh cấu hình, nếu VRAM khác thì đó là khác biệt thật, không cần chạy nhiều lần.

Tôi vốn nghĩ TRL chỉ là lớp bọc tiện lợi quanh `Trainer`. Số liệu nói khác: collator và đường
xử lý padding của nó tốt hơn code tôi tự viết. **Dùng `Trainer` để học cơ chế, dùng TRL để làm thật.**

## 2. Inference: transformers vs vLLM

| Engine | run A | run B | Chi tiết |
|---|---|---|---|
| `transformers.generate` | 108.7 | 110.2 tok/s | 1295 tok / 11.9 s · VRAM 1.68 GB |
| **vLLM** | **175.8** | **173.9 tok/s** | 1267 tok / 7.2 s |
| | **1.62x** | **1.58x** | |

**vLLM nhanh hơn 1.58–1.62x** ở batch 32 prompt (chênh giữa 2 run chỉ 1.1–1.4% → rất ổn định).
Khoảng cách sẽ rộng hơn nhiều khi số request tăng — continuous batching mới phát huy.

⚠️ Đổi lại: cài vLLM mất **4–5 phút** và **ghi đè torch**. Không cài nó trong notebook train.

## 3. `lm-eval` chạy được

```
lm_eval --model hf --model_args pretrained=Qwen/Qwen3-0.6B \
        --tasks arc_easy --limit 40 --batch_size 8 --device cuda:0

|arc_easy|  1|none|  0|acc|↑ |0.625|± |0.0775|
```
⚠️ Sai số **±0.0775** rất lớn vì chỉ 40 mẫu. Khi báo cáo thật phải bỏ `--limit`.
Con số này giống nhau ở cả 2 run (cùng seed, cùng tập con).

## 4. Nhóm an toàn: 15/16 dùng được

| Nhóm | Framework | Bản | Cài (phút) |
|---|---|---|---|
| finetune | transformers | 5.22.1 | 0.31 |
| finetune | peft | 0.21.2 | 0.03 |
| finetune | trl | 1.17.1 | 0.05 |
| finetune | accelerate | 1.15.0 | 0.03 |
| finetune | bitsandbytes | 0.50.2 | 0.10 |
| eval | lm-eval | 0.4.13 | 0.27 |
| eval | lighteval | 0.13.0 | 0.19 |
| eval | mteb | 2.26.33 | 0.10 |
| eval | deepeval | 4.3.3 | 0.14 |
| eval | **ragas** | — | ❌ **import lỗi** |
| eval | evaluate | 0.4.6 | 0.03 |
| eval | sentence-transformers | 5.7.0 | 0.03 |
| inference | optimum | ? | 0.04 |
| tracking | wandb | 0.28.1 | 0.03 |
| tracking | mlflow | 3.18.0 | 0.24 |
| tracking | langfuse | 5.1.0 | 0.14 |

**`ragas` lỗi vì thiếu dependency, không phải vì không tương thích:**
```
ModuleNotFoundError: No module named 'langchain_community'
```
→ Cách sửa: `pip install ragas langchain-community`

⚠️ `sentence-transformers` trên Kaggle resolve về **5.7.0**, local là **6.2.0**. Thêm một lý do
phải `pip freeze` khi báo cáo kết quả.

## 5. Nhóm phá môi trường

| Framework | Bản | Cài (A) | Cài (B) | Kết quả |
|---|---|---|---|---|
| **unsloth** | 2026.11.1 / .2 | 0.5 | 0.5 ph | ✅ |
| **llamafactory** | 0.9.5 | 0.6 | **13.1 ph** | ✅ nhưng thời gian cài **biến động 22x** |
| torchtune | — | 0.0 | 0.0 | ❌ import: `No module named 'torchao.dtypes.nf4tensor'` |
| axolotl | — | 0.3 | — | ❌ A: `Failed to build 'zstandard'` · **B: kernel chết** |
| sglang | — | 0.5 | — | ❌ `outlines_core` |
| **llama-cpp-python** | 0.3.36 | 4.7 | — | ✅ |
| vlmeval | — | — | — | 🔴 **không có trên PyPI**, lần trước làm chết kernel |

🔴 **`llamafactory` cài mất 13.1 phút ở run B so với 0.6 phút ở run A** — chênh 22 lần. Nguyên
nhân có thể là phải build wheel thay vì lấy wheel sẵn. Nếu bạn đưa nó vào pipeline, **đừng giả
định thời gian cài ổn định**.

### Vì sao notebook này báo ERROR dù dữ liệu đầy đủ

Kernel chết **hai lần, ở hai chỗ khác nhau** trong phần probe nhóm phá môi trường:
run A chết ở `vlmeval`, run B chết ở `axolotl`. Cả hai đều là framework **không cài được trên
Kaggle**; chúng không chỉ fail mà làm sập cả kernel.

Đó là bản chất của việc probe: bạn không thể kiểm tra "cái này có phá môi trường không" mà không
chấp nhận rủi ro bị phá. **Giải pháp không phải là làm nó không chết, mà là lưu dữ liệu trước khi
nó chết** — và điều đó đã hoạt động.

Đã bỏ `vlmeval` và `axolotl` khỏi danh sách probe (cả hai đã xác nhận không cài được, giữ lại
chỉ để đốt thời gian GPU). Cách cài nếu bạn vẫn muốn thử:
```bash
pip install git+https://github.com/open-compass/VLMEvalKit.git   # vlmeval
pip install axolotl --no-build-isolation                          # axolotl, van co the fail o zstandard
```

## 6. Khuyến nghị cho 16GB GPU

| Việc | Dùng | Vì sao |
|---|---|---|
| Fine-tune | **TRL `SFTTrainer`** | nhanh 1.18–1.20x, ít VRAM 29% so với `Trainer` tự viết |
| Fine-tune (nếu cần nhanh hơn nữa) | unsloth | cài được, 0.5 phút — nhưng **chưa đo** ở đây |
| Serving | **vLLM** | nhanh 1.58–1.62x, nhưng ghi đè torch → môi trường riêng |
| Eval học thuật | **lm-eval** | chuẩn, chạy ngay |
| Eval embedding | **mteb** | đường vào VN-MTEB |
| Eval RAG/LLM-judge | ragas | cần thêm `langchain-community` |
| Tracking | wandb · mlflow · langfuse | cả ba đều chạy |
| ❌ Tránh trên Kaggle | torchtune · axolotl · sglang · vlmeval | không cài/import được |

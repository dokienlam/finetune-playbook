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

Bảng đầy đủ, lấy từ `wreckers_partial.json` (run D — run đầu tiên đo được **cả 6**):

| Framework | Bản | Phút | Kết quả |
|---|---|---|---|
| **unsloth** | 2026.11.2 | 0.5 | ✅ |
| **llamafactory** | 0.9.5 | 1.3 | ✅ |
| torchtune | — | 0.0 | ❌ import: `No module named 'torchao.dtypes.nf4tensor'` |
| axolotl | — | 0.3 | ❌ cài: `Failed to build 'zstandard'` |
| sglang | — | 0.6 | ❌ cài: `outlines_core` |
| **llama-cpp-python** | 0.3.36 | 4.4 | ✅ |
| vlmeval | — | — | 🔴 **không có trên PyPI** — bỏ khỏi probe |

🔴 **`llamafactory` cài mất 0.6 / 13.1 / 1.3 phút qua ba run** — biến động **22 lần**. Cùng lệnh,
cùng môi trường. Đừng đưa thời gian cài vào kế hoạch mà không có biên an toàn lớn.

### ⭐ Kernel chết ở đâu — và đó chính là bằng chứng

Run D đo xong cả 6 framework, rồi kernel chết ở **ô kiểm tra ngay sau đó**:

```python
print("torch con nguyen ven khong?")
import importlib, torch.distributed as _d
importlib.reload(_d)        # <- kernel chet o day
```

Phép kiểm tra "torch còn nguyên vẹn không?" không in ra được câu trả lời, vì **chỉ reload
`torch.distributed` thôi đã đủ làm sập kernel**. Câu trả lời là: hỏng nặng hơn cả mức có thể báo
lỗi tử tế.

Đó là bằng chứng mạnh nhất cho bài học của cả notebook này: **đừng cài framework "ghi đè torch"
vào cùng kernel bạn đang train.** Không phải vì nó có thể lỗi — mà vì nó phá môi trường tới mức
chính việc kiểm tra cũng không chạy được.

🔴 **`llamafactory` cài mất 13.1 phút ở run B so với 0.6 phút ở run A** — chênh 22 lần. Nguyên
nhân có thể là phải build wheel thay vì lấy wheel sẵn. Nếu bạn đưa nó vào pipeline, **đừng giả
định thời gian cài ổn định**.

### 🔴 Vì sao notebook này báo ERROR dù dữ liệu đầy đủ — và vì sao không sửa được

Notebook chạy **3 lần**, kernel chết **cả 3 lần**, mỗi lần ở **một package khác nhau**:

| Run | Chết ở | Bằng chứng ngược |
|---|---|---|
| A | `vlmeval` (cuối danh sách) | `llama-cpp-python` trước đó **thành công**, 4.7 phút |
| B | `axolotl` | — |
| C | `llama-cpp-python` | `sglang` trước đó báo `INSTALL-FAIL` **sạch sẽ** |
| D | ô `reload(torch.distributed)` | **cả 6 framework đo xong và đã lưu** ✅ |

**Nguyên nhân không phải một package cụ thể.** Bằng chứng: cùng một package vừa thành công ở
run này lại làm chết kernel ở run khác. Nguyên nhân thật là **hư hại môi trường tích luỹ** — sau
3–4 lần cài/gỡ torch, kernel mong manh và bước build nặng tiếp theo (`llama-cpp-python` biên
dịch C++) làm cạn RAM.

🔴 **Tôi đã sai cách tiếp cận 2 lần:** bỏ `vlmeval`, rồi bỏ `axolotl`, mỗi lần tốn ~40 phút GPU
để rồi cái tiếp theo chết. **Đập chuột chũi.** Đáng lẽ phải nhận ra sau lần thứ hai rằng các
package chết khác nhau nghĩa là package không phải biến số.

### Hai cách đúng

1. **Lưu kết quả trước và sau từng bước** → mất tối đa 1 phép đo. Notebook giờ ghi
   `wreckers_partial.json` sau *từng* framework, không phải sau cả vòng lặp.
   ✅ **Đã kiểm chứng ở run D**: kernel vẫn chết, nhưng cả 6 phép đo đều còn.
2. **Một kernel cho một framework** nếu cần kết quả đầy đủ chắc chắn.

`axolotl` đã được **khôi phục** vào danh sách probe — nó không phải nguyên nhân, và kết quả
`INSTALL-FAIL` của nó là thông tin hữu ích.

**Notebook này sẽ luôn báo `ERROR` trên Kaggle. Đó là dự kiến, không phải lỗi cần sửa** — báo cáo
chính đã lưu xong trước khi phần probe bắt đầu.

### Chi tiết từng framework

Kernel chết **hai lần, ở hai chỗ khác nhau** trong phần probe nhóm phá môi trường:
run A chết ở `vlmeval`, run B chết ở `axolotl`. Cả hai đều là framework **không cài được trên
Kaggle**; chúng không chỉ fail mà làm sập cả kernel.

Đó là bản chất của việc probe: bạn không thể kiểm tra "cái này có phá môi trường không" mà không
chấp nhận rủi ro bị phá. **Giải pháp không phải là làm nó không chết, mà là lưu dữ liệu trước khi
nó chết** — và điều đó đã hoạt động.

Chỉ `vlmeval` bị bỏ — vì nó **không có trên PyPI** nên probe nó bằng `pip install vlmeval` là
vô nghĩa. Cách cài nếu bạn muốn thử:
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

⚠️ `unsloth`, `llamafactory`, `llama-cpp-python` **cài được** nhưng vẫn thuộc nhóm phá môi trường —
cài xong thì torch trong kernel đó không còn tin được nữa. Cho mỗi cái **một notebook riêng.**

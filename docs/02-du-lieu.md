# Dữ liệu — nơi quyết định 80% kết quả

Siêu tham số là chỗ người mới dành 90% thời gian và thu được 10% kết quả. Dữ liệu thì ngược lại.

## Bằng chứng từ chính repo của bạn

`smartkids-finetune` đã chứng minh điều này rõ ràng:

| Version | Dữ liệu | Epoch | Điểm |
|---|---|---|---|
| v1 · Qwen3.5-2B | 434 mẫu | 3 | 87,7% |
| v2 · Qwen3.5-2B | 1.411 mẫu | 2 | **86,8%** ← nhiều data hơn mà TỆ hơn |
| v3 · Qwen3.5-2B | 1.224 mẫu | 1 | **97,4%** |

v2 có gấp 3 lần dữ liệu của v1 nhưng điểm thấp hơn. v3 ít mẫu hơn v2 nhưng hơn 10 điểm.
**Chất lượng và tính nhất quán của nhãn ăn đứt số lượng.** Lỗi "dòng Số dư thừa" mà LLM-judge bắt được ở v2 là ví dụ: một pattern sai nhất quán trong data → model học y nguyên cái sai đó.

## Checklist trước khi train

1. **Tự đọc tay 20 mẫu ngẫu nhiên.** Không script, đọc mắt. Nếu bạn thấy mẫu nào sai/mâu thuẫn, model cũng sẽ học cái sai đó.
2. **Nhãn phải nhất quán về định dạng.** Nếu 90% output là JSON không khoảng trắng và 10% có indent, model sẽ học lẫn lộn.
3. **Kiểm tra leak.** Tập test phải không trùng tập train. Cách chắc nhất: hash nội dung và so sánh.
4. **Kiểm tra phân phối độ dài.** Mẫu dài hơn `max_length` sẽ bị cắt → nhãn cụt → model học sinh ra câu cụt.
   ```python
   lens = [len(tok(x)["input_ids"]) for x in texts]
   print(f"p50={sorted(lens)[len(lens)//2]} p95={sorted(lens)[int(len(lens)*.95)]} max={max(lens)}")
   ```
5. **Giữ một tập held-out "paraphrase".** Diễn đạt lại cùng câu hỏi bằng từ khác. Nếu điểm tụt mạnh trên tập này → model học thuộc câu chữ, không học task. `smartkids-finetune` đã làm đúng cái này (`heldout_paraphrase.jsonl`).

## Label masking — lỗi im lặng phổ biến nhất

Trong SFT, chỉ phần **câu trả lời** được tính loss. Phần prompt phải đặt label `-100`.

```python
p_ids = tok(prompt,  add_special_tokens=False)["input_ids"]
a_ids = tok(answer + tok.eos_token, add_special_tokens=False)["input_ids"]
input_ids = p_ids + a_ids
labels    = [-100] * len(p_ids) + a_ids    # <- then chot
```

Nếu không mask, model bị phạt vì "không đoán được câu hỏi của người dùng" — một việc vô nghĩa. Loss vẫn giảm đẹp, bạn vẫn tưởng đang train tốt, nhưng chất lượng kém hơn rõ.

TRL làm việc này tự động qua `assistant_only_loss=True` (cần `transformers>=5.18.0` nếu dataset có ảnh) hoặc `completion_only_loss`. Nhưng **hãy tự viết tay một lần** để hiểu, rồi mới dùng TRL.

## Bao nhiêu mẫu là đủ?

| Mục tiêu | Số mẫu |
|---|---|
| Đổi định dạng/giọng điệu output | 200–500 |
| Dạy một task mới rõ ràng (text→SQL, ảnh→JSON) | 1.000–5.000 |
| Nhiều task cùng lúc | 5.000–20.000 |
| Thay đổi hành vi sâu | 20.000+ |

Track 01 trong repo này: **4.000 mẫu → 73.3%** ở text→SQL. Nâng lên 20.000 là bài tập mở rộng để tự đo mức lợi tăng thêm.

## Dataset công khai đã xác nhận tồn tại (11/2026)

| Dataset | Dùng cho | Quy mô | License |
|---|---|---|---|
| `b-mc2/sql-create-context` | text→SQL | 78.577 | — |
| `naver-clova-ix/cord-v2` | OCR hóa đơn → JSON | 1.000 (800/100/100) | cc-by-4.0 |
| `darentang/sroie` | OCR hóa đơn + NER | 973 | — |
| `nielsr/funsd` | hiểu form | 199 | — |
| `HuggingFaceM4/DocumentVQA` | hỏi đáp tài liệu | 50.000 | — |
| `HuggingFaceM4/ChartQA` | đọc biểu đồ | 32.700 | **GPL-3.0** ⚠️ |
| `trl-lib/llava-instruct-mix` | VLM SFT | — | — |
| `openbmb/RLAIF-V-Dataset` | VLM DPO | — | — |
| `opendatalab/OmniDocBench` | benchmark OCR | — | — |

⚠️ **Common Voice đã bị gỡ khỏi HuggingFace từ 10/2025** — `mozilla-foundation/common_voice_*` giờ là repo rỗng với thông báo chuyển sang Mozilla Data Collective. Mọi tutorial ASR cũ đều gãy ở dòng `load_dataset`.

Dataset ASR tiếng Việt đã xác nhận có trên Hub (chưa kiểm tra chất lượng/license):
`shunyalabs/vietnamese-speech-dataset`, `ud-nlp/vietnamese-speech-recognition`, `UniDataPro/vietnamese-speech-recognition`, `baohuynhbk14/vietnamese-speech-to-text-preprocessed-whisper-large-v3`

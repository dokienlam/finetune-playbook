# Đánh giá — tại sao loss lừa bạn

## Loss không phải kết quả

Loss đo "model dự đoán token tiếp theo tốt đến đâu trên dữ liệu train". Nó **không** đo
"model làm được việc đến đâu". Hai thứ này lệch nhau thường xuyên.

Track 01 trong repo này: loss giảm 0.57 → 0.05 trông rất đẹp. Nhưng con số có nghĩa là
**0/60 → 44/60** trên dữ liệu chưa từng thấy. Chỉ con số thứ hai đáng kể.

## Ba tầng đánh giá, làm đủ cả ba

### Tầng 1 — Thước đo tự động của chính nhiệm vụ
Rẻ, chạy mỗi lần train. Phải khớp nhiệm vụ:

| Nhiệm vụ | Thước đo |
|---|---|
| text→SQL | khớp chính xác sau chuẩn hoá; tốt hơn nữa: **execution accuracy** (chạy query, so kết quả) |
| ảnh→JSON (OCR) | khớp theo từng field + F1 trên tập key; **không** dùng BLEU |
| phân loại | accuracy, macro-F1 (dùng macro khi lệch lớp) |
| dịch / sinh văn bản | BLEU/chrF — yếu, chỉ dùng kèm tầng 3 |
| ASR | WER, CER |
| embedding | nDCG@10, Recall@k, MRR |
| reranker | nDCG@10 trước/sau rerank |

### Tầng 2 — Tập held-out paraphrase
Diễn đạt lại cùng câu hỏi bằng từ khác. Điểm tụt mạnh → model **học thuộc câu chữ**, không học task.
Repo `smartkids-finetune` của bạn đã làm đúng (`heldout_paraphrase.jsonl`) — giữ thói quen đó.

### Tầng 3 — LLM-as-judge
Bắt được thứ luật chấm tự động không thấy. `smartkids-finetune` là ví dụ tốt: judge phát hiện
nhiều model gốc **khen "em làm đúng" khi học sinh làm sai** (5 × 3 = 16) — luật chấm hoàn toàn bỏ sót.

Làm judge cho đúng:
- **Ẩn danh nhãn model** và **xáo trộn thứ tự** mỗi câu (chống position bias)
- **≥2 judge độc lập** mỗi câu, lấy trung bình
- Rubric cụ thể, thang 1–5, kèm định nghĩa từng mức
- Chấm nhiều chiều riêng biệt (độ đúng / rõ ràng / giọng điệu), không gộp một điểm
- Judge phải là model **khác** model đang đánh giá

## Lỗi đánh giá phổ biến

| Lỗi | Hậu quả |
|---|---|
| Test giao train | Điểm cao giả. **Kiểm tra bằng hash, không bằng niềm tin** |
| Báo cáo một lần chạy | Khác biệt <2σ là nhiễu. Chạy ≥3 seed, báo `mean ± std` |
| Không có baseline model gốc | Không biết fine-tune có giúp gì không |
| Chỉ báo accuracy tổng | Không biết sai **kiểu** gì → không biết sửa gì. Phân loại lỗi |
| So model mình với số in trên paper | Khác prompt/shot/harness → vô nghĩa. Phải tự chạy baseline |
| Bỏ qua tỉ lệ bị cắt token | Câu trả lời cụt bị tính là sai, tưởng model dốt |

`smartkids-finetune` đã bắt đúng điểm cuối: cột *"Bị cắt"* cho thấy Qwen3.8-27B gốc bị cắt **73.7%**
câu trả lời — nếu không đo, bạn sẽ kết luận sai hoàn toàn về model đó.

## Harness chuẩn — dùng cái giới researcher dùng

| Harness | Dùng cho |
|---|---|
| `lm-evaluation-harness` (EleutherAI) | **Chuẩn de-facto** cho LLM text. Báo số không qua đây sẽ bị hỏi |
| `lighteval` (HF) | Nhẹ hơn, tích hợp Hub |
| `mteb` | Embedding/reranker → đường vào MTEB + VN-MTEB |
| `VLMEvalKit` | VLM → đường vào OpenVLM |
| `OmniDocBench` | OCR → **sân tốt nhất cho model nhỏ** |
| `deepeval` / `ragas` | LLM-judge, RAG |

Khi báo cáo **phải ghi**: tên task, số shot, phiên bản harness, commit hash. Thiếu là không reproduce được.

Notebook `notebooks/00_frameworks.ipynb` chạy thử các harness này trên Kaggle và báo cái nào cài được.

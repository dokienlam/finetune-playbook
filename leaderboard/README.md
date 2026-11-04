# Đường vào leaderboard

Trạng thái đã kiểm tra **tháng 11/2026**. Mọi URL submit đều lấy từ trang chính thức.

---

## 1. OmniDocBench — OCR / document parsing ⭐ sân tốt nhất

**Vì sao đây là cơ hội thật:** bảng này được chia cột theo *loại* model (Specialized VLM / General VLM / Pipeline Tool), và **model OCR nhỏ chuyên dụng đang đứng đầu tuyệt đối**, trên cả Gemini 3 Pro, GPT-5.2, Kimi K2.5 1T. Một model 0.8–1.2B train đúng cách hơn GPT-5.2 khoảng 10 điểm. Không có sân nào khác mà GPU 16GB lại không bị thiệt.

- Repo: https://github.com/opendatalab/OmniDocBench (update gần nhất 11/9/2026 — đang sống)
- Dataset: https://huggingface.co/datasets/opendatalab/OmniDocBench
- Docker eval: `ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204`

**Cách submit:** không có nút submit. Bạn tự chạy eval rồi mở PR.
1. Train model OCR của bạn (xem `notebooks/04_ocr.ipynb`).
2. Sinh prediction markdown/JSON theo format OmniDocBench quy định.
3. Chạy eval: `python pdf_validation.py --config <config>`. Lưu ý **setup CPU khá nặng**: TeX Live 2025 (~7GB), ImageMagick 7.1.1 build from source, Ghostscript 9.55 — cho metric CDM (công thức toán).
4. Mở PR thêm `tools/model_infer/<tenmodel>_img2md.py` + cập nhật bảng kết quả (có notebook `tools/generate_result_tables.ipynb` để regen bảng).

⚠️ Luồng submit chính xác là **suy ra từ cấu trúc repo**, README không ghi câu "hãy mở PR". Nên mở Issue hỏi maintainer trước khi dồn sức.

**Base model nên bắt đầu:** `PaddlePaddle/PaddleOCR-VL` (0.96B, Apache-2.0) hoặc `deepseek-ai/DeepSeek-OCR` (3B, MIT) hoặc `stepfun-ai/GOT-OCR2_0` (0.7B, Apache-2.0).

---

## 2. VN-MTEB — embedding tiếng Việt ⭐ sân thưa người

MTEB có subtrack riêng cho tiếng Việt: **`VN-MTEB (vie, v1)`** — ~48 task (retrieval, classification, pair classification, clustering, reranking, STS). Trích dẫn: Pham et al., EACL 2026 Findings.

- Leaderboard: https://leaderboard.mteb.org/ (đã chuyển khỏi HF Space)
- Danh sách benchmark: https://docs.mteb.org/overview/available_benchmarks/
- Hướng dẫn thêm model: https://docs.mteb.org/contributing/adding_a_model/

**Cách submit — nhẹ nhất trong tất cả, chỉ là PR file kết quả:**
1. Tạo `ModelMeta`. Model tương thích sentence-transformers thì auto: `ModelMeta.from_hub("dokienlam/ten-model")`.
2. Chạy eval local: `mteb run -m <model> -t VN-MTEB`. Chỉ cần inference → 0.6B chạy thừa sức trên 16GB, **không cần train để eval**.
3. Mở PR lên https://github.com/embeddings-benchmark/results → merge xong ~1 ngày là lên bảng.
4. Không chạy được thì mở "Evaluation Request" issue, maintainer chạy trong ~7 ngày làm việc (chậm hơn).

**Base model:** `Qwen/Qwen3-Embedding-0.6B` (Apache-2.0, có Matryoshka/MRL).

⚠️ Thực tế: bạn vẫn phải cạnh tranh với các model đa ngữ khổng lồ (Qwen3-Embedding-8B, BGE-M3) vì chúng cũng được eval trên VN-MTEB. "Top 1" không chắc, nhưng **top 3–5 trong một sân thưa là rất khả thi**.

---

## 3. VLSP 2026 — đang mở đăng ký ⏰ hạn gần

Hiệp hội Xử lý ngôn ngữ và tiếng nói tiếng Việt. Trang chủ ghi rõ *"Registration for the shared task is now open."*

- Danh sách task: https://vlsp.org.vn/vlsp2026/eval/
- Đăng ký: https://vlsp.org.vn/vlsp2026/eval/registration

8 task. **Phù hợp nhất với một GPU 16GB:**

| # | Task | Form đăng ký |
|---|---|---|
| 4 | Vietnamese Grammatical Error Correction (VGEC) | https://forms.gle/NGpzmhAqZgWjxFHz5 |
| 6 | Vietnamese Physics Reasoning (VIPRA 2026) | https://forms.gle/nG72dFo8a3UVn1DW8 |
| 7 | Numerical Reasoning QA (ViTNumChart) | https://forms.gle/RTU4ncNog5nmis28A |
| 8 | **Document-Level Legal MT with Limited-Parameter Models** | https://forms.gle/dP5Lc6pgAy2VPH9q6 |
| 1 | Speaker Recognition on Edge Device | https://forms.gle/fD1YaaPkoHKdsyZA8 |

Task #8 tên đã nói thẳng *"Limited-Parameter Models"* — thiết kế cho đúng hoàn cảnh của bạn.
Task #7 (ViTNumChart) cần đọc hiểu biểu đồ → **dùng lại y nguyên machinery VLM ở track 03/04.**

⚠️ **Deadline cụ thể không hiện trên trang eval/registration.** Phải mở từng trang task con hoặc email hỏi ban tổ chức. Đừng chậm — đây là sân sống duy nhất có hạn gần.

---

## 4. Open ASR Leaderboard — khả thi nhưng tốn tiền

- Repo: https://github.com/huggingface/open_asr_leaderboard (README cập nhật 24/7/2026)
- Eval **không chạy trên máy bạn** — chạy trên HF Jobs H200 thuê ($0.0833/phút, ~$5/giờ). Một lượt eval English short-form tốn ~$2.92–5.58.
- GPU của bạn chỉ dùng để *train* trước đó.
- Có track **multilingual** với các slice ngôn ngữ ít tài nguyên → đó là chỗ model nhỏ có cơ hội, chứ không phải bảng WER tiếng Anh.
- Cũng có trade-off plot **speed/size Pareto** — một model nhỏ nhanh có thể nằm trên Pareto front.

---

## 5. Nên bỏ qua

| Sân | Lý do |
|---|---|
| **Open LLM Leaderboard** | Dataset đóng băng từ **3/2025**. Space vẫn "Running" nhưng không xử lý submission mới. HF tự đăng blog *"Open-LLM performances are plateauing"*. Không tìm thấy successor chính thức. |
| **ARC Prize 2026** | Hạn submit **2/11/2026 — đã qua** (hôm nay 4/11). Paper track hạn 8/11. Đợi cycle 2027. Lưu ý luật: submission phải open-source, **không được dùng API model**, chỉ model chạy local — rất khớp với workflow fine-tune local. |
| **OpenVLM Leaderboard** | 220+ model, 80+ benchmark, top toàn VLM khổng lồ (GPT, Gemini, InternVL). Không xác nhận được có subtrack theo size. |
| **AIcrowd** | Mọi challenge hiển thị đều "Completed". Không có gì đang mở. |
| **SemEval 2027** | Mới ở giai đoạn *call for task proposals* — chưa tới lượt thí sinh. |
| **DrivenData** | Chỉ 1 comp đang mở có giải ($300k GEMS) nhưng là địa chất, không liên quan. Có "What's Up, Docs? Document Summarization with LLMs" nhưng là practice, không giải. |

---

## Chiến lược đề xuất

1. **Tuần 1–2:** VLSP 2026 — đăng ký ngay (hạn gần nhất). Chọn task #8 hoặc #7.
2. **Tháng 1–2:** VN-MTEB — chi phí submit thấp nhất, chỉ cần PR file JSON. Lấy kết quả làm portfolio.
3. **Tháng 2–4:** OmniDocBench — đầu tư lớn nhất (setup eval nặng) nhưng là chỗ duy nhất một model nhỏ của bạn có thể **thật sự đứng trên GPT-5.2** trong bảng công khai.

Không nhắm Open LLM Leaderboard hay OpenVLM. Hai sân đó bạn không thắng được bằng 16GB, và một sân đã chết.

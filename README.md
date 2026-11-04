# Finetune Playbook

Lộ trình **bài bản** để fine-tune mọi loại model — text LLM, vision-language, OCR, embedding, reranker, speech, diffusion — và **chiến lược để thật sự lọt top leaderboard** với một GPU 16GB hoặc GPU miễn phí của Kaggle.

> Repo này **không dạy lại LoRA cơ bản**. Nếu bạn cần cái đó, xem [`smartkids-finetune`](https://github.com/dokienlam/smartkids-finetune) — đã có QLoRA/Unsloth, benchmark 13 model, LLM-judge, GGUF→Ollama.
> Repo này giải quyết phần còn thiếu: **đa modality + phương pháp luận tối ưu + đường vào bảng xếp hạng.**

---

## Phần quan trọng nhất: bạn có thể thắng ở đâu

Đã research trạng thái **live tháng 11/2026**. Ba sân khả thi, một sân nên bỏ:

### 🥇 OmniDocBench (OCR / document parsing) — sân tốt nhất
Bảng này **bị model OCR nhỏ chuyên dụng thống trị**, không phải model khổng lồ:

| Model | Size | Overall |
|---|---|---|
| OXR-1.0 | — | **97.03** |
| TeleOCR | 1.2B | **96.91** |
| OvisOCR2 | 0.8B | **96.47** |
| PaddleOCR-VL-1.6 | 0.9B | **96.34** |
| MinerU2.5-Pro | 1.2B | 95.75 |
| GLM-OCR | 0.9B | 95.22 |
| *Gemini 3 Pro* | *khổng lồ* | *92.91* |
| *Qwen3-VL-235B* | *235B* | *89.78* |
| *GPT-5.2* | *khổng lồ* | *86.59* |
| *Kimi K2.5* | *1T* | *84.53* |

Một model 0.9B **đánh bại GPT-5.2 10 điểm**. Đây là chỗ duy nhất mà GPU 16GB không bị thiệt thòi về bản chất.
→ [`leaderboard/README.md`](leaderboard/README.md) có cách submit.

### 🥈 VN-MTEB (embedding tiếng Việt) — sân thưa người
MTEB có subtrack **VN-MTEB** (~48 task tiếng Việt, Pham et al., EACL 2026 Findings). Rất ít lab tối ưu riêng cho tiếng Việt. Submit chỉ là **PR file kết quả** lên `github.com/embeddings-benchmark/results` — không cần serve model. Một `Qwen3-Embedding-0.6B` fine-tune tốt cho tiếng Việt có cơ hội top thật.

### 🥉 VLSP 2026 — **đang mở đăng ký, hạn gần**
8 shared task tiếng Việt. Task #8 tên thẳng là *"Document-Level Legal MT with **Limited-Parameter Models**"* — thiết kế cho đúng hoàn cảnh của bạn. Sân học thuật Việt Nam thường chỉ vài chục team.

### ❌ Đừng mất thời gian
- **Open LLM Leaderboard** — đóng băng từ 3/2025, không nhận submission nữa.
- **ARC Prize 2026** — hạn 2/11/2026, **đã qua**. Đợi cycle 2027.
- **OpenVLM Leaderboard** — 220+ model, top toàn VLM khổng lồ, không có subtrack theo size.

---

## Trạng thái code trong repo này — đọc trước khi tin

Tôi ghi thật, không tô hồng:

| Track | Trạng thái | Bằng chứng |
|---|---|---|
| `01` LLM SFT (text→SQL) | ✅ **Đã chạy thật trên 2 GPU, có số** | **RTX 5060 Ti:** 0/60 → 44/60 (73.3%), 108s · **Kaggle T4:** 0/60 → **46/60 (76.67%)**, 623s. Qwen3-0.6B + LoRA r=16, 4000 mẫu. ⚠️ Hai số **không so trực tiếp được** — khác tập eval và khác dtype path; đó đúng là lý do cần multi-seed (mục 10b trong notebook). |
| `04` OCR / VLM | ⚠️ **Pipeline đã verify chạy được** | Qwen3-VL-2B + CORD-v2, TRL tự chọn `DataCollatorForVisionLanguageModeling`, 4 step chạy sạch exit 0. **Chưa đo chất lượng.** |
| `02` DPO | ❌ Chưa chạy | Code viết theo TRL 1.17.1 docs, chưa verify |
| `03` VLM general | ❌ Chưa chạy | Dùng chung machinery với `04` |
| `05` Embedding + Reranker | ❌ Chưa chạy | sentence-transformers 6.2.0 |
| `06` Speech ASR | ❌ Chưa chạy | ⚠️ xem cảnh báo Common Voice bên dưới |
| `07` Diffusion LoRA | ❌ Chưa chạy | SD1.5 là lựa chọn thực tế nhất cho 16GB |

**Mọi track ❌ phải coi là bản nháp cần bạn tự verify.** Track ✅ và ⚠️ đã chạy trên máy thật.

---

## Môi trường: phiên bản đã verify (11/2026)

```
torch                 2.11.0+cu128     # cu128 BẮT BUỘC cho Blackwell sm_120 (RTX 50xx)
torchvision           0.26.0+cu128     # Qwen3-VL kéo theo video processor, thiếu là crash
transformers          5.22.1           # v5! API khác v4 nhiều
trl                   1.17.1
peft                  0.21.2
datasets              5.1.0
sentence-transformers 6.2.0
```

### Bẫy đã gặp thật, đã fix

1. **`transformers` v5 bỏ `warmup_ratio`** → dùng `warmup_steps`. Mọi tutorial viết cho v4 sẽ lỗi `TypeError`. Cách tra nhanh:
   ```python
   import inspect; from transformers import TrainingArguments
   print([p for p in inspect.signature(TrainingArguments.__init__).parameters if 'warm' in p])
   ```
2. **`SFTConfig` dùng `max_length`, không còn `max_seq_length`.**
3. **Fine-tune VLM phải đặt `max_length=None`** — truncate sẽ cắt đứt image placeholder token → crash.
4. **Qwen3-VL cần `torchvision`** dù bạn không dùng video.
5. **Kaggle T4 = sm_75, KHÔNG có bf16 — và `torch.cuda.is_bf16_supported()` trả về `True` trên nó.** Hàm đó tính cả bf16 *emulation*, nên auto-detect kiểu `BF16 = torch.cuda.is_bf16_supported()` sẽ chọn sai và chạy chậm. Phải dùng `torch.cuda.get_device_capability(0)[0] >= 8`. Run đầu của notebook `01` đã rơi vào đúng bẫy này.
6. **Common Voice đã bị gỡ khỏi HuggingFace (10/2025)** → mọi tutorial ASR cũ dùng `mozilla-foundation/common_voice_*` đều gãy. Giờ phải qua Mozilla Data Collective.

---

## Bắt đầu

- Chạy trên **Kaggle** (khuyến nghị — GPU free, không chiếm máy): [`docs/06-kaggle.md`](docs/06-kaggle.md)
- Muốn **vừa chạy vừa học**: mỗi notebook trong [`notebooks/`](notebooks/) có markdown giải thích từng bước + mục **Thí nghiệm** ở cuối với kết quả kỳ vọng để bạn tự đối chiếu.
- Phương pháp luận: [`docs/01-cay-quyet-dinh.md`](docs/01-cay-quyet-dinh.md) → đọc cái này **trước khi** fine-tune bất cứ thứ gì.

## Mục lục

| File | Nội dung |
|---|---|
| [`docs/00-lo-trinh.md`](docs/00-lo-trinh.md) | Khóa học nên theo, thứ tự, mất bao lâu |
| [`docs/01-cay-quyet-dinh.md`](docs/01-cay-quyet-dinh.md) | Có nên fine-tune không? LoRA/QLoRA/full? |
| [`docs/02-du-lieu.md`](docs/02-du-lieu.md) | Dữ liệu — chỗ quyết định 80% kết quả |
| [`docs/03-sieu-tham-so.md`](docs/03-sieu-tham-so.md) | Bảng siêu tham số theo từng modality |
| [`docs/04-danh-gia.md`](docs/04-danh-gia.md) | Đánh giá đúng — tại sao loss lừa bạn |
| [`docs/05-vram-toi-uu.md`](docs/05-vram-toi-uu.md) | Bảng VRAM + các cần điều khiển bộ nhớ |
| [`docs/06-kaggle.md`](docs/06-kaggle.md) | Chạy trên Kaggle, bẫy T4 |
| [`docs/07-frameworks.md`](docs/07-frameworks.md) | Framework nào cho việc gì + bẫy API đã gặp thật |
| [`leaderboard/README.md`](leaderboard/README.md) | Cách submit từng leaderboard |
| [`notebooks/`](notebooks/) | **Notebook Kaggle-ready** — chạy được cả T4 (fp16) và Blackwell (bf16) |
| [`src/`](src/) | Script đã verify chạy trên RTX 5060 Ti |
| [`results/`](results/) | **Dữ liệu đo thật** — output run Kaggle + bảng 24 framework + các bẫy phát hiện được |

## Notebook

| Notebook | Nội dung |
|---|---|
| [`00_frameworks.ipynb`](notebooks/00_frameworks.ipynb) | **Framework shootout đo thật**: probe 24 framework (fine-tune/inference/eval/tracking), benchmark HF Trainer vs TRL, transformers vs vLLM, chạy `lm-eval`, và đo **phương sai multi-seed** — phần quyết định một kết quả có đáng tin hay không |
| [`01_llm_sft.ipynb`](notebooks/01_llm_sft.ipynb) | Full pipeline: data → label masking → train (có val loss) → **3 biểu đồ** → phân loại lỗi → **inference** → merge & xuất model |

Mỗi notebook có mục **Thí nghiệm** ở cuối: bảng các thay đổi kèm **kết quả kỳ vọng** để tự đối chiếu — đó là phần "vừa chạy vừa học".

## Giấy phép
MIT cho code trong repo. Model và dataset giữ giấy phép riêng — chú ý `Llama-3.2` (license riêng của Meta), `gemma-*` (license riêng của Google), `MinerU` (AGPL-3.0), `ChartQA` (GPL-3.0).

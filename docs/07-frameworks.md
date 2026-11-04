# Framework — cái nào dùng cho việc gì

> Bảng này là **tham chiếu**. Số đo thật nằm ở [`notebooks/00_frameworks.ipynb`](../notebooks/00_frameworks.ipynb)
> — notebook đó cài + import từng framework trong subprocess riêng trên Kaggle T4 rồi báo cái nào
> thật sự chạy được, cài mất bao lâu, nhanh chậm thế nào. **Đừng tin bảng, hãy đọc số.**

## Fine-tuning

| Framework | Khi nào dùng | Đánh đổi |
|---|---|---|
| **transformers `Trainer` + peft** | **Học**. Kiểm soát hoàn toàn, tự viết collator, tự mask label | Verbose, dễ sai label masking |
| **TRL** | **Sản xuất**. Mask label + collator tự động, có sẵn DPO/GRPO/KTO/Reward/Distillation, hỗ trợ VLM | API đổi khá nhanh giữa các bản |
| **Unsloth** | Khi VRAM/thời gian là nút cổ chai. Nhanh ~2x, ít VRAM hơn, ~40 notebook sẵn | **Ghi đè phiên bản thư viện** → nên chạy session riêng |
| **LLaMA-Factory** | Muốn config YAML + WebUI, không viết code | Trừu tượng hoá nhiều, khó debug khi lỗi |
| **Axolotl** | Pipeline production config-driven, multi-GPU | Cài nặng, dependency khó |
| **torchtune** | PyTorch native, muốn ít abstraction | Ít model hỗ trợ hơn HF |

**Khuyến nghị:** học bằng `Trainer`, làm bằng `TRL`, tối ưu bằng `Unsloth`.

### TRL 1.17.1 có những trainer gì

Mature: `SFTTrainer`, `DPOTrainer`, `KTOTrainer`, `GRPOTrainer`, `RLOOTrainer`, `RewardTrainer`, `DistillationTrainer`
Experimental: `ORPOTrainer`, `CPOTrainer`, `OnlineDPOTrainer`, `A2POTrainer`, `GMPOTrainer`, `AsyncGRPOTrainer`, `GKDTrainer`, `SDPOTrainer`, `SDFTTrainer`, `GOLDTrainer`, `IWOPDTrainer`, `AsyncDistillationTrainer`

⚠️ **`PPOTrainer` không còn trong taxonomy v1.17.1.** Code cũ dùng PPO sẽ không chạy.

### Bẫy API đã gặp thật

| Lỗi | Nguyên nhân | Fix |
|---|---|---|
| `TypeError: ... unexpected keyword argument 'warmup_ratio'` | transformers v5 bỏ nó | `warmup_steps=N` |
| `TypeError: ... unexpected keyword argument 'group_by_length'` | transformers v5 **cũng bỏ** nó | Sort dataset: `ds.sort("length")` — chạy được mọi bản |
| `SFTConfig` không có `max_seq_length` | đổi tên | `max_length` |
| VLM crash lúc train | truncate cắt image placeholder token | `max_length=None` |
| `ImportError: incompatible version of torchao` khi `get_peft_model()` | Kaggle có torchao 0.10.0, peft 0.21.2 cần ≥0.16.0 | `pip uninstall -y torchao` |
| `Qwen3VLVideoProcessor requires Torchvision` | Qwen3-VL kéo theo video processor | `pip install torchvision` |

Cách tra tên tham số còn tồn tại, thay vì tin tutorial cũ:
```python
import inspect
from transformers import TrainingArguments
print([p for p in inspect.signature(TrainingArguments.__init__).parameters if 'warm' in p])
```

## Inference

| Engine | Dùng cho | Ghi chú |
|---|---|---|
| `transformers.generate` | Thử nghiệm, notebook | Chậm nhất. Nhớ `padding_side="left"` khi generate theo batch |
| **vLLM** | Production nhiều request | PagedAttention + continuous batching. Cài lâu (~5-10 phút) |
| **SGLang** | Tương tự vLLM | Mạnh ở structured/JSON output |
| **llama.cpp / GGUF** | CPU, máy yếu, Ollama | Đường bạn đã đi ở `smartkids-finetune` |
| TensorRT-LLM | NVIDIA, tối đa throughput | Setup phức tạp nhất |
| `optimum` | ONNX/OpenVINO export | Khi deploy ngoài GPU NVIDIA |

## Đánh giá

| Harness | Dùng cho | Vì sao |
|---|---|---|
| **lm-evaluation-harness** (EleutherAI) | LLM text | **Chuẩn de-facto.** Paper và Open LLM Leaderboard cũ đều dùng |
| **lighteval** (HF) | LLM text | Nhẹ hơn, tích hợp Hub |
| **mteb** | Embedding / reranker | → MTEB + **VN-MTEB** leaderboard |
| **VLMEvalKit** (OpenCompass) | VLM | 220+ model, 80+ benchmark → OpenVLM leaderboard |
| **OmniDocBench** | OCR / document | → bảng OCR, **sân tốt nhất cho model nhỏ** |
| `deepeval`, `ragas` | LLM-judge, RAG | Khi không có đáp án chuẩn |
| `evaluate` (HF) | metric lẻ | BLEU/ROUGE/WER/F1. Tiện, không phải harness |

## Theo dõi thí nghiệm

| Tool | Dùng cho |
|---|---|
| **W&B** | Loss curve, so sánh run, sweep siêu tham số. Chuẩn trong giới research |
| **MLflow** | Tự host được, mạnh về model registry |
| TensorBoard | Nhẹ nhất, có sẵn trong transformers (`report_to=["tensorboard"]`) |
| **Langfuse** | **Trace inference ở production**, không phải training. Bạn đã dùng ở `internal-chat` |

## Những gì researcher làm mà tutorial không nói

1. **Multi-seed.** Chạy ≥3 seed, báo `mean ± std`. Một lần chạy không phải một kết quả.
   Cải thiện nhỏ hơn ~2σ là nhiễu. Notebook `00` có ô đo chính xác con số σ này.
2. **Ablation một biến mỗi lần.** Bảng: data size × rank × lr × epoch.
3. **Ghi cả thí nghiệm thất bại.** Đó là thông tin.
4. **Baseline model gốc trong cùng bảng.** Không có thì không biết fine-tune giúp gì.
5. **Ghim và báo cáo mọi version.** `pip freeze > requirements.txt`, commit hash, GPU, dtype.
   **bf16 và fp16 cho kết quả khác nhau** — phải ghi rõ.
6. **Báo cáo chi phí compute** (GPU-giờ). Reviewer sẽ hỏi.
7. **Model card.** Data gì, license gì, giới hạn gì, bias gì, làm được/không làm được gì.

Checklist đầy đủ ở cuối [`notebooks/00_frameworks.ipynb`](../notebooks/00_frameworks.ipynb).

# Cây quyết định — đọc trước khi fine-tune bất cứ thứ gì

## Câu hỏi 0: có cần fine-tune không?

Phần lớn người fine-tune khi lẽ ra không nên. Theo thứ tự **rẻ → đắt**:

| Vấn đề của bạn | Giải pháp đúng | Fine-tune có giúp? |
|---|---|---|
| Model không biết fact/dữ liệu nội bộ | **RAG** | Không. Fine-tune nhồi kiến thức rất kém và dễ gây hallucinate. |
| Model trả lời sai định dạng | **Prompt + structured output / JSON schema** | Chỉ khi prompt đã thất bại |
| Model trả lời đúng nhưng dài dòng / sai giọng | **Fine-tune** ✅ | Đây là chỗ fine-tune mạnh nhất |
| Cần model nhỏ làm được việc của model to | **Fine-tune (distillation)** ✅ | Đúng mục đích |
| Cần model làm task mà prompt không diễn tả nổi | **Fine-tune** ✅ | VD: ảnh hóa đơn → JSON |
| Latency/chi phí quá cao | Quantize / model nhỏ hơn / caching | Gián tiếp (model nhỏ + fine-tune) |
| Model từ chối việc hợp lệ | Sửa system prompt trước | Hiếm khi cần |

**Quy tắc:** fine-tune dạy model **cách làm**, RAG cấp cho model **thông tin**. Lẫn hai cái này là lỗi phổ biến nhất.

Kết quả thật từ track 01 minh họa đúng điểm này: Qwen3-0.6B gốc đạt **0/60** ở text→SQL — không phải vì thiếu kiến thức SQL, mà vì nó **không biết phải xuất ra cái gì**. Fine-tune 108 giây → 44/60. Prompt engineering không giải quyết được ở model 0.6B.

## Câu hỏi 1: LoRA, QLoRA, hay full fine-tune?

```
Model ≤ 1B và bạn có ≥16GB?
├─ Có  → LoRA 16-bit, hoặc full FT nếu ≤0.6B. Nhanh, chất lượng cao nhất.
└─ Không
   └─ Model ≤ 14B?
      ├─ Có → QLoRA 4-bit. Mặc định nên chọn.
      └─ Không → thuê GPU, hoặc chọn model nhỏ hơn.
```

Bảng VRAM chính thức từ Unsloth (tối thiểu tuyệt đối):

| Params | QLoRA 4-bit | LoRA 16-bit |
|---|---|---|
| 3B | 3.5 GB | 8 GB |
| 7B | 5 GB | 19 GB |
| 8B | 6 GB | 22 GB |
| 14B | 8.5 GB | 33 GB |
| 27B | 22 GB | 64 GB |
| 32B | 26 GB | 76 GB |
| 70B | 41 GB | 164 GB |

**Trên 16GB:** QLoRA thoải mái tới 14B. LoRA 16-bit chỉ tới ~3B (7B đã cần 19GB).
**Trên Kaggle T4 (16GB, không bf16):** giống trên, nhưng phải fp16.

Unsloth khuyến nghị **QLoRA 4-bit làm mặc định** vì "dynamic 4-bit" đã gần như lấp xong khoảng cách chất lượng so với 16-bit.

## Câu hỏi 2: SFT, DPO, hay GRPO?

| Bạn có dữ liệu gì | Dùng |
|---|---|
| (input → output đúng) | **SFT** — luôn làm bước này trước |
| (input → đáp án được chọn, đáp án bị loại) | **DPO** |
| Chỉ có nhãn tốt/xấu đơn lẻ, không theo cặp | **KTO** |
| Có hàm chấm điểm tự động (math, code, test pass) | **GRPO** — mạnh nhất cho reasoning |
| Có model teacher to hơn | **DistillationTrainer** |

Thứ tự thực tế: **SFT trước, luôn luôn.** DPO/GRPO là tinh chỉnh trên nền SFT, không thay thế nó.

TRL 1.17.1 có đủ: `SFTTrainer`, `DPOTrainer`, `GRPOTrainer`, `KTOTrainer`, `RewardTrainer`, `RLOOTrainer`, `DistillationTrainer`, và nhóm experimental (`ORPOTrainer`, `CPOTrainer`, `OnlineDPOTrainer`, `GKDTrainer`...). **`PPOTrainer` không còn trong taxonomy v1.17.1.**

## Câu hỏi 3: train phần nào của model?

| Modality | Train | KHÔNG train | Vì sao |
|---|---|---|---|
| LLM text | tất cả linear: `q,k,v,o,gate,up,down` | — | Unsloth: target hết mới match được full FT |
| VLM / OCR | LoRA phần ngôn ngữ (+ projector) | **vision encoder** | Encoder đã nhìn tốt; data nhỏ mà train nó là hỏng biểu diễn thị giác |
| Embedding | toàn bộ hoặc LoRA | — | Model nhỏ (0.3–0.6B), full FT khả thi |
| ASR Whisper | full fine-tune (cách chính thống của HF) | — | LoRA có dùng trong community nhưng guide chính thức là full FT |
| Diffusion | LoRA trên UNet/transformer | text encoder (cần ≥24GB) | Train text encoder không vừa 16GB |

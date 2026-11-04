# Pipeline toàn tập — từ ý tưởng tới production

Tài liệu này là **bản đồ toàn bộ luồng**, kèm mọi trick ở từng chặng. Đọc một lượt để thấy
hình dáng tổng thể, rồi quay lại từng mục khi làm.

Ký hiệu: 🔴 = bẫy có thể giết cả dự án · ⚡ = trick tăng tốc · 📏 = chỗ phải đo

---

## Toàn cảnh

```
┌─ 0. CÓ NÊN FINE-TUNE KHÔNG? ──────────────────────────────┐
│  prompt → structured output → RAG → MỚI tới fine-tune     │
└───────────────────────┬───────────────────────────────────┘
                        ↓
┌─ 1. DỮ LIỆU (80% kết quả nằm ở đây) ──────────────────────┐
│  thu thập → làm sạch → format → chia tập → 📏 validate    │
│  🔴 leak · 🔴 nhãn không nhất quán · 🔴 độ dài vượt max   │
└───────────────────────┬───────────────────────────────────┘
                        ↓
┌─ 2. CHỌN MODEL + PHƯƠNG PHÁP ─────────────────────────────┐
│  size theo VRAM → LoRA/QLoRA/full → SFT/DPO/GRPO          │
└───────────────────────┬───────────────────────────────────┘
                        ↓
┌─ 3. BASELINE TRƯỚC KHI TRAIN ─────────────────────────────┐
│  📏 đo model GỐC trên tập eval. Không có số này thì        │
│     không biết fine-tune có giúp gì                        │
└───────────────────────┬───────────────────────────────────┘
                        ↓
┌─ 4. TRAIN ────────────────────────────────────────────────┐
│  label mask → collator → LoRA → args → 📏 train+val loss  │
│  ⚡ dtype · ⚡ packing · ⚡ group_by_length · ⚡ batch      │
└───────────────────────┬───────────────────────────────────┘
                        ↓
┌─ 5. ĐÁNH GIÁ 3 TẦNG ──────────────────────────────────────┐
│  metric tự động → held-out paraphrase → LLM-judge         │
│  📏 multi-seed mean±std · 📏 phân loại lỗi                 │
└───────────────────────┬───────────────────────────────────┘
                        ↓
            ┌───────────┴───────────┐
            │  đủ tốt?              │
            │  chưa → 6. ABLATION   │──┐
            └───────────┬───────────┘  │ một biến mỗi lần
                        │ rồi          └──→ quay lại 1 hoặc 4
                        ↓
┌─ 7. DEPLOY ───────────────────────────────────────────────┐
│  merge → quantize → serve (vLLM/GGUF) → 📏 đo latency     │
│  🔴 Q2_K phá model                                         │
└───────────────────────┬───────────────────────────────────┘
                        ↓
┌─ 8. MONITOR ──────────────────────────────────────────────┐
│  trace (Langfuse) → phát hiện drift → thu data mới → về 1 │
└───────────────────────────────────────────────────────────┘
```

---

## Chặng 0 — Có nên fine-tune không?

Thử theo thứ tự **rẻ → đắt**. Đừng nhảy cóc:

| Thứ tự | Cách | Giải quyết được gì |
|---|---|---|
| 1 | Sửa system prompt | Giọng điệu, định dạng đơn giản, phạm vi |
| 2 | Few-shot trong prompt | Định dạng phức tạp hơn |
| 3 | Structured output / JSON schema | Bắt buộc đúng schema |
| 4 | **RAG** | Thiếu **kiến thức**, dữ liệu nội bộ, fact mới |
| 5 | **Fine-tune** | Thiếu **cách làm**: giọng, định dạng khó, task prompt không diễn tả nổi, nén model to → nhỏ |

🔴 **Lỗi phổ biến nhất:** fine-tune để nhồi kiến thức. Fine-tune dạy **cách làm**, RAG cấp
**thông tin**. Nhồi fact bằng fine-tune vừa kém vừa tăng hallucinate.

**Dấu hiệu fine-tune là đúng việc:** model gốc đạt gần 0% vì nó *không biết phải xuất ra cái gì*,
chứ không phải vì thiếu kiến thức. Track 01 là ví dụ sạch: Qwen3-0.6B **0/60** ở text→SQL —
nó biết SQL, nhưng không biết format mong đợi. Prompt engineering ở cỡ 0.6B không cứu được.

---

## Chặng 1 — Dữ liệu

### Luồng

```
nguồn thô ──→ lọc/dedupe ──→ chuẩn hoá nhãn ──→ format hội thoại
                                                      ↓
      validate ←── chia train/val/test ←── kiểm tra độ dài
```

### Trick và bẫy

**🔴 Leak giữa các tập.** Kiểm tra bằng hash, không bằng niềm tin:
```python
import hashlib
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
tr = {h(x) for x in train_texts}; te = {h(x) for x in test_texts}
assert not (tr & te), f"LEAK: {len(tr & te)} mau trung"
```

**🔴 Nhãn không nhất quán giết kết quả.** Bằng chứng từ `smartkids-finetune`:
v2 có **gấp 3 lần** dữ liệu của v1 nhưng điểm **thấp hơn** (86.8% vs 87.7%). Một pattern sai
nhất quán trong data (dòng "Số dư" thừa) → model học y nguyên cái sai.

**🔴 Độ dài vượt `max_length` → nhãn bị cắt → model học sinh câu cụt.** 📏 Luôn đo trước:
```python
lens = np.array([len(tok(x)["input_ids"]) for x in texts])
print(f"p50={np.percentile(lens,50):.0f} p95={np.percentile(lens,95):.0f} max={lens.max()}")
print(f"bi cat: {(lens>=MAX_LEN).sum()}/{len(lens)}")   # muc tieu < 1%
```

**⚡ Giữ cột `length`** trong dataset — cần cho `group_by_length=True` sau này.

**Tập held-out paraphrase.** Diễn đạt lại cùng câu hỏi bằng từ khác. Điểm tụt mạnh → model
học thuộc câu chữ. `smartkids-finetune` đã làm đúng (`heldout_paraphrase.jsonl`).

### Bao nhiêu mẫu

| Mục tiêu | Số mẫu |
|---|---|
| Đổi định dạng/giọng | 200–500 |
| Dạy task mới rõ ràng | 1.000–5.000 |
| Nhiều task | 5.000–20.000 |
| Đổi hành vi sâu | 20.000+ |

### Label masking — trái tim của SFT

```python
p = tok(prompt, add_special_tokens=False)["input_ids"]
a = tok(answer + tok.eos_token, add_special_tokens=False)["input_ids"]
input_ids = p + a
labels    = [-100]*len(p) + a      # <- phan prompt KHONG tinh loss
```

🔴 Không mask → model bị phạt vì "không đoán được câu hỏi của user", việc vô nghĩa.
**Loss vẫn giảm đẹp**, nên đây là lỗi im lặng: bạn tưởng train tốt mà chất lượng kém hơn.

Trong collator, pad `labels` bằng **`-100`**, không phải `pad_token_id`:
```python
ids.append(f["input_ids"] + [pad_id]*k)
labs.append(f["labels"]   + [-100] *k)      # <- khac nhau o day
att.append([1]*len(f["input_ids"]) + [0]*k)
```

TRL làm hộ qua `assistant_only_loss=True` / `completion_only_loss`, nhưng **tự viết một lần**
để hiểu, rồi mới dùng TRL.

---

## Chặng 2 — Chọn model + phương pháp

```
Model <= 1B va co >= 16GB?
├─ Co  → LoRA 16-bit (hoac full FT neu <= 0.6B)
└─ Khong → Model <= 14B?
           ├─ Co  → QLoRA 4-bit   <- mac dinh nen chon
           └─ Khong → thue GPU / chon model nho hon
```

| Params | QLoRA 4-bit | LoRA 16-bit |
|---|---|---|
| 3B | 3.5 GB | 8 GB |
| 7B | 5 GB | 19 GB |
| 14B | 8.5 GB | 33 GB |
| 27B | 22 GB | 64 GB |
| 70B | 41 GB | 164 GB |

**16GB:** QLoRA tới 14B thoải mái; LoRA 16-bit chỉ tới ~3B.

### Train phần nào

| Modality | Train | KHÔNG train |
|---|---|---|
| LLM text | cả 7 linear: `q,k,v,o,gate,up,down` | — |
| VLM / OCR | LoRA phần ngôn ngữ + projector | 🔴 **vision encoder** |
| Embedding | toàn bộ hoặc LoRA | — |
| ASR Whisper | full fine-tune | — |
| Diffusion | LoRA trên UNet/transformer | text encoder (cần ≥24GB) |

🔴 Chỉ target `q_proj,v_proj` là **không đủ** để bằng full fine-tune. Phải phủ hết 7 linear.

### SFT / DPO / GRPO

| Dữ liệu có | Dùng |
|---|---|
| (input → output đúng) | **SFT** — luôn làm trước |
| (prompt, chosen, rejected) | DPO |
| nhãn tốt/xấu đơn lẻ | KTO |
| hàm chấm tự động (math/code/test) | GRPO |
| có teacher to hơn | DistillationTrainer |

🔴 **lr của DPO/GRPO là `1e-6`, thấp hơn SFT (`2e-4`) khoảng 200 lần.** Dùng 2e-4 cho DPO là phá model.

---

## Chặng 3 — Baseline trước khi train

📏 Đo model **gốc** trên đúng tập eval, đúng thước đo, **trước khi** train. Không có số này thì
mọi con số sau đó vô nghĩa.

🔴 Đừng so với số in trên paper của người khác — khác prompt, khác số shot, khác harness.
Phải tự chạy baseline trong cùng điều kiện.

---

## Chặng 4 — Train

### Siêu tham số khởi đầu (LLM SFT)

```python
learning_rate      = 2e-4        # LoRA chiu lr cao hon full FT ~100x
num_train_epochs   = 1           # 1-3; >3 tren instruction data = overfit
r                  = 16          # thang 8/16/32/64/128
lora_alpha         = 32          # = r hoac r*2, giu alpha/r >= 1
lora_dropout       = 0           # len 0.1 neu overfit
target_modules     = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
batch hieu dung    = 16          # vd batch=8 x accum=2
warmup_steps       = 5-10% tong step
lr_scheduler_type  = "cosine"
weight_decay       = 0.01
```

### ⚡ Trick tăng tốc, xếp theo mức tác động

| # | Trick | Ghi chú |
|---|---|---|
| 1 | **Chọn đúng dtype** | Xem bẫy `is_bf16_supported()` bên dưới. Lỗi 1 dòng, tốn nhiều nhất |
| 2 | **`packing=True`** (TRL) | Nhồi nhiều mẫu vào 1 sequence. Mạnh nhất khi mẫu ngắn hơn `max_length` nhiều |
| 3 | **`group_by_length=True`** | Một flag, gom mẫu cùng độ dài → giảm padding. Gần như free |
| 4 | **Tăng batch tới sát trần VRAM** | Rẻ nhất |
| 5 | **`attn_implementation="sdpa"`** | 🔴 FlashAttention-2 **cần sm_80+**, không chạy trên T4 |
| 6 | **Unsloth** | Công bố ~2x nhanh + ít VRAM. 🔴 ghi đè torch → session riêng |
| 7 | **Cache model** thành Kaggle Model/Dataset | Khỏi tải lại mỗi session |
| 8 | 2 GPU qua `accelerate launch --multi_gpu` | Chỉ lợi khi model/data đủ lớn |

### ⚡ Trick tiết kiệm VRAM, theo thứ tự nên thử

1. `batch_size` ↓ + `gradient_accumulation_steps` ↑ (giữ batch hiệu dụng) — rẻ nhất
2. `gradient_checkpointing=True` — tiết kiệm nhiều, chậm 20-30%
3. QLoRA 4-bit — giảm ~4x
4. `max_length` ↓ — nhưng xem phân phối độ dài trước
5. `max_pixels` ↓ (VLM) — **cần điều khiển mạnh nhất cho VLM**
6. `optim="adamw_8bit"` — optimizer state giảm 4x
7. `r` ↓ — ảnh hưởng ít nhất, đừng sờ trước các cái trên

### 📏 Phải log train **và** val loss

```python
eval_strategy="steps", eval_steps=25, eval_dataset=val
```

Cách đọc đường loss:
- hai đường cùng giảm → học tốt, có thể train thêm
- train giảm / val đi ngang hoặc **tăng** → 🔴 overfit, dừng
- cả hai đi ngang ở mức cao → underfit: tăng lr, tăng `r`, hoặc thêm data

---

## Chặng 5 — Đánh giá 3 tầng

### Tầng 1: metric tự động của chính nhiệm vụ

| Nhiệm vụ | Thước đo |
|---|---|
| text→SQL | khớp chính xác sau chuẩn hoá; tốt hơn: **execution accuracy** (chạy query, so kết quả) |
| ảnh→JSON (OCR) | khớp từng field + F1 trên tập key. 🔴 **không** dùng BLEU |
| phân loại | accuracy, macro-F1 (lệch lớp thì dùng macro) |
| sinh văn bản | BLEU/chrF — yếu, phải kèm tầng 3 |
| ASR | WER, CER |
| embedding | nDCG@10, Recall@k, MRR |

### Tầng 2: held-out paraphrase
Phát hiện học thuộc câu chữ.

### Tầng 3: LLM-as-judge
Bắt được thứ luật chấm bỏ sót. `smartkids-finetune`: judge phát hiện nhiều model gốc
**khen "em làm đúng" khi học sinh làm sai** (5 × 3 = 16) — luật chấm hoàn toàn không thấy.

Làm judge cho đúng:
- **ẩn danh nhãn model** + **xáo thứ tự** mỗi câu (chống position bias)
- **≥2 judge độc lập**/câu, lấy trung bình
- rubric cụ thể thang 1–5, có định nghĩa từng mức
- chấm **nhiều chiều riêng biệt** (đúng / rõ ràng / giọng), không gộp 1 điểm
- judge phải là model **khác** model đang đánh giá

### 📏 Multi-seed — con số bắt buộc trước khi công bố

```python
accs = [train_and_eval(seed) for seed in [0,1,2]]
mu, sd = np.mean(accs), np.std(accs)
# Moi 'cai thien' nho hon ~2*sd la KHONG phan biet duoc voi nhieu
```

🔴 Phải đo σ của **accuracy**, không phải σ của **train loss**. Hai đại lượng khác nhau —
σ của loss không cho biết gì về việc chênh accuracy có ý nghĩa hay không.

### 📏 Phân loại lỗi
Một con số accuracy không nói model sai **kiểu** gì → không biết sửa gì. Phân nhóm lỗi
(không ra format / sai mệnh đề đầu / thiếu WHERE / sai hàm tổng hợp / sai chi tiết).

### 📏 Đo tỉ lệ bị cắt token
`smartkids-finetune` bắt đúng điểm này: Qwen3.8-27B gốc bị cắt **73.7%** câu trả lời.
Không đo thì sẽ kết luận sai hoàn toàn về model đó.

---

## Chặng 6 — Ablation

🔴 **Một biến mỗi lần.** Bảng: data size × LoRA rank × lr × epoch.
Ghi cả **thí nghiệm thất bại** — đó là thông tin, không phải rác.

| Thí nghiệm | Kỳ vọng thấy gì |
|---|---|
| data ×5 | Điểm tăng **không tuyến tính** |
| `lr=2e-2` | Loss **nổ NaN** |
| `lr=2e-6` | Loss gần như không giảm → underfit |
| bỏ label masking | Loss vẫn giảm đẹp **nhưng eval tệ hơn** |
| `epoch=3` | Train loss giảm, **val loss tăng** → overfit |
| chỉ target `q,v` | Kém hơn rõ so với phủ hết 7 linear |

---

## Chặng 7 — Deploy

```
LoRA adapter ──merge_and_unload()──→ model doc lap
                                          ↓
                            ┌─────────────┴─────────────┐
                            ↓                           ↓
                  vLLM / SGLang (GPU)          GGUF → Ollama (CPU)
```

Lượng hoá — số thật từ `smartkids-finetune`:

| Quant | Điểm |
|---|---|
| Q8_0 | 68.4% |
| Q4_K_M | 63.2% |
| **Q2_K** | **18.4%** 🔴 và 71.1% câu bị cắt |

🔴 **Q2_K phá model.** Q4_K_M là sàn thực dụng.

| Engine | Khi nào |
|---|---|
| `transformers.generate` | Thử nghiệm. Chậm nhất. 🔴 nhớ `padding_side="left"` khi generate theo batch |
| **vLLM** | Production nhiều request. PagedAttention + continuous batching |
| SGLang | Tương tự, mạnh ở structured output |
| llama.cpp / GGUF | CPU, Ollama |

---

## Chặng 8 — Monitor

- **Langfuse** trace inference ở production (không phải training)
- Phát hiện drift: phân phối input đổi → thu data mới → quay lại chặng 1
- Lưu `metrics.json` mỗi lần train

---

## Pipeline theo từng modality

### VLM / OCR
```
anh ──resize (max_pixels) ──→ messages[{type:image},{type:text}]
                                     ↓
              AutoProcessor ──→ SFTTrainer(max_length=None) + LoRA chi phan ngon ngu
```
🔴 `max_length=None` **bắt buộc** — truncate cắt image placeholder token → crash.
⚡ `max_pixels` là cần điều khiển VRAM số 1: token ≈ `pixels/(28×28)`.
🔴 Resize quá mạnh làm mất chữ nhỏ → OCR không đọc được. Thử 768 → nếu CER tệ thì 1024 + giảm batch.

### Embedding / Reranker
```
(anchor, positive) ──→ MultipleNegativesRankingLoss   <- manh nhat, mac dinh
(a, p, negative)   ──→ TripletLoss / CachedMNRL
(A, B, diem 0-1)   ──→ CoSENTLoss
reranker (q,d,0/1) ──→ BinaryCrossEntropyLoss
```
⚡ Batch **càng to càng tốt** với MNRL (negative lấy trong batch).

### ASR
🔴 Common Voice **đã bị gỡ khỏi HuggingFace (10/2025)** — mọi tutorial dùng
`mozilla-foundation/common_voice_*` đều gãy ở dòng `load_dataset`.

---

## 🔴 Bảng bẫy — tất cả đều gặp thật trong khi dựng repo này

| Bẫy | Triệu chứng | Fix |
|---|---|---|
| `warmup_ratio` bị bỏ ở transformers v5 | `TypeError: unexpected keyword argument 'warmup_ratio'` | `warmup_steps=N` |
| **`group_by_length` cũng bị bỏ ở v5** | `TypeError: unexpected keyword argument 'group_by_length'` | `ds.sort("length")` cho cùng hiệu ứng, chạy mọi bản |
| `SFTConfig` đổi tên | không có `max_seq_length` | `max_length` |
| torchao cũ trên Kaggle | `ImportError: incompatible version of torchao` khi `get_peft_model()` — **không hề nhắc LoRA** | `pip uninstall -y torchao` |
| Qwen3-VL cần torchvision | `Qwen3VLVideoProcessor requires the Torchvision library` | `pip install torchvision` |
| VLM truncate | crash giữa train | `max_length=None` |
| **`is_bf16_supported()` nói dối** | T4 cap (7,5) mà báo `True` → chạy bf16 **emulation**, không dùng tensor core | `get_device_capability(0)[0] >= 8` |
| **`pip install` trong subprocess không cô lập env** | cài unsloth/vllm → `ImportError: cannot import name '_chunk_or_narrow_cat' from 'torch._utils'` | Chia SAFE/WRECKERS, benchmark trước, cài phá sau |
| **2 GPU làm `Trainer` âm thầm dùng `nn.DataParallel`** | `SFTTrainer does not support nn.DataParallel`; batch thực tế nhân đôi | `os.environ["CUDA_VISIBLE_DEVICES"]="0"` **trước** `import torch` |
| Kaggle giới hạn session | `Maximum batch GPU session count of 2 reached` | Chỉ 2 batch GPU session đồng thời |
| FlashAttention-2 trên T4 | không cài được / không chạy | Cần sm_80+. Dùng `sdpa` |
| `kaggle kernels output` tải hết | tải 1.2GB model merge mỗi lần xem kết quả | `SAVE_MERGED=False` |
| `device_map="auto"` ≠ nhanh hơn | chia layer (model parallel), model nhỏ chỉ thêm overhead | `accelerate launch --multi_gpu` |

---

## Checklist trước khi công bố bất kỳ con số nào

**Tái lập**
- [ ] Ghim seed (`seed`, `data_seed`) **và báo cáo nó**
- [ ] `pip freeze > requirements.txt`
- [ ] Ghi GPU + dtype — 🔴 **bf16 và fp16 cho kết quả khác nhau**
- [ ] Ghi số GPU và có dùng DataParallel không
- [ ] Commit hash
- [ ] Config ở file riêng, không hardcode trong notebook

**Đo lường**
- [ ] Test không giao train — kiểm tra bằng **hash**
- [ ] **≥3 seed**, báo `mean ± std`
- [ ] Có baseline model gốc trong cùng bảng
- [ ] Có held-out paraphrase
- [ ] Thước đo khớp nhiệm vụ, không chỉ loss
- [ ] Đo tỉ lệ bị cắt token

**Báo cáo**
- [ ] Model card: data, license, giới hạn, bias
- [ ] Nói rõ làm được / **không** làm được gì
- [ ] Chi phí compute (GPU-giờ)
- [ ] Với harness: tên task, số shot, phiên bản harness, commit hash

> Khoảng cách giữa "tôi fine-tune được" và "tôi có kết quả đáng tin" chính là cái checklist này.


---

## Phòng vệ trước thay đổi API — viết code sống được qua các bản

Session dựng repo này gặp **hai** lỗi cùng loại: `warmup_ratio` rồi `group_by_length`, đều bị
transformers v5 bỏ. Bài học không phải "nhớ hai tên đó" mà là **đừng tin tutorial, hãy introspect**.

### Tra tên tham số còn tồn tại
```python
import inspect
from transformers import TrainingArguments
P = set(inspect.signature(TrainingArguments.__init__).parameters)
for k in ["group_by_length","warmup_ratio","warmup_steps","torch_compile"]:
    print(f"{k:22s}", "CO" if k in P else "KHONG CO")
```

### Helper tự lọc tham số không tồn tại
```python
TA_PARAMS = set(inspect.signature(TrainingArguments.__init__).parameters)

def ta(**kw):
    'Tao TrainingArguments, bo im lang tham so ban nay khong ho tro.'
    miss = [k for k in kw if k not in TA_PARAMS]
    if miss: print("[bo tham so khong ton tai]:", miss)
    return TrainingArguments(**{k: v for k, v in kw.items() if k in TA_PARAMS})
```
Code viết kiểu này **không chết** khi chạy trên bản transformers khác — nó chỉ bỏ cần điều
khiển không có và nói cho bạn biết.

### Validate notebook TRƯỚC khi push
Hai lỗi syntax trong notebook `02` chỉ bị phát hiện sau khi Kaggle chạy và ERROR — mất một
lượt GPU vô ích. Kiểm tra trước bằng [`tools/validate_nb.py`](../tools/validate_nb.py):

```bash
python tools/validate_nb.py notebooks/*.ipynb
```

Nó parse AST từng cell Python (bỏ qua cell `!`/`%` magic) và in ra đúng dòng lỗi.
**Thêm bước này vào quy trình trước mọi lần push** — một lượt Kaggle là ~10 phút GPU.

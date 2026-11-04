# Dữ liệu đo thật

Không phải ước lượng. Đây là output của các run có thật, kèm log.

## ⭐ `01_llm_sft_kaggle_t4_v4_CLEAN.json` — run SẠCH, dùng số này

Single-GPU (đã ghim `CUDA_VISIBLE_DEVICES=0`), fp16 đúng, có multi-seed.

| | Kết quả |
|---|---|
| Qwen3-0.6B gốc | **0 / 60 (0.0%)** |
| + LoRA r=16, seed 0 | **48 / 60 (80.0%)** |
| seed 1 | 48 / 60 (80.0%) |
| seed 2 | 47 / 60 (78.3%) |
| **accuracy (3 seed)** | **79.44% ± 0.79%** |
| **→ ngưỡng nhiễu (2σ)** | **1.57%** |
| Thời gian train | **308 s** |
| dtype | `float16` |
| train loss | 0.569 → 0.054 |
| val loss | 0.173 → 0.081 (chưa tăng lại → không overfit) |
| latency | 1.89 s/query |

### Con số quan trọng nhất: **σ = 0.79%**

> Mọi "cải thiện" nhỏ hơn **~1.57%** là **không phân biệt được với nhiễu.**

Đây là con số cho phép bạn nói "cấu hình A tốt hơn B". Không có nó thì mọi so sánh đều là
phỏng đoán. Và phải là σ của **accuracy**, không phải σ của train loss.

## So sánh các run — và vì sao chỉ run cuối đáng tin

| Run | GPU | dtype | Thời gian | Accuracy | Đáng tin? |
|---|---|---|---|---|---|
| local RTX 5060 Ti | 1× (sm_120) | bf16 **native** | 108 s | 73.3% (44/60) | ⚠️ tập eval khác |
| Kaggle v2 | **2×T4 `nn.DataParallel`** | bf16 **emulation** | 623 s | 76.67% (46/60) | ❌ cấu hình khác với khai báo |
| **Kaggle v4** | 1×T4 (đã ghim) | **fp16** | **308 s** | **79.44% ± 0.79%** | ✅ |

**623 s → 308 s = nhanh 2.02x.** Nhưng ⚠️ **hai biến đổi cùng lúc** (bf16-emulation → fp16,
và bỏ `nn.DataParallel` 2 GPU → 1 GPU), nên **không được quy công cho riêng cái nào**.
`notebooks/02_speed_kaggle.ipynb` cô lập riêng biến dtype để trả lời câu đó.

Điều đáng chú ý: **1 GPU với dtype đúng nhanh gấp đôi 2 GPU với dtype sai.** Thêm GPU không
cứu được một flag sai.

### Bẫy phát hiện từ các run này

**1. `torch.cuda.is_bf16_supported()` nói dối.** Log run sạch in thẳng ra:
```
May nay co 2 GPU:
  GPU 0: Tesla T4 ...
  GPU 1: Tesla T4 ...
is_bf16_supported() bao : True   <- KHONG tin duoc
compute cap[0] >= 8     : False  <- day moi dung
-> dung fp16
```
Hàm đó tính cả bf16 *emulation* (không chạm tensor core). Cách đúng:
`torch.cuda.get_device_capability(0)[0] >= 8`.

**2. 2 GPU làm `Trainer` âm thầm dùng `nn.DataParallel`.** `per_device_train_batch_size=8`
thành 16 mẫu/step, batch hiệu dụng 32 thay vì 16. Phát hiện được **chỉ vì** TRL `SFTTrainer`
từ chối thẳng:
```
ValueError: SFTTrainer does not support `nn.DataParallel`, which `Trainer` uses when
several GPUs are visible to a single process.
```
`Trainer` thuần thì chạy im lặng — sẽ không bao giờ biết. **Một lỗi ồn ào tốt hơn một mặc định
im lặng.** Fix: `os.environ["CUDA_VISIBLE_DEVICES"]="0"` **trước** `import torch`.

**3. `SAVE_MERGED=False` giảm output 1.2GB → 50MB.** `kaggle kernels output` tải **toàn bộ**
working dir, không chọn lọc được.

## `framework_probe_kaggle_t4.json` — 24 framework trên Kaggle T4

19/24 dùng được. Tổng thời gian cài 16 phút.

| Nhóm | Dùng được | Thất bại |
|---|---|---|
| fine-tune | transformers 5.22.1 · peft 0.21.2 · trl 1.17.1 · accelerate 1.15.0 · bitsandbytes 0.50.2 · unsloth 2026.11.1 · llamafactory 0.9.5 | torchtune *(import)* · axolotl *(build `zstandard` fail)* |
| inference | vllm 0.33.0 *(5.7 phút cài)* · llama-cpp-python 0.3.36 · optimum | sglang *(`outlines_core`)* |
| eval | lm-eval 0.4.13 · lighteval 0.13.0 · mteb 2.26.33 · deepeval 4.3.3 · evaluate 0.4.6 · sentence-transformers 5.7.0 | vlmeval *(no matching distribution)* · ragas *(import)* |
| tracking | wandb 0.28.1 · mlflow 3.18.0 · langfuse 5.1.0 | — |

Lưu ý: Kaggle resolve `sentence-transformers` về **5.7.0**, local là **6.2.0**. Phiên bản khác
nhau giữa môi trường là chuyện thường — thêm một lý do phải `pip freeze` khi báo cáo.

### Bài học: `pip install` trong subprocess **không** cô lập môi trường
Run v1 của notebook `00` ERROR với:
```
ImportError: cannot import name '_chunk_or_narrow_cat' from 'torch._utils'
```
Vì `subprocess` chỉ cô lập *import*, còn `pip install` vẫn ghi vào **cùng** site-packages.
Cài `unsloth`/`vllm` đã thay torch và làm gãy `torch.distributed`.

**Cách sửa đã áp dụng:** chia framework làm nhóm an toàn / nhóm phá môi trường, và **đảo thứ tự** —
benchmark chạy trước, cài nhóm phá môi trường để cuối cùng. Muốn benchmark sâu một framework
nhóm 2 thì cho nó một session Kaggle riêng.


## Giới hạn Kaggle đã gặp
- **2 batch GPU session đồng thời** là tối đa (`Maximum batch GPU session count of 2 reached`)
- 30h GPU/tuần, 12h/session
- `kaggle kernels output` tải **toàn bộ** working dir, không chọn lọc được

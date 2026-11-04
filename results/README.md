# Dữ liệu đo thật

Không phải ước lượng. Đây là output của các run có thật, kèm log.

## `01_llm_sft_kaggle_t4_v2.json` — LLM SFT text→SQL

| | RTX 5060 Ti (local) | Kaggle T4 |
|---|---|---|
| Qwen3-0.6B gốc | 0/60 (0%) | 0/60 (0%) |
| + LoRA r=16 | 44/60 (**73.3%**) | 46/60 (**76.67%**) |
| Thời gian train | **108 s** | **623 s** (~5.8x chậm hơn) |
| dtype thực tế dùng | bf16 (native, sm_120) | bf16 **emulation** ⚠️ (xem dưới) |
| train loss | → 0.059 | 0.569 → 0.059 |
| val loss | — | 0.170 → 0.090 (chưa tăng lại) |
| VRAM đỉnh | — | 12.75 GB |
| latency / query | — | 1.93 s |
| mẫu bị cắt token | — | 0/4000 (0.00%) |

⚠️ **Hai cột không so trực tiếp được**: khác tập eval (local dùng `range(4000,4060)`, Kaggle dùng
`range(4200,4260)`) và khác dtype path. Chênh 2 mẫu (3.3%) hoàn toàn có thể là nhiễu seed.
Đây chính là lý do notebook `01` có mục **10b multi-seed** đo `σ` của accuracy — không có `σ`
thì không được phép nói cấu hình nào tốt hơn.

### Bẫy phát hiện được từ run này
`torch.cuda.is_bf16_supported()` **trả về `True` trên Tesla T4 (compute cap 7.5)**, vì nó tính cả
bf16 *emulation*. Auto-detect naive sẽ chọn bf16 trên card mà tensor core chỉ hỗ trợ fp16 →
chạy được nhưng chậm. Cách đúng: `torch.cuda.get_device_capability(0)[0] >= 8`.

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


## ⚠️ Cảnh báo về con số 623s của notebook 01

Run đó chạy trên Kaggle với **2×T4 nhìn thấy được**, và `transformers.Trainer` **âm thầm bọc
model vào `nn.DataParallel`** trong trường hợp đó. Nên:

- `per_device_train_batch_size=8` thực tế thành **16 mẫu/step** (8 × 2 GPU)
- batch hiệu dụng thành **32**, không phải 16 như ý định
- `nn.DataParallel` vốn kém hiệu quả (bản thân PyTorch cũng khuyên dùng DDP)

**Con số 623s và 76.67% vẫn là kết quả thật, nhưng ở một cấu hình khác với cấu hình tôi khai báo.**
Đã fix bằng cách ghim `CUDA_VISIBLE_DEVICES=0` **trước** `import torch` trong cả 3 notebook,
để mọi phép đo là single-GPU, công bằng và tái lập được.

Phát hiện ra lỗi này là vì TRL `SFTTrainer` **từ chối thẳng** thay vì chạy im lặng:
```
ValueError: SFTTrainer does not support `nn.DataParallel`, which `Trainer` uses when
several GPUs are visible to a single process. Launch the script with `accelerate launch`
or `torchrun`, or make a single GPU visible with `CUDA_VISIBLE_DEVICES`.
```
Nếu chỉ dùng `Trainer` thuần thì sẽ không bao giờ biết — đó là lý do một lỗi *ồn ào*
tốt hơn một mặc định *im lặng*.

## Giới hạn Kaggle đã gặp
- **2 batch GPU session đồng thời** là tối đa (`Maximum batch GPU session count of 2 reached`)
- 30h GPU/tuần, 12h/session
- `kaggle kernels output` tải **toàn bộ** working dir, không chọn lọc được

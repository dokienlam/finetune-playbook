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

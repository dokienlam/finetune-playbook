# Notebooks (Kaggle-ready)

Mỗi notebook chạy được **cả Kaggle T4 (fp16) và GPU Blackwell (bf16)** — tự nhận diện qua
`torch.cuda.is_bf16_supported()`.

| Notebook | Nội dung | Trạng thái |
|---|---|---|
| `00_frameworks.ipynb` | Probe 24 framework (fine-tune / inference / eval / tracking) + benchmark HF vs TRL + transformers vs vLLM + đo phương sai multi-seed | đang chạy trên Kaggle |
| `01_llm_sft.ipynb` | Full pipeline: data → label masking → train → val loss → 3 biểu đồ → phân tích lỗi → inference → merge | đang chạy trên Kaggle |

## Chạy trên Kaggle

**Cách 1 — qua API (tự động):**
```bash
cd notebooks
cp kernel-metadata-01.json kernel-metadata.json   # doi id thanh user cua ban
kaggle kernels push -p .
kaggle kernels status <user>/<slug>
kaggle kernels output <user>/<slug> -p ./log
```

**Cách 2 — thủ công:** New Notebook → File → Import Notebook → chọn `.ipynb`.
Rồi **Settings → Accelerator = GPU T4 x2** và **Internet = On**.

## Bẫy bắt buộc biết

1. **T4 không có bf16** (sm_75) → phải `fp16`. ⚠️ Nhưng `torch.cuda.is_bf16_supported()`
   **trả về `True` trên T4** (nó tính cả emulation), nên auto-detect naive sẽ chọn sai.
   Notebook dùng `get_device_capability(0)[0] >= 8`. Run đầu đã rơi vào bẫy này — xem log.
2. **Kaggle cài sẵn `torchao 0.10.0`** nhưng `peft 0.21.2` cần `>=0.16.0` → `get_peft_model()`
   raise `ImportError` với thông báo không hề nhắc LoRA. Fix: `pip uninstall -y torchao`.
   Đây là lỗi thật đã gặp khi chạy notebook `01` lần đầu.
3. **Internet phải On**, không thì `load_dataset` fail.
4. Chỉ `/kaggle/working/` được giữ lại sau session.

## Phần "vừa chạy vừa học"

Mỗi notebook có mục **Thí nghiệm** ở cuối: một bảng các thay đổi kèm **kết quả kỳ vọng**.
Sửa, chạy lại, so với cột kỳ vọng. Ra khác nghĩa là bạn học được điều mới (hoặc có bug).

Hai thí nghiệm giá trị nhất ở `01`:
- Đặt `learning_rate=2e-2` → loss nổ NaN. Thấy tận mắt lr quá cao.
- Bỏ label masking (`labels = (p+a)[:MAX_LEN]`) → **loss vẫn giảm đẹp nhưng điểm eval tệ hơn**.
  Đây chính là lý do phải mask, và là lỗi im lặng phổ biến nhất khi fine-tune.

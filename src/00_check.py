"""Bai 0: kiem tra moi truong. Chay: python 00_check.py"""
import torch, transformers, peft, datasets

print("torch        :", torch.__version__)
print("transformers :", transformers.__version__)
print("peft         :", peft.__version__)
print("datasets     :", datasets.__version__)
print("cuda san sang:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU          :", torch.cuda.get_device_name(0))
    print("compute cap  :", torch.cuda.get_device_capability(0), "(5060 Ti = (12,0) Blackwell)")
    print("VRAM         : %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))
    print("bf16 ho tro  :", torch.cuda.is_bf16_supported())
    # thu mot phep nhan matrix tren GPU de chac chan kernel chay duoc
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    import time; t = time.time()
    for _ in range(20):
        c = a @ b
    torch.cuda.synchronize()
    tflops = 20 * 2 * 4096**3 / (time.time() - t) / 1e12
    print("matmul bf16  : %.1f TFLOPS  -> kernel OK" % tflops)
else:
    print("!! Khong thay GPU")

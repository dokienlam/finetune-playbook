"""
Bai 3: So sanh model GOC vs model DA FINE-TUNE tren du lieu CHUA tung thay.
Chay: python 03_compare.py
"""
import re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset

MODEL, ADAPTER, N = "Qwen/Qwen3-0.6B", "out/qwen3-0.6b-sql-lora", 60
SYSTEM = ("Ban la bo dich text-to-SQL. Dua vao schema va cau hoi, "
          "chi tra ve DUY NHAT cau truy van SQLite, khong giai thich.")

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"          # generate theo batch phai pad ben trai

# Lay mau 4000..4060 -> KHONG nam trong tap train (train dung 0..4000 cung seed 42)
ds = load_dataset("b-mc2/sql-create-context", split="train").shuffle(seed=42)
ds = ds.select(range(4000, 4000 + N))

def prompt_of(ex):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user",
             "content": f"Schema:\n{ex['context']}\n\nCau hoi: {ex['question']}"}]
    return tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)

prompts = [prompt_of(e) for e in ds]
golds   = [e["answer"] for e in ds]

def norm(s):  # chuan hoa de so sanh cong bang
    return re.sub(r"\s+", " ", s.strip().rstrip(";")).lower()

@torch.no_grad()
def run(model, label):
    model.eval()
    outs = []
    for i in range(0, len(prompts), 12):
        b = tok(prompts[i:i+12], return_tensors="pt", padding=True,
                add_special_tokens=False).to("cuda")
        g = model.generate(**b, max_new_tokens=96, do_sample=False,
                           pad_token_id=tok.pad_token_id)
        for j in range(g.shape[0]):
            outs.append(tok.decode(g[j][b["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip())
    hit = sum(norm(o) == norm(gd) for o, gd in zip(outs, golds))
    print(f"{label:22s} khop chinh xac: {hit}/{len(golds)}  ({100*hit/len(golds):.1f}%)")
    return outs

base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
out_base = run(base, "Qwen3-0.6B GOC")

ft = PeftModel.from_pretrained(base, ADAPTER)   # dap LoRA len chinh model do
out_ft = run(ft, "Qwen3-0.6B + LoRA")

print("\n" + "=" * 78)
print("VAI VI DU CU THE")
print("=" * 78)
for k in range(3):
    print(f"\n[{k+1}] Cau hoi : {ds[k]['question']}")
    print(f"    Schema  : {ds[k]['context'][:90]}")
    print(f"    DAP AN  : {golds[k]}")
    print(f"    GOC     : {out_base[k][:200]!r}")
    print(f"    LoRA    : {out_ft[k][:200]!r}")

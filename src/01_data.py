"""
Bai 1: Du lieu + cach bien 1 mau thanh (input_ids, labels).
Nhiem vu hoc: text-to-SQL. Cho schema + cau hoi -> model sinh cau SQL.
Chay: python 01_data.py
"""
from transformers import AutoTokenizer
from datasets import load_dataset

MODEL = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(MODEL)

ds = load_dataset("b-mc2/sql-create-context", split="train")
print("So mau:", len(ds))
print("Cac cot:", ds.column_names)
ex = ds[0]
print("\n--- 1 mau tho ---")
for k, v in ex.items():
    print(f"{k}: {v}")

SYSTEM = ("Ban la bo dich text-to-SQL. Dua vao schema va cau hoi, "
          "chi tra ve DUY NHAT cau truy van SQLite, khong giai thich.")

def build(ex):
    """Tra ve (prompt_text, answer_text)."""
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Schema:\n{ex['context']}\n\nCau hoi: {ex['question']}"},
    ]
    prompt = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return prompt, ex["answer"]

prompt, answer = build(ex)
print("\n--- PROMPT (phan model KHONG hoc) ---")
print(repr(prompt))
print("\n--- ANSWER (phan model PHAI hoc) ---")
print(repr(answer))

# Day la trai tim cua SFT: ghep prompt + answer, nhung labels cua phan prompt = -100
p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
a_ids = tok(answer + tok.eos_token, add_special_tokens=False)["input_ids"]
input_ids = p_ids + a_ids
labels = [-100] * len(p_ids) + a_ids

print(f"\nlen(prompt)={len(p_ids)} tokens, len(answer)={len(a_ids)} tokens")
print("labels 10 dau  :", labels[:10],  "<- -100 = bo qua, khong tinh loss")
print("labels 10 cuoi :", labels[-10:], "<- day moi la phan tinh loss")
print("\n=> Model chi bi phat khi sinh sai cau SQL, khong bi phat vi 'khong doan duoc cau hoi'.")

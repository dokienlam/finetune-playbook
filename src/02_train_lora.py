"""
Bai 2: LoRA SFT tren Qwen3-0.6B, nhiem vu text-to-SQL.
Chay: python 02_train_lora.py
"""
import torch
from dataclasses import dataclass
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

MODEL   = "Qwen/Qwen3-0.6B"
OUT     = "out/qwen3-0.6b-sql-lora"
N_TRAIN = 4000      # tang len neu muon ket qua tot hon (va cho lau hon)
MAX_LEN = 512

SYSTEM = ("Ban la bo dich text-to-SQL. Dua vao schema va cau hoi, "
          "chi tra ve DUY NHAT cau truy van SQLite, khong giai thich.")

tok = AutoTokenizer.from_pretrained(MODEL)

def to_features(ex):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user",
             "content": f"Schema:\n{ex['context']}\n\nCau hoi: {ex['question']}"}]
    prompt = tok.apply_chat_template(msgs, tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    a = tok(ex["answer"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (p + a)[:MAX_LEN]
    lab = ([-100] * len(p) + a)[:MAX_LEN]
    return {"input_ids": ids, "labels": lab}

ds = load_dataset("b-mc2/sql-create-context", split="train")
ds = ds.shuffle(seed=42).select(range(N_TRAIN))
ds = ds.map(to_features, remove_columns=ds.column_names, desc="tokenize")

@dataclass
class Collator:
    pad_id: int
    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        ids, labs, att = [], [], []
        for f in feats:
            k = n - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_id] * k)
            labs.append(f["labels"]    + [-100]        * k)
            att.append([1] * len(f["input_ids"]) + [0] * k)
        t = torch.tensor
        return {"input_ids": t(ids), "labels": t(labs), "attention_mask": t(att)}

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
model.config.use_cache = False

lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir=OUT,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,     # batch hieu dung = 16
    num_train_epochs=1,
    learning_rate=2e-4,                # LoRA chiu lr cao hon full finetune nhieu
    lr_scheduler_type="cosine",
    warmup_steps=8,                    # v5 bo warmup_ratio, dung warmup_steps
    logging_steps=10,
    bf16=True,
    save_strategy="no",
    report_to=[],
)

trainer = Trainer(model=model, args=args, train_dataset=ds,
                  data_collator=Collator(tok.pad_token_id))
trainer.train()

model.save_pretrained(OUT)
tok.save_pretrained(OUT)
print(f"\nDa luu LoRA adapter vao: {OUT}")
print("Kich thuoc adapter:")

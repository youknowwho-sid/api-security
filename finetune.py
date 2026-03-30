import os
import gc
import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Kaggle paths ─────────────────────────────────────────────
KAGGLE_INPUT   = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")

# ── Env flags (must be set before torch import) ───────────────
os.environ["CUDA_VISIBLE_DEVICES"]        = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]     = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"]      = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import torch
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, SFTConfig


# ╔══════════════════════════════════════════════════════════════╗
# ║                      CONFIGURATION                          ║
# ╚══════════════════════════════════════════════════════════════╝

@dataclass
class Config:
    # ── Model ────────────────────────────────────────────────
    model_name: str = "codellama/CodeLlama-7b-instruct-hf"

    # ── Data ─────────────────────────────────────────────────
    # Update <dataset-slug> to match your Kaggle dataset URL
    dataset_json: str = "/kaggle/input/datasets/harshar27/10kapi/api_vulnerability_dataset_10k.json"
    train_split:  float = 0.85   # 85 % train, 15 % val
    text_field:   str = "text"   # name of the formatted column we create below

    # ── Sequence ─────────────────────────────────────────────
    max_seq_len: int = 512       # ↑ from 256; covers most API snippets

    # ── Quantisation (QLoRA) ─────────────────────────────────
    load_in_4bit:   bool = True
    bnb_quant_type: str  = "nf4"

    # ── LoRA ─────────────────────────────────────────────────
    lora_r:       int   = 16     # ↑ from 8; better capacity for vuln patterns
    lora_alpha:   int   = 32     # rule-of-thumb: 2 × r
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # ── Training ─────────────────────────────────────────────
    epochs:             int   = 1        # ↓ from 3; model converges fast
    batch_size:         int   = 1        # T4 constraint
    grad_accumulation:  int   = 16       # effective batch = 16
    learning_rate:      float = 1e-4     # ↓ from 2e-4; more stable
    warmup_ratio:       float = 0.05
    weight_decay:       float = 0.01
    lr_scheduler:       str   = "cosine"
    max_grad_norm:      float = 0.3

    # ── Precision ─────────────────────────────────────────────
    # bf16 is NOT supported on T4 (it's fp16 only)
    # We use fp16 here; if you're on A100/H100 flip these
    fp16: bool = True
    bf16: bool = False

    # ── Eval / Save ───────────────────────────────────────────
    eval_steps:               int = 100   # ↑ frequency (was 200)
    save_steps:               int = 100
    logging_steps:            int = 10
    early_stopping_patience:  int = 3

    # ── Output ────────────────────────────────────────────────
    output_dir: str = str(KAGGLE_WORKING / "qlora_vuln_model")
    seed: int = 42


# ╔══════════════════════════════════════════════════════════════╗
# ║                     HELPER UTILITIES                        ║
# ╚══════════════════════════════════════════════════════════════╝

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def check_gpu():
    print("=" * 55)
    print("  GPU Check")
    print("=" * 55)
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU found. Enable GPU in Kaggle: Settings → Accelerator → GPU T4 x1")

    name = torch.cuda.get_device_name(0)
    cap  = torch.cuda.get_device_capability(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    bf16 = torch.cuda.is_bf16_supported()

    print(f"  GPU   : {name}")
    print(f"  VRAM  : {vram:.1f} GB")
    print(f"  Cap   : {cap}")
    print(f"  bf16  : {'✓' if bf16 else '✗  (use fp16)'}")
    print("=" * 55)

    if not bf16 and not "T4" in name and not "P100" in name:
        print("  ⚠  Unknown GPU — check fp16/bf16 support manually")

    return bf16


def analyze_dataset(data, cfg: Config):
    """Print token-length stats to catch truncation issues early."""
    print("\n[Dataset Stats]")
    texts = data["train"][cfg.text_field]
    word_counts = [len(t.split()) for t in texts]
    p50  = int(np.percentile(word_counts, 50))
    p95  = int(np.percentile(word_counts, 95))
    p99  = int(np.percentile(word_counts, 99))
    maxi = max(word_counts)
    print(f"  Train samples : {len(texts)}")
    print(f"  Val   samples : {len(data['validation'])}")
    print(f"  Words  p50/p95/p99/max : {p50}/{p95}/{p99}/{maxi}")
    if p95 > cfg.max_seq_len:
        print(f"  ⚠  p95 words ({p95}) > max_seq_len ({cfg.max_seq_len}) "
              f"— consider increasing max_seq_len")
    else:
        print(f"  ✓  max_seq_len={cfg.max_seq_len} covers p95")


# ╔══════════════════════════════════════════════════════════════╗
# ║                      DATA LOADING                           ║
# ╚══════════════════════════════════════════════════════════════╝

# Dataset columns (confirmed from inspection):
#   id, code, label (HTTP method), language, framework,
#   resource, endpoint_path, flaws (list), cwe (list),
#   severity, vulnerability_description, secure_version,
#   source_dataset

SYSTEM_PROMPT = (
    "You are a security-focused code reviewer specializing in API vulnerability "
    "detection and remediation. Analyze the provided code, identify security flaws, "
    "explain the vulnerabilities, and provide a secure version."
)

def format_row(row) -> dict:
    """
    Build a CodeLlama [INST] prompt from the real dataset schema.

    Instruction  →  code + metadata context
    Response     →  structured vuln report + secure_version
    """
    # ── Normalise list fields (stored as Python lists in JSON) ──
    flaws = row["flaws"]
    if isinstance(flaws, list):
        flaws = ", ".join(flaws)

    cwes = row["cwe"]
    if isinstance(cwes, list):
        cwes = ", ".join(cwes)

    # ── Build instruction ────────────────────────────────────────
    instruction = (
        f"Analyze the following {row['language']} ({row['framework']}) API endpoint "
        f"for security vulnerabilities.\n\n"
        f"HTTP Method : {row['label']}\n"
        f"Endpoint    : {row['endpoint_path']}\n\n"
        f"```{row['language'].lower()}\n{row['code']}\n```"
    )

    # ── Build response ───────────────────────────────────────────
    response = (
        f"## Vulnerability Analysis\n\n"
        f"**Severity**       : {row['severity'].upper()}\n"
        f"**Flaw(s)**        : {flaws}\n"
        f"**CWE**            : {cwes}\n\n"
        f"**Description**\n{row['vulnerability_description']}\n\n"
        f"## Secure Version\n\n"
        f"```{row['language'].lower()}\n{row['secure_version']}\n```"
    )

    # ── CodeLlama instruct format ────────────────────────────────
    text = (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"{instruction} [/INST]\n"
        f"{response} </s>"
    )

    return {"text": text}


def load_and_split(cfg: Config) -> DatasetDict:
    print("\n[1/6] Loading & formatting dataset...")
    from datasets import Dataset

    with open(cfg.dataset_json) as f:
        records = json.load(f)

    raw = Dataset.from_list(records)
    print(f"  Raw columns : {raw.column_names}")
    print(f"  Total rows  : {len(raw)}")

    # Format every row into a single 'text' field
    formatted = raw.map(
        format_row,
        remove_columns=raw.column_names,  # drop originals, keep only 'text'
        desc="Formatting prompts",
    )

    # Preview one sample
    print("\n[Sample prompt — first 400 chars]")
    print("-" * 60)
    print(formatted[0]["text"][:400])
    print("-" * 60)

    # Deterministic train / val split
    split = formatted.train_test_split(
        test_size=round(1 - cfg.train_split, 2),
        seed=cfg.seed,
    )
    return DatasetDict({"train": split["train"], "validation": split["test"]})


# ╔══════════════════════════════════════════════════════════════╗
# ║                   MODEL + TOKENIZER                         ║
# ╚══════════════════════════════════════════════════════════════╝

def load_tokenizer(cfg: Config):
    print(f"\n[2/6] Loading tokenizer: {cfg.model_name}")
    tok = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        use_fast=False,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_model_qlora(cfg: Config, bf16_supported: bool):
    print("\n[3/6] Loading 4-bit QLoRA model...")
    cleanup()

    compute_dtype = torch.bfloat16 if bf16_supported else torch.float16

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=cfg.bnb_quant_type,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_cfg,
        device_map={"": 0},
        max_memory={0: "14GiB", "cpu": "30GiB"},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        attn_implementation="eager",
    )

    model.config.use_cache        = False
    model.config.pretraining_tp   = 1
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


# ╔══════════════════════════════════════════════════════════════╗
# ║                      SFT CONFIG                             ║
# ╚══════════════════════════════════════════════════════════════╝

def build_sft_config(cfg: Config, bf16_supported: bool) -> SFTConfig:
    print("\n[4/6] Building SFT config...")
    os.makedirs(cfg.output_dir, exist_ok=True)

    return SFTConfig(
        output_dir=cfg.output_dir,
        dataset_text_field=cfg.text_field,
        max_length=cfg.max_seq_len,
        packing=False,

        # ── Epochs / Batching ────────────────────────────────
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accumulation,

        # ── Optimiser ────────────────────────────────────────
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler,
        max_grad_norm=cfg.max_grad_norm,
        optim="paged_adamw_8bit",   # saves ~1.5 GB vs adamw_torch

        # ── Precision ────────────────────────────────────────
        fp16=cfg.fp16 and not bf16_supported,
        bf16=cfg.bf16 and bf16_supported,

        # ── Eval / Checkpointing ─────────────────────────────
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # ── Logging ──────────────────────────────────────────
        logging_steps=cfg.logging_steps,
        report_to="none",            # set "wandb" if you use W&B on Kaggle

        # ── Memory ───────────────────────────────────────────
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,    # Kaggle multiprocessing is unreliable
    )


# ╔══════════════════════════════════════════════════════════════╗
# ║                        TRAINING                             ║
# ╚══════════════════════════════════════════════════════════════╝

def train(cfg: Config):
    cleanup()
    bf16_ok = check_gpu()

    dataset  = load_and_split(cfg)
    analyze_dataset(dataset, cfg)

    tokenizer = load_tokenizer(cfg)
    model     = load_model_qlora(cfg, bf16_ok)
    sft_cfg   = build_sft_config(cfg, bf16_ok)

    print("\n[5/6] Initialising SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg.early_stopping_patience
            )
        ],
    )

    print("\n[6/6] Training...")
    trainer.train()

    # ── Save final adapter ────────────────────────────────────
    final = os.path.join(cfg.output_dir, "final_adapter")
    trainer.save_model(final)
    tokenizer.save_pretrained(final)
    print(f"\n✓ Training complete.  Adapter saved → {final}")
    print("  (Download from Kaggle Output tab or push to Hub below)")

    return trainer


# ╔══════════════════════════════════════════════════════════════╗
# ║              OPTIONAL — PUSH TO HUGGING FACE HUB            ║
# ╚══════════════════════════════════════════════════════════════╝

def push_to_hub(model, tokenizer, repo_id: str, hf_token: str):
    """
    Call after train() if you want to persist the adapter.
    repo_id example: "your-username/codellama-api-vuln-lora"
    Store your HF token as a Kaggle Secret named HF_TOKEN.
    """
    print(f"\nPushing adapter to Hub: {repo_id}")
    model.push_to_hub(repo_id, token=hf_token, private=True)
    tokenizer.push_to_hub(repo_id, token=hf_token, private=True)
    print("✓ Push complete.")


# ╔══════════════════════════════════════════════════════════════╗
# ║                    INFERENCE HELPER                         ║
# ╚══════════════════════════════════════════════════════════════╝

def run_inference(model, tokenizer, prompt: str, max_new_tokens: int = 128):
    """Quick sanity-check after training."""
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║                         MAIN                                ║
# ╚══════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    set_seed(42)
    cfg = Config()
    trainer = train(cfg)

    # ── Quick inference test ──────────────────────────────────
    # Uncomment to test after training:
    #
    # tok   = load_tokenizer(cfg)
    # model = trainer.model
    # test_prompt = (
    #     "<s>[INST] <<SYS>>\n"
    #     + SYSTEM_PROMPT + "\n<</SYS>>\n\n"
    #     "Analyze the following PHP (Laravel) API endpoint for security vulnerabilities.\n\n"
    #     "HTTP Method : GET\n"
    #     "Endpoint    : /api/users\n\n"
    #     "```php\n"
    #     "Route::get('/api/users', function(Request $r) {\n"
    #     "    $id = $r->input('id');\n"
    #     "    return DB::select(\"SELECT * FROM users WHERE id = $id\");\n"
    #     "});\n"
    #     "``` [/INST]\n"
    # )
    # print("\n[Inference Test]")
    # print(run_inference(model, tok, test_prompt))

    # ── Push to Hub ───────────────────────────────────────────
    # import os
    # hf_token = os.environ.get("HF_TOKEN")   # set in Kaggle Secrets
    # if hf_token:
    #     push_to_hub(trainer.model, tok, "your-username/codellama-api-vuln", hf_token)
# 🔒 API Security Scanner

An end-to-end API vulnerability detection system that combines
a fine-tuned Code Llama 7B model with a rule-based engine to
scan any GitHub repository for security vulnerabilities.

---

## 📁 Project Structure

```
api_security/
│
├── app.py                          # Streamlit UI (main entry point)
│
├── endpoint_extractor.py           # Extracts API routes from repos
├── inference.py                    # Runs fine-tuned model on endpoints
├── rules_checker.py                # Validates endpoints against rules
├── report_generator.py             # Generates HTML vulnerability report
├── pipeline.py                     # CLI alternative to Streamlit
│
├── data_prep_universal.py          # Prepares dataset for any model
├── finetune.py                     # Fine-tuning script (local)
├── API_Security_Finetune.ipynb     # Fine-tuning notebook (Google Colab)
│
├── fix_dataset.py                  # Fixes class imbalance in dataset
├── merge_diversevul.py             # Merges DiverseVul into dataset
├── generate_synthetic.py           # Generates synthetic samples via Claude API
│
├── api_vulnerability_dataset_balanced.json   # Balanced training dataset
├── api_vulnerability_dataset_final.json      # Final merged dataset
├── api_rules.jsonl                           # Default security rules
│
└── finetuned_model/
    └── final/                      # Saved fine-tuned model weights
```

---

## 🧠 How It Works

### Phase 1 — Fine-Tuning (done once offline)

```
Dataset (2475 samples)
    ↓
Code Llama 7B + QLoRA (4-bit quantization)
    ↓
Fine-tuned model saved to finetuned_model/final/
```

**Dataset:**
- 2475 samples across 10 languages, 12 frameworks, 19 vulnerability types
- Each sample has: vulnerable code, flaw labels, CWE IDs,
  severity, description, and secure version
- Sources: synthetic generated + augmented samples

**Model:**
- Base: `codellama/CodeLlama-7b-instruct-hf`
- Method: QLoRA (only ~0.5% of parameters trained)
- Task: Given a vulnerable API endpoint → output flaw type +
  CWE + severity + description + secure version

---

### Phase 2 — Inference Pipeline (runs on any GitHub repo)

```
GitHub Repo URL
      ↓
1. Endpoint Extractor   → finds all API routes in the codebase
      ↓
2. Fine-tuned Model     → detects code-level vulnerabilities
      ↓
3. Rules Checker        → validates against security rules
      ↓
4. Report Generator     → combined HTML/JSON vulnerability report
```

---

## 🚀 Running the App

```bash
# Install dependencies
pip install streamlit gitpython requests transformers peft torch pyyaml

# Run the Streamlit UI
streamlit run app.py
```

---

## 🖥️ UI Features

### Search Box
- Type keywords to search GitHub repos
- Or paste a full GitHub URL: `https://github.com/user/repo`
- Or paste shorthand: `owner/repo`

### Custom Rules Upload *(optional)*
Upload your own API documentation or rules file:

| Format     | What happens                                      |
|------------|---------------------------------------------------|
| `.jsonl`   | Loaded directly as rules                          |
| `.json`    | Loaded as rules array                             |
| `.yaml`    | Parsed as OpenAPI spec (params + auth rules)      |
| `.md/.txt` | Scanned for security-relevant sentences           |

**Fallback logic:**
```
Upload provided  → use uploaded rules
No upload        → use api_rules.jsonl (default, 1370 rules)
Neither exists   → skip rules check, run model only
```

### Scan Results
- Summary metrics: Total / Vulnerable / Clean / Critical / High / Medium
- Each vulnerable endpoint shows:
  - Severity badge + CWE identifier
  - File location + framework
  - Tab 1: Vulnerable code
  - Tab 2: Secure version (model-generated fix)
  - Tab 3: Rules violations
- Download full JSON report

---

## 🔍 Supported Frameworks

| Language         | Frameworks                    |
|------------------|-------------------------------|
| Python           | Flask, FastAPI, Django        |
| JavaScript       | Express.js                    |
| TypeScript       | NestJS                        |
| Java             | Spring Boot                   |
| PHP              | Laravel                       |
| Go               | Gin, net/http                 |
| Ruby             | Ruby on Rails                 |
| C#               | ASP.NET Core                  |

---

## 🛡️ Vulnerability Types Detected

| Flaw                      | CWE      | Severity |
|---------------------------|----------|----------|
| SQL Injection             | CWE-89   | Critical |
| OS Command Injection      | CWE-78   | Critical |
| Code Injection            | CWE-94   | Critical |
| Eval Injection            | CWE-95   | Critical |
| Insecure Deserialization  | CWE-502  | Critical |
| Missing Authentication    | CWE-306  | High     |
| Missing Authorization     | CWE-284  | High     |
| IDOR                      | CWE-639  | High     |
| Path Traversal            | CWE-22   | High     |
| Mass Assignment           | CWE-915  | High     |
| Unrestricted File Upload  | CWE-434  | High     |
| XSS                       | CWE-79   | Medium   |
| CSRF                      | CWE-352  | Medium   |
| Open Redirect             | CWE-601  | Medium   |
| Improper Input Validation | CWE-20   | Medium   |
| Hardcoded Credentials     | CWE-798  | Medium   |
| Information Disclosure    | CWE-200  | Low      |
| Improper Error Handling   | CWE-209  | Low      |

---

## ⚙️ Fine-Tuning Details

| Setting             | Value                              |
|---------------------|------------------------------------|
| Base Model          | CodeLlama-7b-instruct-hf           |
| Method              | QLoRA (4-bit NF4 quantization)     |
| LoRA Rank           | 16                                 |
| LoRA Alpha          | 32                                 |
| Target Modules      | q_proj, k_proj, v_proj, o_proj     |
| Epochs              | 3 (early stopping patience=3)      |
| Batch Size          | 4 (effective 16 with grad accum)   |
| Learning Rate       | 2e-4                               |
| LR Scheduler        | Cosine                             |
| Optimizer           | paged_adamw_32bit                  |
| GPU Required        | T4 16GB (Google Colab free tier)   |

---

## 📊 Dataset Stats

| Split | Samples |
|-------|---------|
| Train | 1,974   |
| Val   | 241     |
| Test  | 260     |
| Total | 2,475   |

---

## 📦 Requirements

```
torch
transformers
datasets
peft
bitsandbytes
accelerate
trl
streamlit
gitpython
requests
pyyaml
anthropic        # only needed for generate_synthetic.py
```

---

## 🗺️ Research Question

> How can we test API endpoints for security vulnerabilities
> using rules extracted from documentation and provide
> interpretable, actionable feedback?

**Our approach:**
Separate vulnerability detection into three stages:
1. Structured endpoint extraction from source code
2. Fine-tuned LLM-based vulnerability detection
3. Rule-based spec validation

This combination catches both **code-level** vulnerabilities
(SQL injection, IDOR, etc.) and **contract-level** violations
(missing required params, invalid enum values, missing auth).

---

## 👥 Team

Siddhanth Nilesh Jagtap · Tanuj Kenchannavar · Harsha Raj Kumar

CS6380 — API Security Project

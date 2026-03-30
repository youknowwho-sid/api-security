import json
import random
import os
from collections import Counter, defaultdict

random.seed(42)

INPUT_PATH = "./api_vulnerability_dataset_10k.json"
OUTPUT_DIR = "./prepared_data"
MODEL = "all"   # "codellama", "starcoder", "codebert", or "all"
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
SEED = 42

SUPPORTED_MODELS = ["codellama", "starcoder", "codebert", "all"]
SEVERITY_MAP = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


def to_master(sample: dict) -> dict:
    severity_str = str(sample.get("severity", "none")).lower()
    return {
        "id": sample["id"],
        "source_dataset": sample.get("source_dataset", "unknown"),

        "code": sample["code"],
        "secure_version": sample.get("secure_version", ""),
        "language": sample.get("language", ""),
        "framework": sample.get("framework", ""),
        "method": sample.get("label", ""),
        "endpoint_path": sample.get("endpoint_path", ""),
        "resource": sample.get("resource", ""),

        "flaws": sample.get("flaws", []),
        "cwe": sample.get("cwe", []),
        "severity": severity_str,
        "severity_label": SEVERITY_MAP.get(severity_str, 0),
        "vulnerability_description": sample.get("vulnerability_description", ""),
        "is_vulnerable": len(sample.get("flaws", [])) > 0,

        "input_text": (
            f"[LANG] {sample.get('language', '')} "
            f"[FRAMEWORK] {sample.get('framework', '')} "
            f"[METHOD] {sample.get('label', '')} "
            f"[PATH] {sample.get('endpoint_path', '')} "
            f"[CODE] {sample.get('code', '')}"
        ),
    }


def prompt_codellama(master: dict, include_response: bool = True) -> dict:
    prompt = (
        f"### Instruction:\n"
        f"You are a security expert. Analyze the following "
        f"{master['language']} {master['framework']} API endpoint "
        f"for security vulnerabilities.\n"
        f"Identify any flaws, their CWE identifiers, severity, "
        f"and provide a secure version of the code.\n\n"
        f"### Endpoint:\n"
        f"Method: {master['method']}\n"
        f"Path: {master['endpoint_path']}\n\n"
        f"### Code:\n{master['code']}\n\n"
        f"### Response:"
    )

    if include_response:
        flaws = ", ".join(master["flaws"]) if master["flaws"] else "none"
        cwes = ", ".join(master["cwe"]) if master["cwe"] else "N/A"
        secure = master["secure_version"] or "No changes needed."
        prompt += (
            f"\nVulnerability Type: {flaws}"
            f"\nCWE: {cwes}"
            f"\nSeverity: {master['severity']}"
            f"\nDescription: {master['vulnerability_description']}"
            f"\nSecure Version:\n{secure}"
        )

    return {
        "id": master["id"],
        "text": prompt,
        "flaws": master["flaws"],
        "cwe": master["cwe"],
        "severity": master["severity"],
    }


def prompt_starcoder(master: dict, include_response: bool = True) -> dict:
    flaws = ", ".join(master["flaws"]) if master["flaws"] else "none"
    cwes = ", ".join(master["cwe"]) if master["cwe"] else "N/A"

    prompt = (
        f"<filename>vulnerable_api.py\n"
        f"<issue>{flaws} ({cwes}) in "
        f"{master['language']} {master['framework']} "
        f"{master['method']} {master['endpoint_path']}\n"
        f"<code>\n{master['code']}\n"
        f"<fixed_code>"
    )

    if include_response:
        secure = master["secure_version"] or master["code"]
        prompt += f"\n{secure}"

    return {
        "id": master["id"],
        "text": prompt,
        "flaws": master["flaws"],
        "cwe": master["cwe"],
        "severity": master["severity"],
    }


def prompt_codebert(master: dict, include_response: bool = True) -> dict:
    return {
        "id": master["id"],
        "input_text": master["input_text"],
        "severity_label": master["severity_label"],
        "flaw_labels": master["flaws"],
        "is_vulnerable": master["is_vulnerable"],
        "cwe": master["cwe"],
    }


MODEL_BUILDERS = {
    "codellama": prompt_codellama,
    "starcoder": prompt_starcoder,
    "codebert": prompt_codebert,
}

REQUIRED_FIELDS = [
    "id",
    "code",
    "label",
    "language",
    "framework",
    "endpoint_path",
    "flaws",
    "cwe",
    "severity",
    "vulnerability_description",
]


def load_json_any_shape(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Case 1: already a list of records
    if isinstance(data, list):
        return data

    # Case 2: dict containing the records under a common key
    if isinstance(data, dict):
        for key in ["data", "records", "samples", "items"]:
            if key in data and isinstance(data[key], list):
                print(f"[OK] Found records under top-level key: {key}")
                return data[key]

    raise ValueError(
        "Unsupported JSON format. Expected either a list of records "
        "or a dict containing one of: data, records, samples, items."
    )


def load_and_validate(path: str) -> list:
    data = load_json_any_shape(path)
    print(f"[OK] Loaded {len(data)} samples from: {path}")

    valid = []
    skipped_type = []
    skipped_missing = []

    for i, s in enumerate(data):
        if not isinstance(s, dict):
            skipped_type.append((i, type(s).__name__, str(s)[:100]))
            continue

        missing = [f for f in REQUIRED_FIELDS if f not in s]
        if missing:
            skipped_missing.append((s.get("id", f"index_{i}"), missing))
        else:
            valid.append(s)

    if skipped_type:
        print(f"[!] Skipped {len(skipped_type)} non-dict samples")
        for idx, typ, preview in skipped_type[:5]:
            print(f"    index={idx}, type={typ}, value={preview}")

    if skipped_missing:
        print(f"[!] Skipped {len(skipped_missing)} samples with missing fields")
        for sid, fields in skipped_missing[:5]:
            print(f"    {sid}: {fields}")

    print(f"[OK] {len(valid)} valid samples after validation")
    return valid


def report_balance(data: list):
    flaw_counts = Counter()
    severity_counts = Counter()
    lang_counts = Counter()

    for s in data:
        for flaw in s.get("flaws", []):
            flaw_counts[flaw] += 1
        severity_counts[s.get("severity", "?")] += 1
        lang_counts[s.get("language", "?")] += 1

    print("\n-- Severity --")
    for k, v in severity_counts.most_common():
        bar = "█" * (v // 20)
        print(f"  {k:<10} {v:>5}  {bar}")

    print("\n-- Top Flaws --")
    for k, v in flaw_counts.most_common(10):
        print(f"  {k:<35} {v}")

    print("\n-- Languages --")
    for k, v in lang_counts.most_common():
        print(f"  {k:<20} {v}")


def split_dataset(data: list, train=0.80, val=0.10, seed=42):
    random.seed(seed)
    groups = defaultdict(list)

    for s in data:
        key = s["flaws"][0] if s.get("flaws") else "none"
        groups[key].append(s)

    train_set, val_set, test_set = [], [], []

    for group in groups.values():
        random.shuffle(group)
        n = len(group)

        n_train = int(n * train)
        n_val = int(n * val)

        if n >= 3:
            n_train = max(1, min(n_train, n - 2))
            n_val = max(1, min(n_val, n - n_train - 1))
        elif n == 2:
            n_train, n_val = 1, 0
        else:
            n_train, n_val = 1, 0

        train_set.extend(group[:n_train])
        val_set.extend(group[n_train:n_train + n_val])
        test_set.extend(group[n_train + n_val:])

    print(f"\n[OK] Split -> Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
    return train_set, val_set, test_set


def save_jsonl(records: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records):>5} records -> {path}")


def save_master_splits(train, val, test, out_dir: str):
    print("\n[Master Format]")
    save_jsonl([to_master(s) for s in train], f"{out_dir}/master/train.jsonl")
    save_jsonl([to_master(s) for s in val],   f"{out_dir}/master/val.jsonl")
    save_jsonl([to_master(s) for s in test],  f"{out_dir}/master/test.jsonl")


def save_model_splits(train, val, test, model: str, out_dir: str):
    builder = MODEL_BUILDERS[model]
    print(f"\n[{model.upper()} Format]")

    save_jsonl(
        [builder(to_master(s), include_response=True) for s in train],
        f"{out_dir}/{model}/train.jsonl"
    )
    save_jsonl(
        [builder(to_master(s), include_response=True) for s in val],
        f"{out_dir}/{model}/val.jsonl"
    )
    save_jsonl(
        [builder(to_master(s), include_response=False) for s in test],
        f"{out_dir}/{model}/test.jsonl"
    )


def preview(sample: dict, model: str):
    master = to_master(sample)
    builder = MODEL_BUILDERS[model]
    entry = builder(master, include_response=True)

    print(f"\n{'=' * 55}")
    print(f"  PREVIEW — {model.upper()}")
    print(f"{'=' * 55}")

    if model == "codebert":
        for k, v in entry.items():
            print(f"  {k}: {v}")
    else:
        print(entry["text"])


def main():
    print("=" * 55)
    print("  Universal API Security Data Preparation")
    print("=" * 55)

    print(f"[INFO] INPUT_PATH = {INPUT_PATH}")
    print(f"[INFO] OUTPUT_DIR = {OUTPUT_DIR}")

    data = load_and_validate(INPUT_PATH)
    if not data:
        print("[!] No valid samples found. Exiting.")
        return

    report_balance(data)

    train, val, test = split_dataset(
        data,
        train=TRAIN_RATIO,
        val=VAL_RATIO,
        seed=SEED
    )

    save_master_splits(train, val, test, OUTPUT_DIR)

    models_to_run = (
        ["codellama", "starcoder", "codebert"]
        if MODEL == "all"
        else [MODEL]
    )

    for model in models_to_run:
        save_model_splits(train, val, test, model, OUTPUT_DIR)

    if train:
        for model in models_to_run:
            preview(train[0], model)

    print(f"\n[OK] All done! Output directory: {OUTPUT_DIR}")
    print("""
Directory structure:
  /kaggle/working/prepared_data/
    master/
      train.jsonl
      val.jsonl
      test.jsonl
    codellama/
      train.jsonl
      val.jsonl
      test.jsonl
    starcoder/
      train.jsonl
      val.jsonl
      test.jsonl
    codebert/
      train.jsonl
      val.jsonl
      test.jsonl
    """)


main()
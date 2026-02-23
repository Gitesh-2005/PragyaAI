"""
prepare_dataset.py

Collects and merges two datasets:
  1. Abhaykoul/Ancient-Indian-Wisdom  (HuggingFace)
  2. Bhagavad Gita Q&A CSV            (Kaggle - must be downloaded first)

Converts them into the exact JSONL format used by llm-jepa's gsm8k dataset:
  {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

Usage:
  python prepare_dataset.py

Output:
  datasets/ancient_wisdom_train.jsonl
  datasets/ancient_wisdom_test.jsonl

Requirements:
  pip install datasets pandas scikit-learn

Kaggle dataset must be downloaded manually first:
  kaggle datasets download -d rambo011/bhagavad-gita-q-and-a-dataset-for-modern-life-problem
  unzip bhagavad-gita-q-and-a-dataset-for-modern-life-problem.zip -d gita_raw
"""

import os
import json
import glob
import random
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
OUTPUT_DIR        = "datasets"
TRAIN_FILE        = os.path.join(OUTPUT_DIR, "ancient_wisdom_train.jsonl")
TEST_FILE         = os.path.join(OUTPUT_DIR, "ancient_wisdom_test.jsonl")
GITA_RAW_DIR      = "gita_raw"          # folder created by unzip
TEST_SIZE         = 0.15                # 15% held out for test
RANDOM_SEED       = 42

# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a wise and compassionate guide deeply versed in the spiritual, philosophical, and ethical teachings of ancient Indian wisdom traditions — including the Bhagavad Gita, Upanishads, Vedas, and related texts. When answering questions, draw upon these teachings to offer thoughtful, grounded, and practical guidance. Structure your answers clearly: begin by identifying the core principle or teaching that applies, then explain it in accessible language, and finally connect it to the human situation or question being asked. Be respectful of all spiritual paths. Avoid dogmatism. Where relevant, cite the specific text, chapter, or verse that supports your answer (e.g., Bhagavad Gita 2.47). Your tone should be warm, thoughtful, and encouraging — like a knowledgeable teacher speaking to a sincere student."
)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def make_message(user_text: str, assistant_text: str) -> dict:
    """Build a single JSONL record in gsm8k llm-jepa format."""
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_text.strip()},
            {"role": "assistant", "content": assistant_text.strip()},
        ]
    }


def write_jsonl(records: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records):,} records → {path}")


# ─────────────────────────────────────────────
# SOURCE 1 — Ancient Indian Wisdom (HuggingFace)
# ─────────────────────────────────────────────

def load_ancient_wisdom() -> list:
    print("\n[1/2] Loading Abhaykoul/Ancient-Indian-Wisdom from HuggingFace...")
    ds = load_dataset("Abhaykoul/Ancient-Indian-Wisdom", split="train")
    df = pd.DataFrame(ds)

    print(f"  Columns: {list(df.columns)}")
    print(f"  Rows   : {len(df):,}")
    print(df.head(2).to_string())

    records = []

    # Try common column name patterns — adjust if your dataset differs
    # Priority order for question column
    q_candidates = ["question", "query", "input", "prompt", "title", "text"]
    # Priority order for answer column
    a_candidates = ["answer", "response", "output", "explanation", "content", "description"]

    q_col = next((c for c in q_candidates if c in df.columns), None)
    a_col = next((c for c in a_candidates if c in df.columns), None)

    if q_col is None or a_col is None:
        print(f"  WARNING: Could not auto-detect question/answer columns.")
        print(f"  Available columns: {list(df.columns)}")
        print(f"  Edit this script and set q_col / a_col manually.")
        return []

    print(f"  Using question='{q_col}', answer='{a_col}'")

    for _, row in df.iterrows():
        q = str(row[q_col]).strip()
        a = str(row[a_col]).strip()
        if q and a and q != "nan" and a != "nan":
            records.append(make_message(q, a))

    print(f"  Valid records: {len(records):,}")
    return records


# ─────────────────────────────────────────────
# SOURCE 2 — Bhagavad Gita Q&A (Kaggle CSV)
# ─────────────────────────────────────────────

def load_gita_qa() -> list:
    print(f"\n[2/2] Loading Bhagavad Gita Q&A from '{GITA_RAW_DIR}'...")

    if not os.path.exists(GITA_RAW_DIR):
        print(f"  WARNING: '{GITA_RAW_DIR}' not found.")
        print("  Run these commands first:")
        print("    kaggle datasets download -d rambo011/bhagavad-gita-q-and-a-dataset-for-modern-life-problem")
        print("    unzip bhagavad-gita-q-and-a-dataset-for-modern-life-problem.zip -d gita_raw")
        return []

    # Find any CSV files in the folder (recursive)
    csv_files = glob.glob(os.path.join(GITA_RAW_DIR, "**", "*.csv"), recursive=True)
    if not csv_files:
        print(f"  WARNING: No CSV files found inside '{GITA_RAW_DIR}'.")
        return []

    all_frames = []
    for csv_path in csv_files:
        print(f"  Reading: {csv_path}")
        try:
            df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
            all_frames.append(df)
            print(f"    Columns: {list(df.columns)}  |  Rows: {len(df):,}")
        except Exception as e:
            print(f"    SKIP (error: {e})")

    if not all_frames:
        return []

    df = pd.concat(all_frames, ignore_index=True)
    print(f"  Total combined rows: {len(df):,}")

    # Auto-detect columns
    q_candidates = ["question", "query", "input", "prompt", "Question", "Query"]
    a_candidates = ["answer", "response", "output", "Answer", "Response", "explanation"]

    q_col = next((c for c in q_candidates if c in df.columns), None)
    a_col = next((c for c in a_candidates if c in df.columns), None)

    if q_col is None or a_col is None:
        print(f"  WARNING: Could not auto-detect Q/A columns in Gita CSV.")
        print(f"  Available: {list(df.columns)}")
        print("  Edit this script and set q_col / a_col manually.")
        return []

    print(f"  Using question='{q_col}', answer='{a_col}'")

    records = []
    for _, row in df.iterrows():
        q = str(row[q_col]).strip()
        a = str(row[a_col]).strip()
        if q and a and q != "nan" and a != "nan":
            records.append(make_message(q, a))

    print(f"  Valid records: {len(records):,}")
    return records


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Dataset Preparation for LLM-JEPA Fine-tuning")
    print("  Ancient Indian Wisdom + Bhagavad Gita Q&A")
    print("=" * 60)

    # Load both sources
    records_wisdom = load_ancient_wisdom()
    records_gita   = load_gita_qa()

    # Merge
    all_records = records_wisdom + records_gita
    print(f"\nTotal merged records: {len(all_records):,}")

    if len(all_records) == 0:
        print("\nERROR: No records collected. Check warnings above.")
        return

    # Shuffle
    random.seed(RANDOM_SEED)
    random.shuffle(all_records)

    # Train / test split
    train_records, test_records = train_test_split(
        all_records,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED
    )

    print(f"\nSplit → Train: {len(train_records):,}  |  Test: {len(test_records):,}")

    # Write output
    print(f"\nWriting output files to '{OUTPUT_DIR}/'...")
    write_jsonl(train_records, TRAIN_FILE)
    write_jsonl(test_records,  TEST_FILE)

    # Preview one record
    print("\n--- Sample record (train[0]) ---")
    sample = train_records[0]
    for msg in sample["messages"]:
        role    = msg["role"].upper()
        content = msg["content"][:120] + ("..." if len(msg["content"]) > 120 else "")
        print(f"  [{role}]: {content}")

    print("\nDone! You can now fine-tune with:")
    print(f"  python finetune.py --train_file {TRAIN_FILE} --output_dir ./fine-tuned-wisdom \\")
    print(f"    --num_epochs 4 --finetune_seed 42 --regular \\")
    print(f"    --model_name meta-llama/Llama-3.2-1B-Instruct \\")
    print(f"    --learning_rate 1e-5 --lora --lora_rank 16")
    print(f"\n  For LLM-JEPA:")
    print(f"  python finetune.py --train_file {TRAIN_FILE} --output_dir ./fine-tuned-wisdom-jepa \\")
    print(f"    --num_epochs 4 --finetune_seed 42 --last_token -2 --lbd 0.5 --predictors 1 \\")
    print(f"    --model_name meta-llama/Llama-3.2-1B-Instruct \\")
    print(f"    --learning_rate 1e-5 --lora --lora_rank 16")


if __name__ == "__main__":
    main()
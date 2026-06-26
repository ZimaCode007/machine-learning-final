"""
Preprocess datasets/ to add flattened table text to Input column.

Reads datasets/{split}/data.csv and tables/*.csv,
produces datasets/{split}/data_with_tables.csv with Input = "question [SEP] flattened_table"
"""
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent.parent
datasets_dir = project_root / "datasets"


def flatten_table(csv_path):
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except Exception:
        try:
            df = pd.read_csv(csv_path, encoding="latin-1")
        except Exception:
            return ""
    header = " | ".join(str(c) for c in df.columns)
    rows = []
    for _, row in df.iterrows():
        rows.append(" | ".join(str(v) for v in row.values))
    return header + " [SEP] " + " [SEP] ".join(rows)


def process_split(split):
    split_dir = datasets_dir / split
    data_path = split_dir / "data.csv"
    tables_dir = split_dir / "tables"
    output_path = split_dir / "data_with_tables.csv"

    df = pd.read_csv(data_path)
    print(f"\n[{split}] {len(df)} samples, tables dir: {tables_dir}")

    new_inputs = []
    missing = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split}"):
        question = str(row["Input"])
        img_index = str(row["Image Index"])
        table_path = tables_dir / f"{img_index}.csv"

        if table_path.exists():
            table_text = flatten_table(str(table_path))
            if table_text:
                new_inputs.append(f"{question} [SEP] {table_text}")
            else:
                new_inputs.append(question)
                missing += 1
        else:
            new_inputs.append(question)
            missing += 1

    df["Input"] = new_inputs
    df.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")
    print(f"  With table: {len(df) - missing}, Missing table: {missing}")


if __name__ == "__main__":
    for split in ["train", "validation", "test"]:
        process_split(split)
    print("\nDone! Use data_with_tables.csv for training.")

"""
Build data.csv files for VL-T5 training from ChartQA dataset.
Each row: Question ID, Image Index, Input (question + flattened table), Output (answer).

Usage:
    python prepare_data.py --chartqa_dir "../../ChartQA Dataset" --output_dir data
"""
import argparse
import json
import os
import pandas as pd
from tqdm import tqdm


def flatten_table(csv_path):
    """Read a CSV table and flatten it into a text string."""
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


def build_split(chartqa_dir, split_name, output_dir):
    """Build data.csv for one split."""
    split_dir = os.path.join(chartqa_dir, split_name)

    qa_files = []
    for fname in os.listdir(split_dir):
        if fname.endswith(".json"):
            qa_files.append(fname)

    all_qa = []
    for fname in qa_files:
        with open(os.path.join(split_dir, fname), encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                item["_source"] = fname
            all_qa.extend(data)

    print(f"[{split_name}] Loaded {len(all_qa)} QA pairs from {len(qa_files)} files")

    tables_dir = os.path.join(split_dir, "tables")

    table_cache = {}
    rows = []
    skipped = 0

    for qid, item in enumerate(tqdm(all_qa, desc=f"Building {split_name}")):
        imgname = item["imgname"]
        question = item["query"]
        answer = str(item["label"])
        img_index = os.path.splitext(imgname)[0]

        table_file = img_index + ".csv"
        table_path = os.path.join(tables_dir, table_file)

        if img_index not in table_cache:
            if os.path.exists(table_path):
                table_cache[img_index] = flatten_table(table_path)
            else:
                table_cache[img_index] = ""

        flat_table = table_cache[img_index]
        if not flat_table:
            skipped += 1
            continue

        input_text = f"{question} [SEP] {flat_table}"

        rows.append({
            "Question ID": qid,
            "Image Index": img_index,
            "Input": input_text,
            "Output": answer,
        })

    df = pd.DataFrame(rows)
    out_split_dir = os.path.join(output_dir, split_name)
    os.makedirs(out_split_dir, exist_ok=True)
    csv_path = os.path.join(out_split_dir, "data.csv")
    df.to_csv(csv_path, index=False)
    print(f"[{split_name}] Wrote {len(df)} rows to {csv_path} (skipped {skipped} missing tables)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chartqa_dir", type=str, default="../../ChartQA Dataset")
    parser.add_argument("--output_dir", type=str, default="data")
    args = parser.parse_args()

    split_map = {"train": "train", "validation": "val", "test": "test"}
    for vlt5_split, chartqa_split in split_map.items():
        build_split(args.chartqa_dir, chartqa_split, args.output_dir)
        built_dir = os.path.join(args.output_dir, chartqa_split)
        target_dir = os.path.join(args.output_dir, vlt5_split)
        if chartqa_split != vlt5_split:
            if os.path.exists(target_dir):
                import shutil
                shutil.rmtree(target_dir)
            os.rename(built_dir, target_dir)
            print(f"  Renamed {built_dir} -> {target_dir}")

    print("\nDone! Data directory structure:")
    for d in ["train", "validation", "test"]:
        p = os.path.join(args.output_dir, d, "data.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            print(f"  {p}: {len(df)} rows")


if __name__ == "__main__":
    main()

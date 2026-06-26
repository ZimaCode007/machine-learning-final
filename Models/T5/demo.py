"""
T5 Demo: evaluate on ChartQA test set and visualize predictions.

Part 1: Random 200 human + 200 augmented samples -> accuracy stats
Part 2: Random 5 human + 5 augmented samples -> one image per sample with question and prediction

Usage:
    .venv/Scripts/python.exe T5/demo.py
    .venv/Scripts/python.exe T5/demo.py --model T5/saved_model --chartqa_dir "ChartQA Dataset"
"""
import argparse
import os
import json
import random
import shutil

import torch
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from transformers import T5ForConditionalGeneration, T5TokenizerFast


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


def load_model(model_path, device):
    print(f"Loading model from {model_path}...")
    tokenizer = T5TokenizerFast.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.eval().to(device)
    return model, tokenizer


@torch.no_grad()
def predict(model, tokenizer, question, table_path, device):
    input_text = question
    if table_path and os.path.exists(table_path):
        table_text = flatten_table(table_path)
        if table_text:
            input_text = f"{question} [SEP] {table_text}"

    encoding = tokenizer(
        f"chartqa: {input_text}",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_beams=3,
        max_length=64,
    )
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer


def load_qa_data(chartqa_dir):
    with open(os.path.join(chartqa_dir, "test", "test_human.json"), encoding="utf-8") as f:
        human = json.load(f)
    with open(os.path.join(chartqa_dir, "test", "test_augmented.json"), encoding="utf-8") as f:
        augmented = json.load(f)
    return human, augmented


def run_accuracy_test(model, tokenizer, samples, chartqa_dir, device, label):
    correct = 0
    total = len(samples)
    results = []

    for item in tqdm(samples, desc=f"Evaluating {label}"):
        imgname = item["imgname"]
        question = item["query"]
        gt_answer = str(item["label"])
        img_index = os.path.splitext(imgname)[0]

        table_path = os.path.join(chartqa_dir, "test", "tables", img_index + ".csv")

        pred = predict(model, tokenizer, question, table_path, device)

        is_correct = pred.strip() == gt_answer.strip()
        if is_correct:
            correct += 1

        results.append({
            "image": imgname,
            "question": question,
            "prediction": pred,
            "ground_truth": gt_answer,
            "correct": is_correct,
        })

    acc = 100 * correct / total if total > 0 else 0
    return acc, results


def save_single_sample(item, chartqa_dir, save_path, index, category):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    image_path = os.path.join(chartqa_dir, "test", "png", item["image"])
    try:
        img = mpimg.imread(image_path)
        ax.imshow(img)
    except Exception:
        ax.text(0.5, 0.5, "Image not found", ha="center", va="center", transform=ax.transAxes)

    ax.set_xticks([])
    ax.set_yticks([])

    color = "green" if item["correct"] else "red"
    status = "CORRECT" if item["correct"] else "WRONG"

    fig.suptitle(f"[{category}] Sample {index + 1} (T5 text-only)", fontsize=16, fontweight="bold")

    info = (
        f"Q: {item['question']}\n"
        f"Prediction: {item['prediction']}    |    Ground truth: {item['ground_truth']}    |    [{status}]"
    )
    ax.set_title(info, fontsize=11, color=color, loc="left", pad=8, wrap=True)

    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="T5 Demo on ChartQA")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    parser.add_argument("--model", type=str, default=os.path.join(script_dir, "saved_model"))
    parser.add_argument("--chartqa_dir", type=str, default=os.path.join(project_dir, "ChartQA Dataset"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_eval", type=int, default=200)
    parser.add_argument("--n_show", type=int, default=5)
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    demo_dir = os.path.join(script_dir, "demo_output")
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
        print(f"Cleaned previous results in {demo_dir}")
    os.makedirs(demo_dir, exist_ok=True)

    model, tokenizer = load_model(args.model, device)

    print("Loading test data...")
    human, augmented = load_qa_data(args.chartqa_dir)

    # =========================================
    # Part 1: Accuracy evaluation (200 + 200)
    # =========================================
    print(f"\n{'='*60}")
    print(f"  PART 1: Accuracy evaluation ({args.n_eval} human + {args.n_eval} augmented)")
    print(f"{'='*60}")

    human_sample = random.sample(human, min(args.n_eval, len(human)))
    aug_sample = random.sample(augmented, min(args.n_eval, len(augmented)))

    human_acc, human_results = run_accuracy_test(
        model, tokenizer, human_sample, args.chartqa_dir, device, "Human")
    aug_acc, aug_results = run_accuracy_test(
        model, tokenizer, aug_sample, args.chartqa_dir, device, "Augmented")

    overall_correct = sum(r["correct"] for r in human_results + aug_results)
    overall_total = len(human_results) + len(aug_results)
    overall_acc = 100 * overall_correct / overall_total if overall_total > 0 else 0

    print(f"\n{'='*60}")
    print(f"  ACCURACY RESULTS (T5 text-only)")
    print(f"{'='*60}")
    print(f"  Human:     {human_acc:.1f}%  ({sum(r['correct'] for r in human_results)}/{len(human_results)})")
    print(f"  Augmented: {aug_acc:.1f}%  ({sum(r['correct'] for r in aug_results)}/{len(aug_results)})")
    print(f"  Overall:   {overall_acc:.1f}%  ({overall_correct}/{overall_total})")
    print(f"{'='*60}")

    with open(os.path.join(demo_dir, "accuracy_results.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model": "T5 (text-only)",
            "human_accuracy": round(human_acc, 2),
            "augmented_accuracy": round(aug_acc, 2),
            "overall_accuracy": round(overall_acc, 2),
            "human_results": human_results,
            "augmented_results": aug_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"Detailed results saved to {demo_dir}/accuracy_results.json")

    # =========================================
    # Part 2: Visual comparison (5 + 5)
    # =========================================
    print(f"\n{'='*60}")
    print(f"  PART 2: Visual comparison ({args.n_show} human + {args.n_show} augmented)")
    print(f"{'='*60}")

    human_dir = os.path.join(demo_dir, "human")
    aug_dir = os.path.join(demo_dir, "augmented")
    os.makedirs(human_dir, exist_ok=True)
    os.makedirs(aug_dir, exist_ok=True)

    human_show = random.sample(human, min(args.n_show, len(human)))
    aug_show = random.sample(augmented, min(args.n_show, len(augmented)))

    for i, item in enumerate(human_show):
        imgname = item["imgname"]
        img_index = os.path.splitext(imgname)[0]
        table_path = os.path.join(args.chartqa_dir, "test", "tables", img_index + ".csv")
        pred = predict(model, tokenizer, item["query"], table_path, device)
        result = {
            "image": imgname,
            "question": item["query"],
            "prediction": pred,
            "ground_truth": str(item["label"]),
            "correct": pred.strip() == str(item["label"]).strip(),
        }
        save_path = os.path.join(human_dir, f"sample_{i+1}.png")
        save_single_sample(result, args.chartqa_dir, save_path, i, "Human")
        status = "CORRECT" if result["correct"] else "WRONG"
        print(f"  [Human {i+1}] Q: {item['query']}")
        print(f"             Pred: {pred}  |  GT: {item['label']}  |  {status}")
        print(f"             Saved: {save_path}")

    for i, item in enumerate(aug_show):
        imgname = item["imgname"]
        img_index = os.path.splitext(imgname)[0]
        table_path = os.path.join(args.chartqa_dir, "test", "tables", img_index + ".csv")
        pred = predict(model, tokenizer, item["query"], table_path, device)
        result = {
            "image": imgname,
            "question": item["query"],
            "prediction": pred,
            "ground_truth": str(item["label"]),
            "correct": pred.strip() == str(item["label"]).strip(),
        }
        save_path = os.path.join(aug_dir, f"sample_{i+1}.png")
        save_single_sample(result, args.chartqa_dir, save_path, i, "Augmented")
        status = "CORRECT" if result["correct"] else "WRONG"
        print(f"  [Aug {i+1}]   Q: {item['query']}")
        print(f"             Pred: {pred}  |  GT: {item['label']}  |  {status}")
        print(f"             Saved: {save_path}")

    print(f"\n{'='*60}")
    print(f"  DEMO COMPLETE (T5 text-only)")
    print(f"{'='*60}")
    print(f"  Results directory: {demo_dir}/")
    print(f"    accuracy_results.json")
    print(f"    human/sample_1.png ~ sample_{args.n_show}.png")
    print(f"    augmented/sample_1.png ~ sample_{args.n_show}.png")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

"""
VL-T5 Demo: evaluate on ChartQA test set and visualize predictions.

Part 1: Random 200 human + 200 augmented samples -> accuracy stats
Part 2: Random 5 human + 5 augmented samples -> one image per sample

Usage:
    python demo.py --model output_v4/BEST.pth --chartqa_dir "../../ChartQA Dataset"
"""
import argparse
import sys
import os
import json
import random
import shutil

import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from tokenization import VLT5TokenizerFast
from vqa_model import VLT5VQA
from transformers import T5Config


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


class GridFeatureExtractor:
    def __init__(self, device):
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.backbone.eval().to(device)
        self.pool = nn.AdaptiveAvgPool2d((6, 6))
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, image_path):
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        feat_map = self.backbone(img_tensor)
        pooled = self.pool(feat_map)
        feats = pooled.squeeze(0).view(2048, 36).permute(1, 0)
        boxes = []
        for r in range(6):
            for c in range(6):
                boxes.append([c / 6, r / 6, (c + 1) / 6, (r + 1) / 6])
        return feats, torch.FloatTensor(boxes)


def load_model(model_path, backbone, device):
    if os.path.isdir(model_path):
        tokenizer = VLT5TokenizerFast.from_pretrained(model_path)
        model = VLT5VQA.from_pretrained(model_path)
    else:
        tokenizer = VLT5TokenizerFast.from_pretrained(backbone)
        config = T5Config.from_pretrained(backbone)
        config.feat_dim = 2048
        config.pos_dim = 4
        config.n_images = 2
        config.use_vis_order_embedding = True
        config.use_vis_layer_norm = True
        config.individual_vis_layer_norm = True
        config.share_vis_lang_layer_norm = False
        config.classifier = False
        config.losses = "lm,obj,attr,feat"
        model = VLT5VQA(config)
        model.resize_token_embeddings(len(tokenizer))
        state_dict = torch.load(model_path, map_location="cpu")
        for k in list(state_dict.keys()):
            if k.startswith("module."):
                state_dict[k[7:]] = state_dict.pop(k)
        model.load_state_dict(state_dict, strict=False)

    model.eval().to(device)
    model.tokenizer = tokenizer

    return model, tokenizer


def predict(model, tokenizer, feature_extractor, image_path, question, table_path, device):
    input_text = question
    if table_path and os.path.exists(table_path):
        table_text = flatten_table(table_path)
        if table_text:
            input_text = f"{question} [SEP] {table_text}"

    input_ids = tokenizer.encode(f"chartqa: {input_text}", max_length=400, truncation=True)
    input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device)

    vis_feats, boxes = feature_extractor.extract(image_path)
    vis_feats = vis_feats.unsqueeze(0).to(device)
    boxes = boxes.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            vis_inputs=(vis_feats, boxes),
            num_beams=1,
            max_length=20,
        )
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer


def load_qa_data(chartqa_dir):
    with open(os.path.join(chartqa_dir, "test", "test_human.json"), encoding="utf-8") as f:
        human = json.load(f)
    with open(os.path.join(chartqa_dir, "test", "test_augmented.json"), encoding="utf-8") as f:
        augmented = json.load(f)
    return human, augmented


def run_accuracy_test(model, tokenizer, feat_extractor, samples, chartqa_dir, device, label):
    correct = 0
    total = len(samples)
    results = []

    for item in tqdm(samples, desc=f"Evaluating {label}"):
        imgname = item["imgname"]
        question = item["query"]
        gt_answer = str(item["label"])
        img_index = os.path.splitext(imgname)[0]

        image_path = os.path.join(chartqa_dir, "test", "png", imgname)
        table_path = os.path.join(chartqa_dir, "test", "tables", img_index + ".csv")

        if not os.path.exists(image_path):
            total -= 1
            continue

        pred = predict(model, tokenizer, feat_extractor, image_path, question, table_path, device)

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

    fig.suptitle(f"[{category}] Sample {index + 1}", fontsize=16, fontweight="bold")

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="output_v4/BEST")
    parser.add_argument("--chartqa_dir", type=str, default="../../ChartQA Dataset")
    parser.add_argument("--backbone", type=str, default="t5-base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_eval", type=int, default=200)
    parser.add_argument("--n_show", type=int, default=5)
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    demo_dir = "output/demo"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
        print(f"Cleaned previous results in {demo_dir}")
    os.makedirs(demo_dir, exist_ok=True)

    print("Loading model...")
    model, tokenizer = load_model(args.model, args.backbone, device)
    feat_extractor = GridFeatureExtractor(device)

    print("Loading test data...")
    human, augmented = load_qa_data(args.chartqa_dir)

    print(f"\n{'='*60}")
    print(f"  PART 1: Accuracy evaluation ({args.n_eval} human + {args.n_eval} augmented)")
    print(f"{'='*60}")

    human_sample = random.sample(human, min(args.n_eval, len(human)))
    aug_sample = random.sample(augmented, min(args.n_eval, len(augmented)))

    human_acc, human_results = run_accuracy_test(
        model, tokenizer, feat_extractor, human_sample, args.chartqa_dir, device, "Human")
    aug_acc, aug_results = run_accuracy_test(
        model, tokenizer, feat_extractor, aug_sample, args.chartqa_dir, device, "Augmented")

    overall_correct = sum(r["correct"] for r in human_results + aug_results)
    overall_total = len(human_results) + len(aug_results)
    overall_acc = 100 * overall_correct / overall_total if overall_total > 0 else 0

    print(f"\n{'='*60}")
    print(f"  ACCURACY RESULTS")
    print(f"{'='*60}")
    print(f"  Human:     {human_acc:.1f}%  ({sum(r['correct'] for r in human_results)}/{len(human_results)})")
    print(f"  Augmented: {aug_acc:.1f}%  ({sum(r['correct'] for r in aug_results)}/{len(aug_results)})")
    print(f"  Overall:   {overall_acc:.1f}%  ({overall_correct}/{overall_total})")
    print(f"{'='*60}")

    with open(os.path.join(demo_dir, "accuracy_results.json"), "w", encoding="utf-8") as f:
        json.dump({
            "human_accuracy": round(human_acc, 2),
            "augmented_accuracy": round(aug_acc, 2),
            "overall_accuracy": round(overall_acc, 2),
            "human_results": human_results,
            "augmented_results": aug_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"Detailed results saved to {demo_dir}/accuracy_results.json")

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
        image_path = os.path.join(args.chartqa_dir, "test", "png", imgname)
        table_path = os.path.join(args.chartqa_dir, "test", "tables", img_index + ".csv")
        pred = predict(model, tokenizer, feat_extractor, image_path, item["query"], table_path, device)
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
        image_path = os.path.join(args.chartqa_dir, "test", "png", imgname)
        table_path = os.path.join(args.chartqa_dir, "test", "tables", img_index + ".csv")
        pred = predict(model, tokenizer, feat_extractor, image_path, item["query"], table_path, device)
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
    print(f"  DEMO COMPLETE")
    print(f"{'='*60}")
    print(f"  Results directory: {demo_dir}/")
    print(f"    accuracy_results.json")
    print(f"    human/sample_1.png ~ sample_{args.n_show}.png")
    print(f"    augmented/sample_1.png ~ sample_{args.n_show}.png")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

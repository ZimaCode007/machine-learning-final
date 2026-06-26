"""
Single image inference for VL-T5.

Usage:
    python predict_single.py --image "../../ChartQA Dataset/test/png/10099.png" --question "What is the highest value?" --table "../../ChartQA Dataset/test/tables/10099.csv" --model output/BEST.pth
"""
import argparse
import sys
import os
import json

import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from param import parse_args
from tokenization import VLT5TokenizerFast
from modeling_t5 import VLT5
from vqa_model import VLT5VQA
from transformers import T5Config


def flatten_table(csv_path):
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except Exception:
        df = pd.read_csv(csv_path, encoding="latin-1")
    header = " | ".join(str(c) for c in df.columns)
    rows = []
    for _, row in df.iterrows():
        rows.append(" | ".join(str(v) for v in row.values))
    return header + " [SEP] " + " [SEP] ".join(rows)


def extract_grid_features(image_path, device):
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    backbone = nn.Sequential(*list(resnet.children())[:-2])
    backbone.eval().to(device)
    pool = nn.AdaptiveAvgPool2d((6, 6))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat_map = backbone(img_tensor)
        pooled = pool(feat_map)

    feats = pooled.squeeze(0).view(2048, 36).permute(1, 0)

    boxes = []
    for r in range(6):
        for c in range(6):
            boxes.append([c / 6, r / 6, (c + 1) / 6, (r + 1) / 6])

    return feats, torch.FloatTensor(boxes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--table", type=str, default=None)
    parser.add_argument("--model", type=str, default="output/BEST.pth")
    parser.add_argument("--backbone", type=str, default="t5-base")
    parser.add_argument("--num_beams", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build input text
    input_text = args.question
    if args.table:
        table_text = flatten_table(args.table)
        input_text = f"{args.question} [SEP] {table_text}"

    print(f"Question: {args.question}")
    if args.table:
        print(f"Table: {args.table}")
    print(f"Image: {args.image}")
    print()

    # Tokenizer
    tokenizer = VLT5TokenizerFast.from_pretrained(args.backbone)

    # Model config
    config = T5Config.from_pretrained(args.backbone)
    config.feat_dim = 2048
    config.pos_dim = 4
    config.n_images = 2
    config.use_vis_order_embedding = True
    config.use_vis_layer_norm = True
    config.individual_vis_layer_norm = True
    config.share_vis_lang_layer_norm = False
    config.classifier = False
    config.losses = "lm,obj,attr,feat"

    # Build model
    model = VLT5VQA(config)
    model.resize_token_embeddings(len(tokenizer))

    # Load weights
    print(f"Loading model from {args.model}...")
    state_dict = torch.load(args.model, map_location="cpu")
    original_keys = list(state_dict.keys())
    for key in original_keys:
        if key.startswith("module."):
            state_dict[key[len("module."):]] = state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    model.tokenizer = tokenizer

    # Extract visual features
    print("Extracting visual features...")
    vis_feats, boxes = extract_grid_features(args.image, device)
    vis_feats = vis_feats.unsqueeze(0).to(device)
    boxes = boxes.unsqueeze(0).to(device)

    # Tokenize input
    input_ids = tokenizer.encode(f"chartqa: {input_text}", max_length=400, truncation=True)
    input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device)

    # Generate answer
    print("Generating answer...")
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            vis_inputs=(vis_feats, boxes),
            num_beams=args.num_beams,
            max_length=20,
        )
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    print()
    print("=" * 50)
    print(f"  Question: {args.question}")
    print(f"  Answer:   {answer}")
    print("=" * 50)


if __name__ == "__main__":
    main()

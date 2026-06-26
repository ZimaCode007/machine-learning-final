"""
Extract ResNet50 grid-based visual features from chart images.
Splits each image into a 6x6 grid → 36 regions of 2048-dim features.
Compatible with VL-T5's expected input format.

Usage:
    python extract_features.py --chartqa_dir "../../ChartQA Dataset" --output_dir data
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


class GridFeatureExtractor:
    def __init__(self, device, grid_size=6):
        self.device = device
        self.grid_size = grid_size
        self.n_boxes = grid_size * grid_size  # 36

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.backbone.eval().to(device)
        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))

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

        G = self.grid_size
        feats = pooled.squeeze(0).view(2048, G * G).permute(1, 0).cpu().numpy()

        boxes = []
        for r in range(G):
            for c in range(G):
                x1 = c / G
                y1 = r / G
                x2 = (c + 1) / G
                y2 = (r + 1) / G
                boxes.append([x1, y1, x2, y2])
        boxes = np.array(boxes)

        return {
            "visual_feats": feats.tolist(),
            "bboxes": boxes.tolist(),
        }


def process_split(extractor, chartqa_split_dir, output_features_dir):
    png_dir = os.path.join(chartqa_split_dir, "png")
    os.makedirs(output_features_dir, exist_ok=True)

    images = [f for f in os.listdir(png_dir) if f.lower().endswith(".png")]
    existing = set(os.listdir(output_features_dir))
    todo = [f for f in images if os.path.splitext(f)[0] + ".json" not in existing]
    print(f"{len(todo)} images to process from {png_dir} ({len(images) - len(todo)} already done)")

    for fname in tqdm(todo, desc="Extracting"):
        img_index = os.path.splitext(fname)[0]
        out_path = os.path.join(output_features_dir, f"{img_index}.json")
        try:
            features = extractor.extract(os.path.join(png_dir, fname))
            with open(out_path, "w") as f:
                json.dump(features, f)
        except Exception as e:
            print(f"Error {fname}: {e}")
            features = {
                "visual_feats": np.zeros((36, 2048)).tolist(),
                "bboxes": np.zeros((36, 4)).tolist(),
            }
            with open(out_path, "w") as f:
                json.dump(features, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chartqa_dir", type=str, default="../../ChartQA Dataset")
    parser.add_argument("--output_dir", type=str, default="data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    extractor = GridFeatureExtractor(device)

    split_map = {"train": "train", "validation": "val", "test": "test"}
    for vlt5_split, chartqa_split in split_map.items():
        print(f"\n=== {vlt5_split} ===")
        process_split(
            extractor,
            os.path.join(args.chartqa_dir, chartqa_split),
            os.path.join(args.output_dir, vlt5_split, "features"),
        )

    print("\nDone!")
    for d in ["train", "validation", "test"]:
        feat_dir = os.path.join(args.output_dir, d, "features")
        if os.path.exists(feat_dir):
            print(f"  {feat_dir}: {len(os.listdir(feat_dir))} files")


if __name__ == "__main__":
    main()

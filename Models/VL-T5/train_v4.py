"""
VL-T5 training script for the redesigned dataset at E:/UE/machine learning/final/datasets/
Automatically enriches Input with flattened table data and extracts visual features on-the-fly.
Fine-tunes from Epoch30.pth pretrained checkpoint.

Usage:
    python train_v4.py
    python train_v4.py --epochs 10 --batch_size 16
"""
import sys
import os
import json
import argparse

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from tokenization import VLT5TokenizerFast
from vqa_model import VLT5VQA
from transformers import T5Config
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup


DATASETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets"))
CHECKPOINT = os.path.join(os.path.dirname(__file__), "Epoch30.pth")


def flatten_table(csv_path):
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except Exception:
        try:
            df = pd.read_csv(csv_path, encoding="latin-1")
        except Exception:
            return ""
    header = " | ".join(str(c) for c in df.columns)
    rows = " [SEP] ".join(
        " | ".join(str(v) for v in row.values) for _, row in df.iterrows()
    )
    return header + " [SEP] " + rows


class ChartQADataset(Dataset):
    def __init__(self, split_dir, tokenizer, feature_extractor, max_text_length=400):
        self.split_dir = split_dir
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_text_length = max_text_length

        self.df = pd.read_csv(os.path.join(split_dir, "data.csv"))
        self.tables_dir = os.path.join(split_dir, "tables")
        self.png_dir = os.path.join(split_dir, "png")

        self.table_cache = {}

    def __len__(self):
        return len(self.df)

    def _get_table(self, img_index):
        img_index = str(img_index)
        if img_index not in self.table_cache:
            table_path = os.path.join(self.tables_dir, img_index + ".csv")
            if os.path.exists(table_path):
                self.table_cache[img_index] = flatten_table(table_path)
            else:
                self.table_cache[img_index] = ""
        return self.table_cache[img_index]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        question = str(row["Input"])
        answer = str(row["Output"])
        img_index = str(row["Image Index"])
        qid = int(row["Question ID"])

        table_text = self._get_table(img_index)
        if table_text:
            input_text = f"{question} [SEP] {table_text}"
        else:
            input_text = question

        input_ids = self.tokenizer.encode(
            f"chartqa: {input_text}", max_length=self.max_text_length, truncation=True
        )
        target_ids = self.tokenizer.encode(answer, max_length=100, truncation=True)

        image_path = os.path.join(self.png_dir, img_index + ".png")
        feat_path = os.path.join(self.split_dir, "features", img_index + ".json")

        if os.path.exists(feat_path):
            with open(feat_path) as f:
                feat_data = json.load(f)
            vis_feats = torch.FloatTensor(feat_data["visual_feats"])
            boxes = torch.FloatTensor(feat_data["bboxes"])
        elif os.path.exists(image_path):
            vis_feats, boxes = self.feature_extractor.extract(image_path)
        else:
            vis_feats = torch.zeros(36, 2048)
            boxes = torch.zeros(36, 4)

        return {
            "input_ids": torch.LongTensor(input_ids),
            "target_ids": torch.LongTensor(target_ids),
            "vis_feats": vis_feats[:36],
            "boxes": boxes[:36],
            "answer": answer,
            "question_id": qid,
        }


def collate_fn(batch, pad_token_id):
    B = len(batch)
    max_input = max(len(b["input_ids"]) for b in batch)
    max_target = max(len(b["target_ids"]) for b in batch)

    input_ids = torch.full((B, max_input), pad_token_id, dtype=torch.long)
    target_ids = torch.full((B, max_target), -100, dtype=torch.long)
    vis_feats = torch.zeros(B, 36, 2048)
    boxes = torch.zeros(B, 36, 4)
    scores = torch.ones(B)

    answers = []
    question_ids = []

    for i, b in enumerate(batch):
        input_ids[i, : len(b["input_ids"])] = b["input_ids"]
        target_ids[i, : len(b["target_ids"])] = b["target_ids"]
        vis_feats[i] = b["vis_feats"]
        boxes[i] = b["boxes"]
        answers.append(b["answer"])
        question_ids.append(b["question_id"])

    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "vis_feats": vis_feats,
        "boxes": boxes,
        "scores": scores,
        "answers": answers,
        "question_ids": question_ids,
    }


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
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, image_path):
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        feat_map = self.backbone(img_tensor)
        pooled = self.pool(feat_map)
        feats = pooled.squeeze(0).view(2048, 36).permute(1, 0).cpu()
        boxes = []
        for r in range(6):
            for c in range(6):
                boxes.append([c / 6, r / 6, (c + 1) / 6, (r + 1) / 6])
        return feats, torch.FloatTensor(boxes)


def build_model(backbone, checkpoint_path, device):
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
    config.dropout_rate = 0.1
    config.dropout = 0.1
    config.attention_dropout = 0.1
    config.activation_dropout = 0.1

    model = VLT5VQA(config)
    model.resize_token_embeddings(len(tokenizer))

    print(f"Loading pretrained weights from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    for k in list(state_dict.keys()):
        if k.startswith("module."):
            state_dict[k[7:]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)

    model.tokenizer = tokenizer
    model.to(device)
    return model, tokenizer


def evaluate(model, tokenizer, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", ncols=80):
            input_ids = batch["input_ids"].to(device)
            vis_feats = batch["vis_feats"].to(device)
            boxes = batch["boxes"].to(device)

            output = model.generate(
                input_ids=input_ids,
                vis_inputs=(vis_feats, boxes),
                num_beams=1,
                max_length=20,
            )
            preds = tokenizer.batch_decode(output, skip_special_tokens=True)

            for pred, gt in zip(preds, batch["answers"]):
                if pred.strip() == str(gt).strip():
                    correct += 1
                total += 1

    return 100 * correct / total if total > 0 else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_dir", type=str, default=DATASETS_DIR)
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT)
    parser.add_argument("--backbone", type=str, default="t5-base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--clip_grad_norm", type=float, default=5.0)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--output", type=str, default="output_v4")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(args.output, exist_ok=True)

    print("Building model...")
    model, tokenizer = build_model(args.backbone, args.checkpoint, device)
    feat_extractor = GridFeatureExtractor(device)

    print(f"Params: {sum(p.numel() for p in model.parameters()) // 1_000_000}M")

    print("Loading datasets...")
    train_ds = ChartQADataset(
        os.path.join(args.datasets_dir, "train"), tokenizer, feat_extractor
    )
    val_ds = ChartQADataset(
        os.path.join(args.datasets_dir, "validation"), tokenizer, feat_extractor
    )
    test_ds = ChartQADataset(
        os.path.join(args.datasets_dir, "test"), tokenizer, feat_extractor
    )

    pad_id = tokenizer.pad_token_id
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-6)

    t_total = len(train_loader) * args.epochs
    warmup_steps = int(t_total * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, t_total)

    scaler = GradScaler() if args.fp16 else None

    from torch.utils.tensorboard import SummaryWriter
    tb_writer = SummaryWriter(os.path.join(args.output, "tb_logs"))
    metrics_log = []
    metrics_path = os.path.join(args.output, "metrics.json")

    best_val_acc = 0
    global_step = 0

    print(f"\nTraining for {args.epochs} epochs...")
    print(f"Output: {args.output}/")
    print(f"TensorBoard: {args.output}/tb_logs/")
    print()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", ncols=120)

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            vis_feats = batch["vis_feats"].to(device)
            boxes = batch["boxes"].to(device)
            scores = batch["scores"].to(device)

            if args.fp16:
                with autocast():
                    output = model(
                        input_ids=input_ids,
                        vis_inputs=(vis_feats, boxes),
                        labels=target_ids,
                        return_dict=True,
                    )
                    loss = output["loss"]
                    lm_mask = (target_ids != -100).float()
                    B, L = target_ids.size()
                    loss = loss.view(B, L) * lm_mask
                    loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)
                    loss = (loss * scores).mean()

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(
                    input_ids=input_ids,
                    vis_inputs=(vis_feats, boxes),
                    labels=target_ids,
                    return_dict=True,
                )
                loss = output["loss"]
                lm_mask = (target_ids != -100).float()
                B, L = target_ids.size()
                loss = loss.view(B, L) * lm_mask
                loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)
                loss = (loss * scores).mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

            loss_val = loss.item()
            epoch_loss += loss_val
            global_step += 1

            lr = scheduler.get_last_lr()[0]
            pbar.set_description(f"Epoch {epoch} | LR {lr:.6f} | Loss {loss_val:.4f}")

            tb_writer.add_scalar("Train/loss_step", loss_val, global_step)
            tb_writer.add_scalar("Train/lr", lr, global_step)

        avg_loss = epoch_loss / len(train_loader)

        val_acc = evaluate(model, tokenizer, val_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.output, "BEST.pth"))
            print(f"  Saved new BEST model (val acc: {val_acc:.2f}%)")

        tb_writer.add_scalar("Epoch/train_loss", avg_loss, epoch)
        tb_writer.add_scalar("Epoch/val_accuracy", val_acc, epoch)
        tb_writer.flush()

        metrics_log.append({
            "epoch": epoch,
            "train_loss": round(avg_loss, 6),
            "val_accuracy": round(val_acc, 2),
            "best_val_accuracy": round(best_val_acc, 2),
            "lr": lr,
        })
        with open(metrics_path, "w") as f:
            json.dump(metrics_log, f, indent=2)

        print(f"\nEpoch {epoch}: Train Loss {avg_loss:.4f}")
        print(f"Epoch {epoch}: Val Accuracy {val_acc:.2f}%")
        print(f"Epoch {epoch}: Best {best_val_acc:.2f}%\n")

    torch.save(model.state_dict(), os.path.join(args.output, "LAST.pth"))

    print("Evaluating on test set...")
    test_acc = evaluate(model, tokenizer, test_loader, device)
    print(f"Test Accuracy: {test_acc:.2f}%")

    metrics_log.append({"test_accuracy": round(test_acc, 2)})
    with open(metrics_path, "w") as f:
        json.dump(metrics_log, f, indent=2)

    tb_writer.close()

    print(f"\nDone! Results saved to {args.output}/")
    print(f"  BEST.pth: best validation model ({best_val_acc:.2f}%)")
    print(f"  LAST.pth: final epoch model")
    print(f"  metrics.json: training history")


if __name__ == "__main__":
    main()

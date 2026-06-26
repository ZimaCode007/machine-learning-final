"""
Pure T5 (text-only) training on ChartQA.
No visual features - input is question + flattened data table, output is answer.

Usage:
    .venv/Scripts/python.exe Models/VL-T5/train_local.py
"""
import sys
import os
import json
import re

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from transformers import T5ForConditionalGeneration, T5TokenizerFast, get_linear_schedule_with_warmup

proj_dir = Path(__file__).resolve().parent


class ChartQADataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_input_len=512, max_output_len=64):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

        df = pd.read_csv(os.path.join(data_dir, "data.csv"))
        self.inputs = df["Input"].values
        self.outputs = df["Output"].values if "Output" in df.columns else None
        self.question_ids = df["Question ID"].values

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = f"chartqa: {self.inputs[idx]}"
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        item = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "question_id": int(self.question_ids[idx]),
        }
        if self.outputs is not None:
            target = self.tokenizer(
                str(self.outputs[idx]),
                max_length=self.max_output_len,
                truncation=True,
                padding=False,
                return_tensors=None,
            )
            item["labels"] = target["input_ids"]
            item["answer"] = str(self.outputs[idx])
        return item


class CollateFn:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        return _collate_fn(batch, self.pad_token_id)


def _collate_fn(batch, pad_token_id):
    max_input = max(len(x["input_ids"]) for x in batch)
    max_label = max(len(x["labels"]) for x in batch) if "labels" in batch[0] else 0

    input_ids = torch.full((len(batch), max_input), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_input, dtype=torch.long)
    labels = torch.full((len(batch), max_label), -100, dtype=torch.long) if max_label > 0 else None

    question_ids = []
    answers = []

    for i, item in enumerate(batch):
        ids = item["input_ids"]
        input_ids[i, :len(ids)] = torch.tensor(ids)
        attention_mask[i, :len(ids)] = 1
        if labels is not None:
            lab = item["labels"]
            labels[i, :len(lab)] = torch.tensor(lab)
        question_ids.append(item["question_id"])
        if "answer" in item:
            answers.append(item["answer"])

    out = {"input_ids": input_ids, "attention_mask": attention_mask, "question_ids": question_ids}
    if labels is not None:
        out["labels"] = labels
        out["answers"] = answers
    return out


class Evaluator:
    def __init__(self, data_dir):
        df = pd.read_csv(os.path.join(data_dir, "data.csv"))
        self.qid2ans = dict(zip(df["Question ID"].values, df["Output"].values))

    def evaluate(self, qid2pred):
        correct = sum(
            1 for qid, pred in qid2pred.items()
            if str(pred).strip() == str(self.qid2ans[qid]).strip()
        )
        total = len(qid2pred)
        return {"overall": round(100 * correct / total, 2) if total > 0 else 0}


def train():
    # Config
    data_dir = proj_dir / "data"
    output_dir = proj_dir / "output_T5"
    backbone = "t5-base"
    epochs = 10
    batch_size = 8
    valid_batch_size = 4
    lr = 3e-4
    max_input_len = 512
    max_output_len = 64
    warmup_ratio = 0.05
    fp16 = True

    device = torch.device("cuda:0")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    # Tokenizer & Model
    tokenizer = T5TokenizerFast.from_pretrained(backbone)
    model = T5ForConditionalGeneration.from_pretrained(backbone)
    model.to(device)

    # Data
    train_ds = ChartQADataset(str(data_dir / "train"), tokenizer, max_input_len, max_output_len)
    valid_ds = ChartQADataset(str(data_dir / "valid"), tokenizer, max_input_len, max_output_len)
    test_ds = ChartQADataset(str(data_dir / "test"), tokenizer, max_input_len, max_output_len)

    collate = CollateFn(tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4,
                              collate_fn=collate, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=valid_batch_size, shuffle=False, num_workers=4,
                              collate_fn=collate, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=valid_batch_size, shuffle=False, num_workers=4,
                             collate_fn=collate, pin_memory=True)

    valid_evaluator = Evaluator(str(data_dir / "valid"))
    test_evaluator = Evaluator(str(data_dir / "test"))

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=lr, eps=1e-6)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda") if fp16 else None

    print(f"Data: train={len(train_ds)}, valid={len(valid_ds)}, test={len(test_ds)}")
    print(f"Batch: {batch_size}, Steps/epoch: {len(train_loader)}, Total: {total_steps}, Warmup: {warmup_steps}")
    print(f"Epochs: {epochs}, LR: {lr}, FP16: {fp16}")
    print(f"Output: {output_dir}\n")

    best_valid = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, ncols=120, desc=f"Epoch {epoch}")

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if fp16:
                with autocast():
                    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = out.loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            cur_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{cur_lr:.6f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}: avg loss = {avg_loss:.4f}")

        # Validation
        qid2pred = predict(model, valid_loader, tokenizer, device)
        score = valid_evaluator.evaluate(qid2pred)
        valid_acc = score["overall"]
        print(f"  Valid Acc: {valid_acc:.2f}%")

        if valid_acc > best_valid or epoch == 0:
            best_valid = valid_acc
            best_epoch = epoch
            save_model(model, tokenizer, output_dir, "BEST")
            print(f"  -> New best! Saved BEST")

        print(f"  Best: Epoch {best_epoch}, Acc {best_valid:.2f}%\n")

    save_model(model, tokenizer, output_dir, "LAST")

    # Final test with BEST model
    model = T5ForConditionalGeneration.from_pretrained(str(output_dir / "BEST"))
    model.to(device)
    qid2pred = predict(model, test_loader, tokenizer, device)
    test_score = test_evaluator.evaluate(qid2pred)
    print(f"\n{'='*50}")
    print(f"  TEST Accuracy: {test_score['overall']:.2f}%")
    print(f"{'='*50}")

    with open(output_dir / "results.json", "w") as f:
        json.dump({"best_epoch": best_epoch, "best_valid": best_valid,
                   "test_accuracy": test_score["overall"]}, f, indent=2)


@torch.no_grad()
def predict(model, loader, tokenizer, device):
    model.eval()
    qid2pred = {}
    for batch in tqdm(loader, ncols=120, desc="Predicting"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                 num_beams=3, max_length=64)
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for qid, pred in zip(batch["question_ids"], preds):
            qid2pred[qid] = pred
    return qid2pred


def save_model(model, tokenizer, output_dir, name):
    save_path = output_dir / name
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"  Model saved to {save_path}")


if __name__ == "__main__":
    cudnn.benchmark = True
    train()

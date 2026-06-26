"""
VisionTapas Classification Training Script
Per-epoch training trend visualization.
"""
import os
import sys
import time
import math
import json
import numpy as np
from typing import Dict
from datetime import timedelta

import torch
from transformers import (
    ViTImageProcessor, TapasTokenizer, ViTModel, TapasModel,
    Trainer, TrainingArguments, EvalPrediction, TrainerCallback,
)

from model.vision_tapas_for_classification import VisionTapasForClassification
from data.classification_dataset import VisionTapasForClassificationDataset
from model.config import VisionTapasConfig

# ================================================================
# Console helpers (ASCII-safe for Windows GBK)
# ================================================================

def banner(text, width=64, char="="):
    pad = (width - len(text) - 2) // 2
    print(f"\n{char * pad} {text} {char * pad}")

def kv(key, value, indent=2):
    print(f"{' ' * indent}{key:<30s}: {value}")

def fmt_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"

def fmt_mem(bytes_val):
    return f"{bytes_val / 1024**3:.2f} GB"

def make_bar(value, lo, hi, width=25, fill="#", empty="-"):
    if hi <= lo:
        frac = 0.5
    else:
        frac = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    n = int(width * frac)
    return fill * n + empty * (width - n)

def trend_arrow(prev, cur):
    if cur < prev - 1e-6:
        return "v (down)"
    elif cur > prev + 1e-6:
        return "^ (up)"
    return "= (flat)"


# ================================================================
# Epoch-level tracker & visualizer
# ================================================================

class EpochVisualizationCallback(TrainerCallback):
    def __init__(self, total_epochs, steps_per_epoch):
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch

        # per-step accumulation within current epoch
        self.cur_epoch_losses = []
        self.cur_epoch_lrs = []
        self.cur_epoch_grads = []

        # per-epoch history
        self.epoch_train_losses = []   # avg train loss per epoch
        self.epoch_eval_losses = []    # eval loss per epoch
        self.epoch_eval_accs = []      # eval accuracy per epoch
        self.epoch_lrs = []            # final lr per epoch
        self.epoch_grad_norms = []     # avg grad norm per epoch
        self.epoch_times = []          # seconds per epoch

        self.train_start = None
        self.epoch_start = None
        self.current_epoch_int = 0
        self.best_acc = 0.0
        self.best_loss = float('inf')
        self.best_epoch = 0

        # step-level for live progress
        self.step_in_epoch = 0
        self._pending_summary = None  # deferred epoch summary data

    # ---- lifecycle hooks ----

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start = time.time()
        self.epoch_start = time.time()
        banner("TRAINING STARTED")
        print(f"  {self.total_epochs} epochs x {self.steps_per_epoch} steps = {self.total_steps} total steps\n")

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = time.time()
        self.cur_epoch_losses = []
        self.cur_epoch_lrs = []
        self.cur_epoch_grads = []
        self.step_in_epoch = 0
        epoch_num = len(self.epoch_train_losses) + 1
        print(f"  --- Epoch {epoch_num}/{self.total_epochs} started ---")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        # accumulate step-level metrics
        if "loss" in logs:
            self.cur_epoch_losses.append(logs["loss"])
            self.cur_epoch_lrs.append(logs.get("learning_rate", 0))
            self.cur_epoch_grads.append(logs.get("grad_norm", 0))
            self.step_in_epoch = int(logs.get("epoch", 0) * self.steps_per_epoch) % self.steps_per_epoch
            if self.step_in_epoch == 0 and logs.get("epoch", 0) > 0:
                self.step_in_epoch = self.steps_per_epoch
            step = state.global_step
            bar = make_bar(self.step_in_epoch, 0, self.steps_per_epoch, width=20)
            print(f"\r    Step {step:>4d}/{self.total_steps}  |{bar}|  loss={logs['loss']:.4f}  lr={logs.get('learning_rate',0):.2e}", end="", flush=True)

        # when eval results arrive, attach them to pending summary and print
        if "eval_loss" in logs:
            self.epoch_eval_losses.append(logs["eval_loss"])
            self.epoch_eval_accs.append(logs.get("eval_accuracy", 0))
            if self._pending_summary is not None:
                self._print_epoch_summary(**self._pending_summary)
                self._pending_summary = None

    def on_epoch_end(self, args, state, control, **kwargs):
        print()  # newline after live progress
        epoch_num = len(self.epoch_train_losses) + 1
        epoch_time = time.time() - self.epoch_start
        elapsed = time.time() - self.train_start

        avg_loss = np.mean(self.cur_epoch_losses) if self.cur_epoch_losses else 0
        avg_grad = np.mean(self.cur_epoch_grads) if self.cur_epoch_grads else 0
        final_lr = self.cur_epoch_lrs[-1] if self.cur_epoch_lrs else 0
        min_loss = min(self.cur_epoch_losses) if self.cur_epoch_losses else 0
        max_loss = max(self.cur_epoch_losses) if self.cur_epoch_losses else 0

        self.epoch_train_losses.append(avg_loss)
        self.epoch_lrs.append(final_lr)
        self.epoch_grad_norms.append(avg_grad)
        self.epoch_times.append(epoch_time)

        # defer printing until eval results arrive via on_log
        self._pending_summary = dict(
            epoch_num=epoch_num, epoch_time=epoch_time, elapsed=elapsed,
            avg_loss=avg_loss, min_loss=min_loss, max_loss=max_loss,
            avg_grad=avg_grad, final_lr=final_lr,
            step_losses=list(self.cur_epoch_losses),
        )

    def _print_epoch_summary(self, epoch_num, epoch_time, elapsed,
                             avg_loss, min_loss, max_loss,
                             avg_grad, final_lr, step_losses):
        eval_loss = self.epoch_eval_losses[-1] if self.epoch_eval_losses else None
        eval_acc = self.epoch_eval_accs[-1] if self.epoch_eval_accs else None

        is_best = False
        if eval_acc is not None and eval_acc >= self.best_acc:
            if eval_acc > self.best_acc or (eval_loss is not None and eval_loss < self.best_loss):
                self.best_acc = eval_acc
                self.best_loss = eval_loss if eval_loss is not None else self.best_loss
                self.best_epoch = epoch_num
                is_best = True

        remaining = (self.total_epochs - epoch_num) * epoch_time
        print(f"\n  +{'=' * 62}+")
        print(f"  |  EPOCH {epoch_num}/{self.total_epochs} SUMMARY{' ' * 43}|")
        print(f"  +{'=' * 62}+")
        print(f"  |  {'Time':.<22s} {fmt_time(epoch_time):>10s}   (Total: {fmt_time(elapsed):>10s})    |")
        print(f"  |  {'ETA':.<22s} {fmt_time(remaining):>10s}{' ' * 29}|")
        print(f"  +{'-' * 62}+")

        # train loss
        loss_trend = ""
        if len(self.epoch_train_losses) > 1:
            loss_trend = trend_arrow(self.epoch_train_losses[-2], avg_loss)
        print(f"  |  {'Avg Train Loss':.<22s} {avg_loss:>10.4f}   {loss_trend:<28s}|")
        print(f"  |  {'  Min / Max':.<22s} {min_loss:>10.4f} / {max_loss:<10.4f}{' ' * 18}|")

        # loss bar chart across epoch steps
        if step_losses:
            bar = make_bar(avg_loss, max(step_losses), min(step_losses), width=30)
            print(f"  |  {'  Within-epoch':.<22s} |{bar}|{' ' * 7}|")

        # grad norm
        grad_trend = ""
        if len(self.epoch_grad_norms) > 1:
            grad_trend = trend_arrow(self.epoch_grad_norms[-2], avg_grad)
        print(f"  |  {'Avg Grad Norm':.<22s} {avg_grad:>10.2f}   {grad_trend:<28s}|")

        # learning rate
        print(f"  |  {'Learning Rate':.<22s} {final_lr:>10.2e}{' ' * 29}|")

        # eval
        if eval_loss is not None:
            print(f"  +{'-' * 62}+")
            best_marker = " << BEST" if is_best else ""
            eval_trend = ""
            if len(self.epoch_eval_losses) > 1:
                eval_trend = trend_arrow(self.epoch_eval_losses[-2], eval_loss)
            print(f"  |  {'Eval Loss':.<22s} {eval_loss:>10.4f}   {eval_trend:<28s}|")
            acc_trend = ""
            if len(self.epoch_eval_accs) > 1:
                acc_trend = trend_arrow(self.epoch_eval_accs[-2], eval_acc)
            print(f"  |  {'Eval Accuracy':.<22s} {eval_acc:>10.4f}   {acc_trend}{best_marker:<20s}|")

        # GPU
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated()
            total_mem = torch.cuda.get_device_properties(0).total_memory
            pct = alloc / total_mem * 100
            bar = make_bar(alloc, 0, total_mem, width=15)
            print(f"  +{'-' * 62}+")
            print(f"  |  {'GPU Memory':.<22s} |{bar}| {fmt_mem(alloc):>8s} / {fmt_mem(total_mem):<8s}  |")

        print(f"  +{'=' * 62}+")

        # ---- cross-epoch trend table ----
        if len(self.epoch_train_losses) > 1:
            print(f"\n  Cross-epoch trends:")
            print(f"  +-------+------------+------------+----------+----------+")
            print(f"  | Epoch | Train Loss |  Eval Loss | Eval Acc | Time     |")
            print(f"  +-------+------------+------------+----------+----------+")
            for i in range(len(self.epoch_train_losses)):
                e_loss = f"{self.epoch_eval_losses[i]:.4f}" if i < len(self.epoch_eval_losses) else "   --   "
                e_acc = f"{self.epoch_eval_accs[i]:.4f}" if i < len(self.epoch_eval_accs) else "  --  "
                marker = " *" if (i + 1) == self.best_epoch else "  "
                print(f"  | {i+1:>3d}{marker}| {self.epoch_train_losses[i]:>10.4f} | {e_loss:>10s} | {e_acc:>8s} | {fmt_time(self.epoch_times[i]):>8s} |")
            print(f"  +-------+------------+------------+----------+----------+")
            print(f"  (* = best checkpoint)")

            # ASCII trend lines
            print(f"\n  Train loss : [{make_bar(self.epoch_train_losses[-1], max(self.epoch_train_losses), min(self.epoch_train_losses), width=30)}]  {self.epoch_train_losses[0]:.3f} -> {self.epoch_train_losses[-1]:.3f}")
            if self.epoch_eval_accs:
                print(f"  Eval acc   : [{make_bar(self.epoch_eval_accs[-1], min(self.epoch_eval_accs), max(self.epoch_eval_accs), width=30)}]  {self.epoch_eval_accs[0]:.3f} -> {self.epoch_eval_accs[-1]:.3f}")

        print()

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.train_start

        banner("TRAINING COMPLETE")
        print()
        kv("Total time", fmt_time(total_time))
        kv("Total epochs", str(self.total_epochs))
        kv("Total steps", str(self.total_steps))
        kv("Final avg train loss", f"{self.epoch_train_losses[-1]:.4f}" if self.epoch_train_losses else "N/A")
        kv("Best eval loss", f"{self.best_loss:.4f}")
        kv("Best eval accuracy", f"{self.best_acc:.4f}")
        kv("Best epoch", str(self.best_epoch))
        if self.epoch_times:
            kv("Avg epoch time", fmt_time(np.mean(self.epoch_times)))
            kv("Avg samples/sec", f"{len(self.cur_epoch_losses) * 2 / np.mean(self.epoch_times) if self.epoch_times else 0:.1f}")
        print()


# ================================================================
# Metrics
# ================================================================

def compute_metrics(p: EvalPrediction) -> Dict:
    preds = np.argmax(p.predictions, axis=-1)
    return {'accuracy': float((preds == p.label_ids).mean())}


# ================================================================
# Main
# ================================================================

def main():
    base_dataset = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'ChartQA Dataset'
    )
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

    num_labels = 30
    batch_size = 6
    epochs = 20
    x_layers = 4
    lr = 3e-5

    # ---- 1. Environment ----
    banner("ENVIRONMENT")
    kv("PyTorch", torch.__version__)
    kv("CUDA available", str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        kv("GPU", torch.cuda.get_device_name(0))
        kv("GPU Memory", fmt_mem(torch.cuda.get_device_properties(0).total_memory))
    kv("Output dir", out_dir)

    # ---- 2. Tokenizer & Image Processor ----
    banner("LOADING TOKENIZER & IMAGE PROCESSOR")
    t0 = time.time()
    tokenizer = TapasTokenizer.from_pretrained('google/tapas-base-finetuned-wtq')
    feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    print(f"  Done in {time.time()-t0:.1f}s")

    # ---- 3. Model ----
    banner("BUILDING MODEL")
    print("  Architecture: VisionTapas for Classification")
    print(f"  |-- ViT encoder   : google/vit-base-patch16-224-in21k")
    print(f"  |-- TaPas encoder : google/tapas-base-finetuned-wtq")
    print(f"  |-- Cross-modal   : {x_layers} x LxmertXLayer")
    print(f"  `-- Classifier    : MLP -> {num_labels} classes")

    t0 = time.time()
    config = VisionTapasConfig(x_layers=x_layers, num_labels=num_labels)
    model = VisionTapasForClassification(config)
    model.visiontapas.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    model.visiontapas.tapas = TapasModel.from_pretrained('google/tapas-base-finetuned-wtq')
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    # Freeze pretrained encoders to prevent overfitting
    for param in model.visiontapas.vit.parameters():
        param.requires_grad = False
    for param in model.visiontapas.tapas.parameters():
        param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    vit_params = sum(p.numel() for p in model.visiontapas.vit.parameters())
    tapas_params = sum(p.numel() for p in model.visiontapas.tapas.parameters())
    cross_params = sum(p.numel() for p in model.visiontapas.x_layers.parameters())
    cls_params = sum(p.numel() for p in model.classifier.parameters())
    pooler_params = sum(p.numel() for p in model.visiontapas.pooler.parameters())

    print(f"\n  +---------------------+--------------+--------+---------+")
    print(f"  | Component           |   Parameters |  Share | Status  |")
    print(f"  +---------------------+--------------+--------+---------+")
    print(f"  | ViT encoder         | {vit_params:>12,} | {vit_params/total_params*100:5.1f}% | FROZEN  |")
    print(f"  | TaPas encoder       | {tapas_params:>12,} | {tapas_params/total_params*100:5.1f}% | FROZEN  |")
    print(f"  | Cross-modal layers  | {cross_params:>12,} | {cross_params/total_params*100:5.1f}% | TRAIN   |")
    print(f"  | Pooler              | {pooler_params:>12,} | {pooler_params/total_params*100:5.1f}% | TRAIN   |")
    print(f"  | Classification head | {cls_params:>12,} | {cls_params/total_params*100:5.1f}% | TRAIN   |")
    print(f"  +---------------------+--------------+--------+---------+")
    print(f"  | TOTAL               | {total_params:>12,} | 100.0% |         |")
    print(f"  | Trainable           | {trainable_params:>12,} | {trainable_params/total_params*100:5.1f}% |         |")
    print(f"  | Frozen              | {frozen_params:>12,} | {frozen_params/total_params*100:5.1f}% |         |")
    print(f"  +---------------------+--------------+--------+---------+")

    # ---- 4. Dataset ----
    banner("LOADING DATASETS")
    train_dataset = VisionTapasForClassificationDataset(
        qa_file_path='converted_data/train.json',
        tables_folder=os.path.join(base_dataset, 'train', 'tables'),
        images_folder=os.path.join(base_dataset, 'train', 'png'),
        tokenizer=tokenizer, feature_extractor=feature_extractor
    )
    val_dataset = VisionTapasForClassificationDataset(
        qa_file_path='converted_data/val.json',
        tables_folder=os.path.join(base_dataset, 'val', 'tables'),
        images_folder=os.path.join(base_dataset, 'val', 'png'),
        tokenizer=tokenizer, feature_extractor=feature_extractor
    )
    steps_per_epoch = math.ceil(len(train_dataset) / batch_size)
    total_steps = steps_per_epoch * epochs
    kv("Train samples", str(len(train_dataset)))
    kv("Val samples", str(len(val_dataset)))
    kv("Steps per epoch", str(steps_per_epoch))
    kv("Total steps", str(total_steps))

    # ---- 5. Training config ----
    early_stopping_patience = 5

    banner("TRAINING CONFIGURATION")
    kv("Epochs (max)", str(epochs))
    kv("Early stopping patience", str(early_stopping_patience))
    kv("Batch size", str(batch_size))
    kv("Learning rate", f"{lr:.1e}")
    kv("Warmup steps", "5")
    kv("Weight decay", "0.01")
    kv("FP16 mixed precision", "True")
    kv("Frozen encoders", "ViT + TaPas")
    kv("Eval at", "end of each epoch")

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=5,
        learning_rate=lr,
        weight_decay=0.01,
        dataloader_num_workers=0,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=max(1, steps_per_epoch // 3),
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=3,
        fp16=True,
        report_to="none",
        disable_tqdm=True,
    )

    from transformers import EarlyStoppingCallback
    vis_callback = EpochVisualizationCallback(epochs, steps_per_epoch)
    early_stop_callback = EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[vis_callback, early_stop_callback],
    )

    # ---- 6. Train ----
    trainer.train()

    # ---- 7. Save ----
    banner("SAVING BEST MODEL")
    best_model_dir = os.path.join(out_dir, "best_model")
    trainer.model.save_pretrained(best_model_dir)
    model_files = os.listdir(best_model_dir)
    kv("Save location", best_model_dir)
    kv("Files", ", ".join(model_files))
    total_size = sum(os.path.getsize(os.path.join(best_model_dir, f)) for f in model_files)
    kv("Total size on disk", fmt_mem(total_size))

    # ---- 8. Final eval ----
    banner("FINAL EVALUATION (best checkpoint)")
    metrics = trainer.evaluate()
    print()
    for k, v in metrics.items():
        if isinstance(v, float):
            kv(k, f"{v:.4f}")
        else:
            kv(k, str(v))

    banner("ALL DONE", char="*")
    print()


if __name__ == '__main__':
    import logging
    logging.disable(logging.INFO)
    main()

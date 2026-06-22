"""
VisionTapas Question Answering Training Script
Full data, frozen encoders, per-epoch visualization.
"""
import os
import sys
import time
import math
import pickle
import numpy as np
import argparse
from typing import Dict
from datetime import timedelta

import torch
from transformers import (
    ViTImageProcessor, TapasTokenizer, ViTModel, TapasModel,
    Trainer, TrainingArguments, EvalPrediction, TrainerCallback,
    EarlyStoppingCallback,
)

from model.vision_tapas_for_question_answering import VisionTapasForQuestionAnswering
from data.question_answering_dataset import VisionTapasForQuestionAnsweringDataset
from model.config import VisionTapasConfig

# ================================================================
# Console helpers (ASCII-safe)
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
# Epoch-level callback
# ================================================================

class EpochVisualizationCallback(TrainerCallback):
    def __init__(self, total_epochs, steps_per_epoch):
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        self.cur_epoch_losses = []
        self.cur_epoch_lrs = []
        self.cur_epoch_grads = []
        self.epoch_train_losses = []
        self.epoch_eval_losses = []
        self.epoch_times = []
        self.train_start = None
        self.epoch_start = None
        self.best_loss = float('inf')
        self.best_epoch = 0
        self._pending_summary = None

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
        epoch_num = len(self.epoch_train_losses) + 1
        print(f"  --- Epoch {epoch_num}/{self.total_epochs} started ---")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            self.cur_epoch_losses.append(logs["loss"])
            self.cur_epoch_lrs.append(logs.get("learning_rate", 0))
            self.cur_epoch_grads.append(logs.get("grad_norm", 0))
            step = state.global_step
            frac = (step % self.steps_per_epoch) / self.steps_per_epoch if self.steps_per_epoch > 0 else 0
            bar = make_bar(frac, 0, 1, width=20)
            print(f"\r    Step {step:>5d}/{self.total_steps}  |{bar}|  loss={logs['loss']:.4f}  lr={logs.get('learning_rate',0):.2e}", end="", flush=True)

        if "eval_loss" in logs:
            self.epoch_eval_losses.append(logs["eval_loss"])
            if self._pending_summary is not None:
                self._print_epoch_summary(**self._pending_summary)
                self._pending_summary = None

    def on_epoch_end(self, args, state, control, **kwargs):
        print()
        epoch_num = len(self.epoch_train_losses) + 1
        epoch_time = time.time() - self.epoch_start
        elapsed = time.time() - self.train_start

        avg_loss = np.mean(self.cur_epoch_losses) if self.cur_epoch_losses else 0
        avg_grad = np.mean(self.cur_epoch_grads) if self.cur_epoch_grads else 0
        final_lr = self.cur_epoch_lrs[-1] if self.cur_epoch_lrs else 0
        min_loss = min(self.cur_epoch_losses) if self.cur_epoch_losses else 0
        max_loss = max(self.cur_epoch_losses) if self.cur_epoch_losses else 0

        self.epoch_train_losses.append(avg_loss)
        self.epoch_times.append(epoch_time)

        self._pending_summary = dict(
            epoch_num=epoch_num, epoch_time=epoch_time, elapsed=elapsed,
            avg_loss=avg_loss, min_loss=min_loss, max_loss=max_loss,
            avg_grad=avg_grad, final_lr=final_lr,
        )

    def _print_epoch_summary(self, epoch_num, epoch_time, elapsed,
                             avg_loss, min_loss, max_loss, avg_grad, final_lr):
        eval_loss = self.epoch_eval_losses[-1] if self.epoch_eval_losses else None

        is_best = False
        if eval_loss is not None and eval_loss < self.best_loss:
            self.best_loss = eval_loss
            self.best_epoch = epoch_num
            is_best = True

        remaining = (self.total_epochs - epoch_num) * epoch_time
        print(f"\n  +{'=' * 62}+")
        print(f"  |  EPOCH {epoch_num}/{self.total_epochs} SUMMARY{' ' * 43}|")
        print(f"  +{'=' * 62}+")
        print(f"  |  {'Time':.<22s} {fmt_time(epoch_time):>10s}   (Total: {fmt_time(elapsed):>10s})    |")
        print(f"  |  {'ETA':.<22s} {fmt_time(remaining):>10s}{' ' * 29}|")
        print(f"  +{'-' * 62}+")

        loss_trend = ""
        if len(self.epoch_train_losses) > 1:
            loss_trend = trend_arrow(self.epoch_train_losses[-2], avg_loss)
        print(f"  |  {'Avg Train Loss':.<22s} {avg_loss:>10.4f}   {loss_trend:<28s}|")
        print(f"  |  {'  Min / Max':.<22s} {min_loss:>10.4f} / {max_loss:<10.4f}{' ' * 18}|")

        grad_trend = ""
        if len(self.epoch_times) > 1 and len(self.cur_epoch_grads) > 0:
            grad_trend = trend_arrow(avg_grad, avg_grad)
        print(f"  |  {'Avg Grad Norm':.<22s} {avg_grad:>10.2f}   {grad_trend:<28s}|")
        print(f"  |  {'Learning Rate':.<22s} {final_lr:>10.2e}{' ' * 29}|")

        if eval_loss is not None:
            print(f"  +{'-' * 62}+")
            best_marker = " << BEST" if is_best else ""
            eval_trend = ""
            if len(self.epoch_eval_losses) > 1:
                eval_trend = trend_arrow(self.epoch_eval_losses[-2], eval_loss)
            print(f"  |  {'Eval Loss':.<22s} {eval_loss:>10.4f}   {eval_trend}{best_marker:<20s}|")

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated()
            total_mem = torch.cuda.get_device_properties(0).total_memory
            pct = alloc / total_mem * 100
            bar = make_bar(alloc, 0, total_mem, width=15)
            print(f"  +{'-' * 62}+")
            print(f"  |  {'GPU Memory':.<22s} |{bar}| {fmt_mem(alloc):>8s} / {fmt_mem(total_mem):<8s}  |")

        print(f"  +{'=' * 62}+")

        if len(self.epoch_train_losses) > 1:
            print(f"\n  Cross-epoch trends:")
            print(f"  +-------+------------+------------+----------+")
            print(f"  | Epoch | Train Loss |  Eval Loss | Time     |")
            print(f"  +-------+------------+------------+----------+")
            for i in range(len(self.epoch_train_losses)):
                e_loss = f"{self.epoch_eval_losses[i]:.4f}" if i < len(self.epoch_eval_losses) else "   --   "
                marker = " *" if (i + 1) == self.best_epoch else "  "
                print(f"  | {i+1:>3d}{marker}| {self.epoch_train_losses[i]:>10.4f} | {e_loss:>10s} | {fmt_time(self.epoch_times[i]):>8s} |")
            print(f"  +-------+------------+------------+----------+")
            print(f"  (* = best checkpoint)")

            print(f"\n  Train loss : [{make_bar(self.epoch_train_losses[-1], max(self.epoch_train_losses), min(self.epoch_train_losses), width=30)}]  {self.epoch_train_losses[0]:.3f} -> {self.epoch_train_losses[-1]:.3f}")
            if len(self.epoch_eval_losses) > 1:
                print(f"  Eval loss  : [{make_bar(self.epoch_eval_losses[-1], max(self.epoch_eval_losses), min(self.epoch_eval_losses), width=30)}]  {self.epoch_eval_losses[0]:.3f} -> {self.epoch_eval_losses[-1]:.3f}")
        print()

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.train_start
        banner("TRAINING COMPLETE")
        print()
        kv("Total time", fmt_time(total_time))
        kv("Total epochs completed", str(len(self.epoch_train_losses)))
        kv("Total steps", str(self.total_steps))
        kv("Final avg train loss", f"{self.epoch_train_losses[-1]:.4f}" if self.epoch_train_losses else "N/A")
        kv("Best eval loss", f"{self.best_loss:.4f}")
        kv("Best epoch", str(self.best_epoch))
        if self.epoch_times:
            kv("Avg epoch time", fmt_time(np.mean(self.epoch_times)))
        print()


# ================================================================
# Main
# ================================================================

def main():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    base_dataset = os.path.join(base, 'ChartQA Dataset')
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_qa')
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'converted_data')

    batch_size = 8
    grad_accum = 2          # effective batch size = 8 * 2 = 16
    epochs_phase1 = 5       # phase 1: frozen encoders
    epochs_phase2 = 3       # phase 2: unfreeze top encoder layers
    lr_phase1 = 5e-5        # higher lr for randomly init cross-modal layers
    lr_phase2 = 2e-6        # very low lr for fine-tuning encoder layers
    x_layers = 4
    early_stopping_patience = 3

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

    # ---- 3. Fixed vocab & aggregation labels ----
    banner("LOADING VOCABULARY")
    fixed_vocab = pickle.load(open(os.path.join(data_dir, 'fixed_vocab.pkl'), 'rb'))
    aggregation_labels = {'0': 'NONE', '1': 'SUM', '2': 'AVERAGE', '3': 'COUNT', '4': 'Diff', '5': 'Ratio'}
    init_length = len(aggregation_labels)
    for i in range(len(fixed_vocab)):
        aggregation_labels[str(i + init_length)] = str(fixed_vocab[i])
    classes_mappings = dict((v, int(k)) for k, v in aggregation_labels.items())
    kv("Base operations", "NONE, SUM, AVERAGE, COUNT, Diff, Ratio")
    kv("Fixed vocab answers", str(len(fixed_vocab)))
    kv("Total aggregation labels", str(len(aggregation_labels)))

    # ---- 4. Model ----
    banner("BUILDING MODEL")
    print("  Architecture: VisionTapas for Question Answering")
    print(f"  |-- ViT encoder   : google/vit-base-patch16-224-in21k")
    print(f"  |-- TaPas encoder : google/tapas-base-finetuned-wtq")
    print(f"  |-- Cross-modal   : {x_layers} x LxmertXLayer")
    print(f"  `-- QA heads      : cell selection + aggregation ({len(aggregation_labels)} ops)")

    t0 = time.time()
    config = VisionTapasConfig(x_layers=x_layers)
    args_mock = argparse.Namespace(
        answer_loss_cutoff=50,
        select_one_column=True,
        cell_selection_preference=0.001,
    )
    model = VisionTapasForQuestionAnswering(config, aggregation_labels=aggregation_labels, args=args_mock)
    model.visiontapas.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    model.visiontapas.tapas = TapasModel.from_pretrained('google/tapas-base-finetuned-wtq')

    model.visiontapas.tapas.config.answer_loss_cutoff = 50
    model.visiontapas.tapas.config.select_one_column = True
    model.visiontapas.tapas.config.cell_selection_preference = 0.001
    model.visiontapas.tapas.config.num_aggregation_labels = len(aggregation_labels)
    model.visiontapas.tapas.config.aggregation_labels = aggregation_labels
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    # Freeze encoders
    for param in model.visiontapas.vit.parameters():
        param.requires_grad = False
    for param in model.visiontapas.tapas.parameters():
        param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\n  +---------------------+--------------+--------+---------+")
    print(f"  | Component           |   Parameters |  Share | Status  |")
    print(f"  +---------------------+--------------+--------+---------+")
    vit_p = sum(p.numel() for p in model.visiontapas.vit.parameters())
    tapas_p = sum(p.numel() for p in model.visiontapas.tapas.parameters())
    cross_p = sum(p.numel() for p in model.visiontapas.x_layers.parameters())
    print(f"  | ViT encoder         | {vit_p:>12,} | {vit_p/total_params*100:5.1f}% | FROZEN  |")
    print(f"  | TaPas encoder       | {tapas_p:>12,} | {tapas_p/total_params*100:5.1f}% | FROZEN  |")
    print(f"  | Cross-modal + heads | {cross_p:>12,} | {cross_p/total_params*100:5.1f}% | TRAIN   |")
    print(f"  +---------------------+--------------+--------+---------+")
    print(f"  | TOTAL               | {total_params:>12,} |        |         |")
    print(f"  | Trainable           | {trainable_params:>12,} | {trainable_params/total_params*100:5.1f}% |         |")
    print(f"  | Frozen              | {frozen_params:>12,} | {frozen_params/total_params*100:5.1f}% |         |")
    print(f"  +---------------------+--------------+--------+---------+")

    # ---- 5. Dataset ----
    banner("LOADING DATASETS")
    t0 = time.time()
    train_dataset = VisionTapasForQuestionAnsweringDataset(
        os.path.join(data_dir, 'train_qa.json'),
        os.path.join(base_dataset, 'train', 'tables'),
        os.path.join(base_dataset, 'train', 'png'),
        tokenizer, feature_extractor, classes_mappings
    )
    val_dataset = VisionTapasForQuestionAnsweringDataset(
        os.path.join(data_dir, 'val_qa.json'),
        os.path.join(base_dataset, 'val', 'tables'),
        os.path.join(base_dataset, 'val', 'png'),
        tokenizer, feature_extractor, classes_mappings
    )
    print(f"  Loaded in {time.time()-t0:.1f}s")
    steps_per_epoch = math.ceil(len(train_dataset) / batch_size)
    total_steps = steps_per_epoch * (epochs_phase1 + epochs_phase2)
    kv("Train samples", str(len(train_dataset)))
    kv("Val samples", str(len(val_dataset)))
    kv("Steps per epoch", str(steps_per_epoch))
    kv("Total steps (max)", str(total_steps))

    # ---- 6. Custom Trainer ----
    class QATrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            return (loss, outputs) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=self.args.fp16):
                    outputs = model(**inputs)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            return (loss, None, None)

    # ============================================================
    # PHASE 1: Train cross-modal layers + heads (encoders frozen)
    # ============================================================
    banner("PHASE 1: TRAIN CROSS-MODAL LAYERS")
    kv("Epochs", str(epochs_phase1))
    kv("Learning rate", f"{lr_phase1:.1e}")
    kv("Effective batch size", f"{batch_size} x {grad_accum} = {batch_size * grad_accum}")
    kv("Warmup steps", "100")
    kv("LR schedule", "cosine")
    kv("Frozen", "ViT + TaPas")

    eff_steps_per_epoch = math.ceil(steps_per_epoch / grad_accum)
    total_steps_p1 = eff_steps_per_epoch * epochs_phase1

    training_args_p1 = TrainingArguments(
        output_dir=os.path.join(out_dir, "phase1"),
        num_train_epochs=epochs_phase1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=100,
        learning_rate=lr_phase1,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        dataloader_num_workers=0,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=max(1, eff_steps_per_epoch // 5),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        fp16=True,
        report_to="none",
        disable_tqdm=True,
    )

    vis_cb_p1 = EpochVisualizationCallback(epochs_phase1, steps_per_epoch)
    early_cb_p1 = EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)

    trainer_p1 = QATrainer(
        model=model,
        args=training_args_p1,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[vis_cb_p1, early_cb_p1],
    )
    trainer_p1.args.remove_unused_columns = False
    trainer_p1.train()

    # Save phase 1 best
    p1_dir = os.path.join(out_dir, "phase1_best")
    trainer_p1.model.save_pretrained(p1_dir)
    banner("PHASE 1 COMPLETE")
    kv("Best eval loss", f"{vis_cb_p1.best_loss:.4f}")
    kv("Best epoch", str(vis_cb_p1.best_epoch))

    # ============================================================
    # PHASE 2: Unfreeze top encoder layers + fine-tune with low lr
    # ============================================================
    banner("PHASE 2: FINE-TUNE WITH UNFROZEN TOP LAYERS")

    # Unfreeze top 2 ViT layers and top 2 TaPas layers
    vit_layers = list(model.visiontapas.vit.encoder.layer) if hasattr(model.visiontapas.vit, 'encoder') else list(model.visiontapas.vit.layers) if hasattr(model.visiontapas.vit, 'layers') else []
    tapas_layers = list(model.visiontapas.tapas.encoder.layer)

    unfreeze_top_n = 2
    for layer in vit_layers[-unfreeze_top_n:]:
        for param in layer.parameters():
            param.requires_grad = True
    for layer in tapas_layers[-unfreeze_top_n:]:
        for param in layer.parameters():
            param.requires_grad = True

    trainable_p2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    kv("Unfrozen ViT layers", f"top {unfreeze_top_n} of {len(vit_layers)}")
    kv("Unfrozen TaPas layers", f"top {unfreeze_top_n} of {len(tapas_layers)}")
    kv("Trainable params now", f"{trainable_p2:,} ({trainable_p2/total_params*100:.1f}%)")
    kv("Learning rate", f"{lr_phase2:.1e}")
    kv("Epochs", str(epochs_phase2))

    total_steps_p2 = eff_steps_per_epoch * epochs_phase2

    training_args_p2 = TrainingArguments(
        output_dir=os.path.join(out_dir, "phase2"),
        num_train_epochs=epochs_phase2,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=50,
        learning_rate=lr_phase2,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        dataloader_num_workers=0,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=max(1, eff_steps_per_epoch // 5),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        fp16=True,
        report_to="none",
        disable_tqdm=True,
    )

    vis_cb_p2 = EpochVisualizationCallback(epochs_phase2, steps_per_epoch)
    early_cb_p2 = EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)

    trainer_p2 = QATrainer(
        model=model,
        args=training_args_p2,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[vis_cb_p2, early_cb_p2],
    )
    trainer_p2.args.remove_unused_columns = False
    trainer_p2.train()

    # ---- 8. Save final ----
    banner("SAVING FINAL MODEL")
    best_model_dir = os.path.join(out_dir, "best_model")
    trainer_p2.model.save_pretrained(best_model_dir)
    model_files = os.listdir(best_model_dir)
    kv("Save location", best_model_dir)
    kv("Files", ", ".join(model_files))
    total_size = sum(os.path.getsize(os.path.join(best_model_dir, f)) for f in model_files)
    kv("Total size on disk", fmt_mem(total_size))

    # ---- 9. Final eval ----
    banner("FINAL EVALUATION")
    metrics = trainer_p2.evaluate()
    print()
    for k, v in metrics.items():
        if isinstance(v, float):
            kv(k, f"{v:.4f}")
        else:
            kv(k, str(v))

    banner("TRAINING SUMMARY")
    kv("Phase 1 best eval loss", f"{vis_cb_p1.best_loss:.4f} (epoch {vis_cb_p1.best_epoch})")
    kv("Phase 2 best eval loss", f"{vis_cb_p2.best_loss:.4f} (epoch {vis_cb_p2.best_epoch})")
    total_time = sum(vis_cb_p1.epoch_times) + sum(vis_cb_p2.epoch_times)
    kv("Total training time", fmt_time(total_time))

    banner("ALL DONE", char="*")
    print()


if __name__ == '__main__':
    import logging
    logging.disable(logging.INFO)
    main()

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
        self.epoch_train_losses = []
        self.epoch_eval_losses = []
        self.epoch_times = []
        self.train_start = None
        self.epoch_start = None
        self.best_loss = float('inf')
        self.best_epoch = 0
        self._pending = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start = time.time()
        self.epoch_start = time.time()
        print(f"\n  Epoch  |{'Progress':^24s}| Train Loss | Eval Loss  | Time")
        print(f"  -------+{'-'*24}+------------+------------+--------")

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = time.time()
        self.cur_epoch_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            self.cur_epoch_losses.append(logs["loss"])
            epoch_num = len(self.epoch_train_losses) + 1
            step = state.global_step
            frac = (step % self.steps_per_epoch) / self.steps_per_epoch if self.steps_per_epoch > 0 else 0
            bar = make_bar(frac, 0, 1, width=20)
            print(f"\r  {epoch_num:>3d}/{self.total_epochs}  |{bar}|    {logs['loss']:.4f}  |            |", end="", flush=True)

        if "eval_loss" in logs:
            self.epoch_eval_losses.append(logs["eval_loss"])
            if self._pending is not None:
                self._print_row(**self._pending)
                self._pending = None

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_num = len(self.epoch_train_losses) + 1
        epoch_time = time.time() - self.epoch_start
        avg_loss = np.mean(self.cur_epoch_losses) if self.cur_epoch_losses else 0
        self.epoch_train_losses.append(avg_loss)
        self.epoch_times.append(epoch_time)
        self._pending = dict(epoch_num=epoch_num, avg_loss=avg_loss, epoch_time=epoch_time)

    def _print_row(self, epoch_num, avg_loss, epoch_time):
        eval_loss = self.epoch_eval_losses[-1] if self.epoch_eval_losses else None
        is_best = False
        if eval_loss is not None and eval_loss < self.best_loss:
            self.best_loss = eval_loss
            self.best_epoch = epoch_num
            is_best = True
        bar = make_bar(1, 0, 1, width=20)
        e_str = f"{eval_loss:.4f}" if eval_loss is not None else "   --   "
        best = " *" if is_best else "  "
        print(f"\r  {epoch_num:>3d}/{self.total_epochs}  |{bar}|    {avg_loss:.4f}  |  {e_str}{best}| {fmt_time(epoch_time):>6s}")

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.train_start
        print(f"  -------+{'-'*24}+------------+------------+--------")
        print(f"  Best: epoch {self.best_epoch}, eval_loss = {self.best_loss:.4f}, total = {fmt_time(total_time)}\n")


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

    # ---- Setup ----
    print("\n  Loading model and data...")
    tokenizer = TapasTokenizer.from_pretrained('google/tapas-base-finetuned-wtq')
    feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

    fixed_vocab = pickle.load(open(os.path.join(data_dir, 'fixed_vocab.pkl'), 'rb'))
    aggregation_labels = {'0': 'NONE', '1': 'SUM', '2': 'AVERAGE', '3': 'COUNT', '4': 'Diff', '5': 'Ratio'}
    init_length = len(aggregation_labels)
    for i in range(len(fixed_vocab)):
        aggregation_labels[str(i + init_length)] = str(fixed_vocab[i])
    classes_mappings = dict((v, int(k)) for k, v in aggregation_labels.items())

    config = VisionTapasConfig(x_layers=x_layers)
    args_mock = argparse.Namespace(answer_loss_cutoff=50, select_one_column=True, cell_selection_preference=0.001)
    model = VisionTapasForQuestionAnswering(config, aggregation_labels=aggregation_labels, args=args_mock)
    model.visiontapas.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    model.visiontapas.tapas = TapasModel.from_pretrained('google/tapas-base-finetuned-wtq')
    model.visiontapas.tapas.config.answer_loss_cutoff = 50
    model.visiontapas.tapas.config.select_one_column = True
    model.visiontapas.tapas.config.cell_selection_preference = 0.001
    model.visiontapas.tapas.config.num_aggregation_labels = len(aggregation_labels)
    model.visiontapas.tapas.config.aggregation_labels = aggregation_labels

    for param in model.visiontapas.vit.parameters():
        param.requires_grad = False
    for param in model.visiontapas.tapas.parameters():
        param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
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
    steps_per_epoch = math.ceil(len(train_dataset) / batch_size)
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Params: {trainable_params:,}/{total_params:,} trainable, bs={batch_size}x{grad_accum}")

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
    print(f"  Phase 1: {epochs_phase1} epochs, lr={lr_phase1:.0e}, encoders frozen")

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

    p1_dir = os.path.join(out_dir, "phase1_best")
    trainer_p1.model.save_pretrained(p1_dir)

    # ---- PHASE 2: Unfreeze top encoder layers ----
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
    banner(f"PHASE 2: {epochs_phase2} epochs, lr={lr_phase2:.0e}, top-{unfreeze_top_n} layers unfrozen ({trainable_p2:,} params)")

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

    # ---- Save & Summary ----
    best_model_dir = os.path.join(out_dir, "best_model")
    trainer_p2.model.save_pretrained(best_model_dir)

    total_time = sum(vis_cb_p1.epoch_times) + sum(vis_cb_p2.epoch_times)
    print(f"\n  Done. Phase1 best={vis_cb_p1.best_loss:.4f}(ep{vis_cb_p1.best_epoch}), Phase2 best={vis_cb_p2.best_loss:.4f}(ep{vis_cb_p2.best_epoch}), total={fmt_time(total_time)}")
    print(f"  Model saved to {best_model_dir}\n")


if __name__ == '__main__':
    import logging, warnings
    logging.disable(logging.INFO)
    warnings.filterwarnings("ignore")
    main()

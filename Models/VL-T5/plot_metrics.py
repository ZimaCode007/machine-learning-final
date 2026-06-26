"""
Plot training metrics from metrics.json.
Can be run during training to see progress so far.

Usage:
    python plot_metrics.py --metrics output/metrics.json --save output/training_curves.png
"""
import argparse
import json
import matplotlib.pyplot as plt
import os


def plot(metrics_path, save_path=None):
    with open(metrics_path) as f:
        metrics = json.load(f)

    if not metrics:
        print("No metrics recorded yet.")
        return

    epochs = [m['epoch'] for m in metrics]
    train_loss = [m['train_loss'] for m in metrics]
    val_acc = [m['val_accuracy'] for m in metrics]
    best_acc = [m['best_val_accuracy'] for m in metrics]
    lrs = [m['lr'] for m in metrics]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('VL-T5 Training Metrics', fontsize=16, fontweight='bold')

    # Train Loss
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, 'b-o', linewidth=2, markersize=6)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    if len(train_loss) > 1:
        ax.annotate(f'{train_loss[-1]:.4f}', xy=(epochs[-1], train_loss[-1]),
                     fontsize=10, fontweight='bold', color='blue',
                     xytext=(5, 5), textcoords='offset points')

    # Validation Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, val_acc, 'g-o', linewidth=2, markersize=6, label='Val Accuracy')
    ax.plot(epochs, best_acc, 'r--', linewidth=1.5, alpha=0.7, label='Best So Far')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    if len(val_acc) > 1:
        ax.annotate(f'{val_acc[-1]:.1f}%', xy=(epochs[-1], val_acc[-1]),
                     fontsize=10, fontweight='bold', color='green',
                     xytext=(5, 5), textcoords='offset points')

    # Learning Rate
    ax = axes[1, 0]
    ax.plot(epochs, lrs, 'm-o', linewidth=2, markersize=6)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    # Summary Table
    ax = axes[1, 1]
    ax.axis('off')
    best_epoch_idx = val_acc.index(max(val_acc))
    table_data = [
        ['Epochs Completed', f'{len(metrics)}'],
        ['Final Train Loss', f'{train_loss[-1]:.4f}'],
        ['Final Val Accuracy', f'{val_acc[-1]:.2f}%'],
        ['Best Val Accuracy', f'{max(val_acc):.2f}%'],
        ['Best Epoch', f'{epochs[best_epoch_idx]}'],
        ['Loss Δ (first→last)', f'{train_loss[0] - train_loss[-1]:.4f}'],
    ]
    table = ax.table(cellText=table_data, colLabels=['Metric', 'Value'],
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    for i in range(len(table_data) + 1):
        for j in range(2):
            cell = table[i, j]
            if i == 0:
                cell.set_facecolor('#4472C4')
                cell.set_text_props(color='white', fontweight='bold')
            elif i % 2 == 0:
                cell.set_facecolor('#D9E2F3')

    ax.set_title('Summary', pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=str, default="output/metrics.json")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()
    plot(args.metrics, args.save)

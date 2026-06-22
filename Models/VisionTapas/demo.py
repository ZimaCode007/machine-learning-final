"""
VisionTapas QA Demo
1. Evaluate accuracy on 300 random test samples
2. Visualize 5 random predictions with chart images
"""
import json, os, pickle, argparse, random, warnings, sys
warnings.filterwarnings('ignore')
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from safetensors.torch import load_file
from transformers import TapasTokenizer, ViTImageProcessor, ViTModel, TapasModel
from transformers.models.lxmert.modeling_lxmert import LxmertXLayer
from model.vision_tapas_for_question_answering import VisionTapasForQuestionAnswering
from model.config import VisionTapasConfig
from data.tapas_utils import convert_logits_to_predictions, get_final_answer
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# ================================================================
# Model loading
# ================================================================

def load_model(ckpt_path):
    fv = pickle.load(open('converted_data/fixed_vocab.pkl', 'rb'))
    agg = {'0': 'NONE', '1': 'SUM', '2': 'AVERAGE', '3': 'COUNT', '4': 'Diff', '5': 'Ratio'}
    for i in range(len(fv)):
        agg[str(i + 6)] = str(fv[i])

    tok = TapasTokenizer.from_pretrained('google/tapas-base-finetuned-wtq')
    feat = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

    config = VisionTapasConfig(x_layers=4)
    args_m = argparse.Namespace(answer_loss_cutoff=50, select_one_column=True, cell_selection_preference=0.001)
    model = VisionTapasForQuestionAnswering(config, aggregation_labels=agg, args=args_m)
    model.visiontapas.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    model.visiontapas.tapas = TapasModel.from_pretrained('google/tapas-base-finetuned-wtq')
    model.visiontapas.tapas.config.num_aggregation_labels = len(agg)
    model.visiontapas.tapas.config.aggregation_labels = agg

    state = load_file(os.path.join(ckpt_path, 'model.safetensors'))
    model.load_state_dict(state, strict=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    return model, tok, feat, fv, agg, device


# ================================================================
# Single prediction
# ================================================================

OPS = {0: 'SELECT', 1: 'SUM', 2: 'AVERAGE', 3: 'COUNT', 4: 'Diff', 5: 'Ratio'}

def predict(model, tok, feat, fv, imgname, question, tables_folder, images_folder, device='cpu'):
    df = pd.read_csv(os.path.join(tables_folder, imgname + '.csv'), encoding='utf8')
    df.columns = [str(c) if not (isinstance(c, float) and c != c) else f'col_{i}' for i, c in enumerate(df.columns)]
    df = df.fillna('').astype(object)
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip('%')

    encoding = tok(table=df, queries=[question], padding='max_length', truncation=True, return_tensors='pt')
    img = Image.open(os.path.join(images_folder, imgname + '.png')).convert('RGB')
    vis = feat(images=img, return_tensors='pt')
    encoding['pixel_values'] = vis['pixel_values']

    encoding = {k: v.to(device) for k, v in encoding.items()}
    with torch.no_grad():
        outputs = model(return_dict=True, **encoding)

    logits = outputs.logits.cpu().numpy()
    logits_agg = outputs.aggregation_logits.cpu()
    data_np = {k: v.cpu().numpy() for k, v in encoding.items() if k in ['input_ids', 'token_type_ids', 'attention_mask']}
    (pred_coords, agg_idx), _ = convert_logits_to_predictions(data_np, logits, logits_agg=logits_agg, cell_classification_threshold=0.5)

    op_idx = agg_idx[0] if agg_idx else 0
    op_name = f'VOCAB({fv[op_idx - 6]})' if op_idx > 5 else OPS.get(op_idx, f'OP_{op_idx}')

    if op_idx > 5:
        pred_answer = fv[op_idx - 6]
    elif len(pred_coords) > 0 and len(pred_coords[0]) > 0:
        pred_answer = str(get_final_answer(df, pred_coords[0], op_idx, fv))
    else:
        pred_answer = ''

    cells = pred_coords[0] if pred_coords and len(pred_coords[0]) > 0 else []
    return pred_answer, op_name, cells, img


# ================================================================
# Accuracy check (relaxed match)
# ================================================================

def is_relaxed_match(gold, pred):
    gold_s = gold.strip().lower()
    pred_s = pred.strip().lower()
    if gold_s == pred_s:
        return True, True
    # Numeric comparison: "2" vs "2.0" should be exact match
    try:
        g = float(gold.replace(',', ''))
        p = float(str(pred).replace(',', ''))
        if g == p:
            return True, True
        if abs(g) < 1e-9:
            relaxed = abs(p) < 1e-9
        else:
            relaxed = abs(g - p) / abs(g) <= 0.05
        return False, relaxed
    except (ValueError, TypeError):
        return False, False


# ================================================================
# Part 1: Evaluate 300 samples
# ================================================================

def evaluate_accuracy(model, tok, feat, fv, val_data, tables_folder, images_folder, device, n=200):
    indices = random.sample(range(len(val_data)), min(n, len(val_data)))

    exact, relaxed, total, errors = 0, 0, 0, 0

    for idx in tqdm(indices, desc="  Evaluating", ncols=70):
        item = val_data[idx]
        imgname = item['image_index']
        question = item['question']
        gold = str(item['answer'][1][0]).strip()

        try:
            pred, _, _, _ = predict(model, tok, feat, fv, imgname, question, tables_folder, images_folder, device)
            is_exact, is_relaxed = is_relaxed_match(gold, pred)
            if is_exact:
                exact += 1
            if is_relaxed:
                relaxed += 1
            total += 1
        except:
            errors += 1

    print()
    print("  Results:")
    print(f"  +------------------------+---------+")
    print(f"  | Metric                 | Value   |")
    print(f"  +------------------------+---------+")
    print(f"  | Samples evaluated      | {total:>7d} |")
    print(f"  | Exact match accuracy   | {exact/total*100:>6.2f}% |")
    print(f"  | Relaxed accuracy (5%)  | {relaxed/total*100:>6.2f}% |")
    print(f"  | Errors (skipped)       | {errors:>7d} |")
    print(f"  +------------------------+---------+")
    print()
    return exact / total, relaxed / total


# ================================================================
# Part 2: Visualize 5 samples
# ================================================================

def visualize_samples(model, tok, feat, fv, val_data, tables_folder, images_folder, device, out_dir, n=5, label=""):
    print("=" * 60)
    print(f"  PART 2: Visualize {n} Random Predictions ({label})")
    print("=" * 60)

    os.makedirs(out_dir, exist_ok=True)

    indices = random.sample(range(len(val_data)), min(n * 4, len(val_data)))

    results = []
    for idx in indices:
        if len(results) >= n:
            break
        item = val_data[idx]
        imgname = item['image_index']
        question = item['question']
        gold = str(item['answer'][1][0]).strip()
        try:
            pred, op_name, cells, img = predict(model, tok, feat, fv, imgname, question, tables_folder, images_folder, device)
            results.append((imgname, question, gold, pred, op_name, cells, img))
        except:
            continue

    for i, (imgname, question, gold, pred, op_name, cells, img) in enumerate(results):
        is_exact, is_relaxed = is_relaxed_match(gold, pred)

        # Console output
        if is_exact:
            tag = "EXACT MATCH"
        elif is_relaxed:
            tag = "RELAXED MATCH"
        else:
            tag = "INCORRECT"
        print(f"\n  [{i+1}] {imgname}")
        print(f"      Question : {question}")
        print(f"      Gold     : {gold}")
        print(f"      Predicted: {pred} ({op_name})")
        print(f"      Result   : {tag}")

        # Generate visualization
        fig = plt.figure(figsize=(16, 7))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.05)

        # Left: chart image
        ax_img = fig.add_subplot(gs[0])
        ax_img.imshow(img)
        ax_img.axis('off')

        # Right: QA panel
        ax_info = fig.add_subplot(gs[1])
        ax_info.axis('off')
        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(0, 1)

        y = 0.95
        gap = 0.055

        def draw_text(label, value, color='black', bold_value=False):
            nonlocal y
            ax_info.text(0.05, y, label, fontsize=12, fontweight='bold', va='top', transform=ax_info.transAxes)
            y -= gap
            weight = 'bold' if bold_value else 'normal'
            # word wrap long text
            if len(value) > 55:
                lines = [value[j:j+55] for j in range(0, len(value), 55)]
                for line in lines:
                    ax_info.text(0.05, y, line, fontsize=12, color=color, fontweight=weight, va='top', transform=ax_info.transAxes)
                    y -= gap
            else:
                ax_info.text(0.05, y, value, fontsize=12, color=color, fontweight=weight, va='top', transform=ax_info.transAxes)
                y -= gap
            y -= 0.01

        draw_text("Question:", question)
        draw_text("Gold Answer:", gold, color='#1565C0', bold_value=True)
        draw_text("Predicted:", pred if pred else '[no answer]',
                  color='#2E7D32' if is_exact else '#E65100' if is_relaxed else '#C62828',
                  bold_value=True)
        draw_text("Operation:", op_name, color='#616161')

        if cells:
            cells_str = ', '.join([f'({r},{c})' for r, c in cells[:6]])
            if len(cells) > 6:
                cells_str += f' +{len(cells)-6} more'
            draw_text("Selected Cells:", cells_str, color='#616161')

        # Result badge
        y -= 0.02
        if is_exact:
            badge_color, badge_text = '#2E7D32', 'EXACT MATCH'
        elif is_relaxed:
            badge_color, badge_text = '#E65100', 'RELAXED MATCH (within 5%)'
        else:
            badge_color, badge_text = '#C62828', 'INCORRECT'

        ax_info.add_patch(plt.Rectangle((0.03, y - 0.04), 0.94, 0.06,
                          facecolor=badge_color, alpha=0.15, transform=ax_info.transAxes,
                          linewidth=2, edgecolor=badge_color))
        ax_info.text(0.5, y - 0.01, badge_text, fontsize=14, fontweight='bold',
                     color=badge_color, ha='center', va='top', transform=ax_info.transAxes)

        save_path = os.path.join(out_dir, f'{label}_{i+1}_{imgname}.png')
        plt.savefig(save_path, dpi=130, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"      Saved: {save_path}")

    print(f"\n  All visualizations saved to {out_dir}/")


# ================================================================
# Main
# ================================================================

def main():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    base_dataset = os.path.join(base, 'ChartQA Dataset')
    tables_folder = os.path.join(base_dataset, 'val', 'tables')
    images_folder = os.path.join(base_dataset, 'val', 'png')
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demo_output')
    ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_qa', 'phase1_best')

    import shutil
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demo_output')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    print()
    print("  VisionTapas QA Demo")
    print("  Model: VisionTapas (ViT + TaPas + 4x Cross-Modal)")
    print(f"  Checkpoint: {ckpt}")
    print()

    print("  Loading model...")
    model, tok, feat, fv, agg, device = load_model(ckpt)
    print(f"  Model loaded on {device}.\n")

    val_data = json.load(open('converted_data/val_qa.json', encoding='utf-8'))

    # Split into H (human) and M (augmented/machine)
    aug_qs = set((x['imgname'].replace('.png', ''), x['query'])
                 for x in json.load(open(os.path.join(base_dataset, 'val', 'val_augmented.json'), encoding='utf-8')))
    hum_qs = set((x['imgname'].replace('.png', ''), x['query'])
                 for x in json.load(open(os.path.join(base_dataset, 'val', 'val_human.json'), encoding='utf-8')))

    val_h = [x for x in val_data if (x['image_index'], x['question']) in hum_qs]
    val_m = [x for x in val_data if (x['image_index'], x['question']) in aug_qs]
    print(f"  Test data: {len(val_data)} total ({len(val_h)} Human + {len(val_m)} Augmented)\n")

    # Part 1: Accuracy on H and M separately
    print("=" * 60)
    print("  PART 1: Accuracy Evaluation (200 Human + 200 Augmented)")
    print("=" * 60)

    h_exact, h_relaxed = evaluate_accuracy(model, tok, feat, fv, val_h, tables_folder, images_folder, device, n=200)
    m_exact, m_relaxed = evaluate_accuracy(model, tok, feat, fv, val_m, tables_folder, images_folder, device, n=200)

    overall_exact = (h_exact + m_exact) / 2
    overall_relaxed = (h_relaxed + m_relaxed) / 2

    print("  Combined Results:")
    print(f"  +------------------+---------+---------+---------+")
    print(f"  |                  |  Exact  | Relaxed |  Paper  |")
    print(f"  +------------------+---------+---------+---------+")
    print(f"  | ChartQA-H (human)| {h_exact*100:>6.2f}% | {h_relaxed*100:>6.2f}% |  29.60% |")
    print(f"  | ChartQA-M (augm) | {m_exact*100:>6.2f}% | {m_relaxed*100:>6.2f}% |  61.44% |")
    print(f"  | Overall          | {overall_exact*100:>6.2f}% | {overall_relaxed*100:>6.2f}% |  45.52% |")
    print(f"  +------------------+---------+---------+---------+")
    print()

    # Part 2: Visualize 5 from H + 5 from M
    visualize_samples(model, tok, feat, fv, val_h, tables_folder, images_folder, device, out_dir, n=5, label="Human")
    visualize_samples(model, tok, feat, fv, val_m, tables_folder, images_folder, device, out_dir, n=5, label="Augmented")

    # Final summary
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  ChartQA-H  exact: {h_exact*100:.2f}%  relaxed: {h_relaxed*100:.2f}%")
    print(f"  ChartQA-M  exact: {m_exact*100:.2f}%  relaxed: {m_relaxed*100:.2f}%")
    print(f"  Overall    exact: {overall_exact*100:.2f}%  relaxed: {overall_relaxed*100:.2f}%")
    print(f"  Visualizations  : {out_dir}/")
    print()


if __name__ == '__main__':
    main()

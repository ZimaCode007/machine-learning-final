"""
VisionTapas QA Demo - Single chart prediction with visualization.
Loads best Phase 1 checkpoint, shows chart image, question, predicted answer vs gold answer.
"""
import json, os, pickle, argparse, random, warnings
warnings.filterwarnings('ignore')
import torch
import numpy as np
import pandas as pd
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
    model.eval()
    return model, tok, feat, fv, agg


def predict_single(model, tok, feat, fv, imgname, question, tables_folder, images_folder):
    df = pd.read_csv(os.path.join(tables_folder, imgname + '.csv'), encoding='utf8')
    df.columns = [str(c) if not (isinstance(c, float) and c != c) else f'col_{i}' for i, c in enumerate(df.columns)]
    df = df.fillna('').astype(object)
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip('%')

    encoding = tok(table=df, queries=[question], padding='max_length', truncation=True, return_tensors='pt')
    img = Image.open(os.path.join(images_folder, imgname + '.png')).convert('RGB')
    vis = feat(images=img, return_tensors='pt')
    encoding['pixel_values'] = vis['pixel_values']

    with torch.no_grad():
        outputs = model(return_dict=True, **encoding)

    logits = outputs.logits.numpy()
    logits_agg = outputs.aggregation_logits
    data_np = {k: v.numpy() for k, v in encoding.items() if k in ['input_ids', 'token_type_ids', 'attention_mask']}
    (pred_coords, agg_idx), _ = convert_logits_to_predictions(data_np, logits, logits_agg=logits_agg, cell_classification_threshold=0.5)

    ops = {0: 'SELECT', 1: 'SUM', 2: 'AVERAGE', 3: 'COUNT', 4: 'Diff', 5: 'Ratio'}
    op_idx = agg_idx[0] if agg_idx else 0
    if op_idx > 5:
        op_name = f'FIXED_VOCAB[{fv[op_idx - 6]}]'
    else:
        op_name = ops.get(op_idx, f'OP_{op_idx}')

    if op_idx > 5:
        pred_answer = fv[op_idx - 6]
    elif len(pred_coords) > 0 and len(pred_coords[0]) > 0:
        pred_answer = str(get_final_answer(df, pred_coords[0], op_idx, fv))
        cells = [f'({r},{c})={df.iloc[r, c]}' for r, c in pred_coords[0]]
    else:
        pred_answer = '[no cells selected]'
        cells = []

    return pred_answer, op_name, pred_coords[0] if pred_coords else [], img


def visualize(img, question, gold, pred, op_name, selected_cells, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [3, 2]})

    # Left: chart image
    axes[0].imshow(img)
    axes[0].axis('off')
    axes[0].set_title('Chart Image', fontsize=14, fontweight='bold')

    # Right: QA info
    axes[1].axis('off')

    correct = pred.strip().lower() == gold.strip().lower()
    try:
        g_f = float(gold.replace(',', ''))
        p_f = float(pred.replace(',', ''))
        relaxed = abs(g_f - p_f) / max(abs(g_f), 1e-9) <= 0.05
    except (ValueError, TypeError):
        relaxed = correct

    info_lines = []
    info_lines.append(('Question:', question, 'black'))
    info_lines.append(('', '', 'black'))
    info_lines.append(('Gold Answer:', gold, '#2196F3'))
    info_lines.append(('Predicted:', pred, '#4CAF50' if correct else '#FF9800' if relaxed else '#F44336'))
    info_lines.append(('Operation:', op_name, 'gray'))
    if selected_cells:
        cells_str = ', '.join([f'({r},{c})' for r, c in selected_cells[:5]])
        if len(selected_cells) > 5:
            cells_str += f' ... +{len(selected_cells)-5} more'
        info_lines.append(('Selected Cells:', cells_str, 'gray'))
    info_lines.append(('', '', 'black'))

    if correct:
        info_lines.append(('Result:', 'EXACT MATCH', '#4CAF50'))
    elif relaxed:
        info_lines.append(('Result:', 'RELAXED MATCH (within 5%)', '#FF9800'))
    else:
        info_lines.append(('Result:', 'INCORRECT', '#F44336'))

    y = 0.92
    for label, value, color in info_lines:
        if label:
            axes[1].text(0.02, y, label, fontsize=11, fontweight='bold', transform=axes[1].transAxes, verticalalignment='top')
        wrapped = value
        if len(value) > 50:
            wrapped = value[:50] + '\n' + value[50:100]
        axes[1].text(0.02, y - 0.06, wrapped, fontsize=11, color=color, transform=axes[1].transAxes, verticalalignment='top', wrap=True)
        y -= 0.16

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {save_path}')


def main():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    base_dataset = os.path.join(base, 'ChartQA Dataset')

    print('Loading model...')
    model, tok, feat, fv, agg = load_model('output_qa/phase1_best')
    print('Model loaded.\n')

    val_data = json.load(open('converted_data/val_qa.json', encoding='utf-8'))
    tables_folder = os.path.join(base_dataset, 'val', 'tables')
    images_folder = os.path.join(base_dataset, 'val', 'png')

    os.makedirs('demo_output', exist_ok=True)

    # Pick 5 diverse samples
    random.seed(123)
    indices = random.sample(range(len(val_data)), min(20, len(val_data)))

    count = 0
    for idx in indices:
        if count >= 5:
            break
        item = val_data[idx]
        imgname = item['image_index']
        question = item['question']
        gold = str(item['answer'][1][0]).strip()

        try:
            pred, op_name, cells, img = predict_single(
                model, tok, feat, fv, imgname, question, tables_folder, images_folder
            )
        except Exception as e:
            continue

        count += 1
        print(f'  [{count}] {imgname}')
        print(f'      Q: {question}')
        print(f'      Gold: {gold}')
        print(f'      Pred: {pred} ({op_name})')
        match = "CORRECT" if pred.strip().lower() == gold.lower() else "WRONG"
        print(f'      -> {match}')

        save_path = os.path.join('demo_output', f'demo_{count}_{imgname}.png')
        visualize(img, question, gold, pred, op_name, cells, save_path)
        print()

    print(f'Done! {count} predictions saved to demo_output/')


if __name__ == '__main__':
    main()

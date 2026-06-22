"""
Convert ChartQA dataset to VisionTapas QA format.
Uses ALL data (no top_k filtering).

QA format: {"image_index": "xxx", "question": "...", "answer": ["FIXED/OPEN", ["answer_text"]]}

Also generates a fixed_vocab.pkl for common non-numeric answers.
"""
import json
import os
import pickle
import re
from collections import Counter


def is_numeric(s):
    try:
        float(str(s).replace(",", "").replace("%", ""))
        return True
    except ValueError:
        return False


def convert_chartqa_to_qa(input_jsons, tables_folder, images_folder, output_json):
    all_data = []
    for input_json in input_jsons:
        with open(input_json, 'r', encoding='utf-8') as f:
            all_data.extend(json.load(f))

    converted = []
    skipped = 0
    for item in all_data:
        imgname = item['imgname'].replace('.png', '')
        table_path = os.path.join(tables_folder, imgname + '.csv')
        image_path = os.path.join(images_folder, imgname + '.png')
        if not os.path.exists(table_path) or not os.path.exists(image_path):
            skipped += 1
            continue

        label = str(item['label'])
        converted.append({
            'image_index': imgname,
            'question': item['query'],
            'answer': ['FIXED/OPEN', [label]]
        })

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"  Converted {len(converted)} samples -> {output_json}")
    if skipped > 0:
        print(f"  Skipped {skipped} (missing files)")
    return converted


def build_fixed_vocab(train_data, min_count=5, max_vocab=200):
    answer_counts = Counter()
    for item in train_data:
        answer = item['answer'][1][0]
        if not is_numeric(answer):
            answer_counts[answer.lower()] += 1

    fixed_vocab = [ans for ans, cnt in answer_counts.most_common(max_vocab) if cnt >= min_count]
    print(f"  Fixed vocab: {len(fixed_vocab)} non-numeric answers (min_count={min_count})")
    print(f"  Examples: {fixed_vocab[:15]}")
    return fixed_vocab


if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_base = os.path.join(base, 'ChartQA Dataset')
    out_dir = os.path.join(base, 'Models', 'VisionTapas', 'converted_data')
    os.makedirs(out_dir, exist_ok=True)

    print("=== Converting training data ===")
    train_data = convert_chartqa_to_qa(
        input_jsons=[
            os.path.join(dataset_base, 'train', 'train_augmented.json'),
            os.path.join(dataset_base, 'train', 'train_human.json'),
        ],
        tables_folder=os.path.join(dataset_base, 'train', 'tables'),
        images_folder=os.path.join(dataset_base, 'train', 'png'),
        output_json=os.path.join(out_dir, 'train_qa.json'),
    )

    print("\n=== Converting validation data ===")
    convert_chartqa_to_qa(
        input_jsons=[
            os.path.join(dataset_base, 'val', 'val_augmented.json'),
            os.path.join(dataset_base, 'val', 'val_human.json'),
        ],
        tables_folder=os.path.join(dataset_base, 'val', 'tables'),
        images_folder=os.path.join(dataset_base, 'val', 'png'),
        output_json=os.path.join(out_dir, 'val_qa.json'),
    )

    print("\n=== Building fixed vocabulary ===")
    fixed_vocab = build_fixed_vocab(train_data)
    vocab_path = os.path.join(out_dir, 'fixed_vocab.pkl')
    with open(vocab_path, 'wb') as f:
        pickle.dump(fixed_vocab, f)
    print(f"  Saved to {vocab_path}")

    print("\n=== Summary ===")
    print(f"  Train QA: {os.path.join(out_dir, 'train_qa.json')}")
    print(f"  Val QA:   {os.path.join(out_dir, 'val_qa.json')}")
    print(f"  Vocab:    {vocab_path}")

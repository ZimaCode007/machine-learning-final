"""
Convert ChartQA dataset to VisionTapas classification format.
Maps frequent answers to class indices; infrequent answers are dropped.
"""
import json
import os
from collections import Counter

def convert_chartqa_to_classification(input_json, tables_folder, images_folder, output_json, max_samples=None, top_k_answers=50):
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    answer_counts = Counter(item['label'] for item in data)
    top_answers = [a for a, _ in answer_counts.most_common(top_k_answers)]
    answer_to_idx = {a: i for i, a in enumerate(top_answers)}
    print(f"Top {top_k_answers} answers cover {sum(answer_counts[a] for a in top_answers)}/{len(data)} samples")

    converted = []
    for item in data:
        label = item['label']
        if label not in answer_to_idx:
            continue

        imgname = item['imgname'].replace('.png', '')
        table_path = os.path.join(tables_folder, imgname + '.csv')
        image_path = os.path.join(images_folder, imgname + '.png')
        if not os.path.exists(table_path) or not os.path.exists(image_path):
            continue

        converted.append({
            'image_index': imgname,
            'question': item['query'],
            'answer': answer_to_idx[label]
        })

        if max_samples and len(converted) >= max_samples:
            break

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(converted)} samples -> {output_json}")
    print(f"Num classes: {len(answer_to_idx)}")
    return answer_to_idx


if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_base = os.path.join(base, 'ChartQA Dataset')

    os.makedirs(os.path.join(base, 'Models', 'VisionTapas', 'converted_data'), exist_ok=True)

    answer_to_idx = convert_chartqa_to_classification(
        input_json=os.path.join(dataset_base, 'train', 'train_augmented.json'),
        tables_folder=os.path.join(dataset_base, 'train', 'tables'),
        images_folder=os.path.join(dataset_base, 'train', 'png'),
        output_json=os.path.join(base, 'Models', 'VisionTapas', 'converted_data', 'train.json'),
        max_samples=None,
        top_k_answers=30
    )

    # Merge val_augmented + val_human for a larger validation set
    import json as json_mod
    val_aug = json_mod.load(open(os.path.join(dataset_base, 'val', 'val_augmented.json'), encoding='utf-8'))
    val_hum = json_mod.load(open(os.path.join(dataset_base, 'val', 'val_human.json'), encoding='utf-8'))
    merged_val_path = os.path.join(base, 'Models', 'VisionTapas', 'converted_data', '_val_merged.json')
    json_mod.dump(val_aug + val_hum, open(merged_val_path, 'w', encoding='utf-8'), ensure_ascii=False)
    print(f"Merged val: {len(val_aug)} + {len(val_hum)} = {len(val_aug)+len(val_hum)} samples")

    convert_chartqa_to_classification(
        input_json=merged_val_path,
        tables_folder=os.path.join(dataset_base, 'val', 'tables'),
        images_folder=os.path.join(dataset_base, 'val', 'png'),
        output_json=os.path.join(base, 'Models', 'VisionTapas', 'converted_data', 'val.json'),
        max_samples=None,
        top_k_answers=30
    )
    os.remove(merged_val_path)

    print(f"\nNum classes: {len(answer_to_idx)}")
    print("Answer mapping:", answer_to_idx)

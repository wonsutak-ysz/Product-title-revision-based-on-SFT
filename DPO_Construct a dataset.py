import json
import random

random.seed(1)


sft_train_file = 'LLaMA-Factory/data/标题-train.json'


def make_dpo_candidate_file():
    """构造dpo候选集，训练集复制5次，使用更大的温度系数/topp预测"""
    samples = []
    with open(sft_train_file, 'r', encoding='utf-8') as f:
        sft_train_samples = json.load(f)
        for sample in sft_train_samples:
            for i in range(5):
                samples.append(sample)
    with open('LLaMA-Factory/data/标题-dpo-candidate.json', 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=4)


def make_dpo_train_file():
    sft_train_predict_file = 'LLaMA-Factory/saves/Custom/lora/predict_qwen2.5_7b_chat_dpo_candidate_p0.9_t1.4/generated_predictions.jsonl'
    dpo_train_file = 'LLaMA-Factory/data/标题-dpo-train.json'

    with open('LLaMA-Factory/data/标题-dpo-candidate.json', 'r', encoding='utf-8') as f:
        sft_train_samples = json.load(f)
    new_samples = []
    duplicated_set = set()  # 用于新数据去重
    with open(sft_train_predict_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            line = json.loads(line)
            if line['label'] == line['predict']:
                continue
            if line['predict'] in duplicated_set:
                continue
            print(line)
            new_sample = {
                "conversations": [sft_train_samples[i]['conversations'][0]],
                "chosen": {"from": "gpt", "value": line['label']},
                "rejected": {"from": "gpt", "value": line['predict']}
            }
            new_samples.append(new_sample)
            duplicated_set.add(line['predict'])
    random.shuffle(new_samples)

    with open(dpo_train_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(new_samples, ensure_ascii=False, indent=4))


if __name__ == '__main__':
    # make_dpo_candidate_file()
    make_dpo_train_file()

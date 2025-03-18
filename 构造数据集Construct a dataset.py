import pandas as pd
import os
import random
import json

random.seed(1)

def _read_data():
    # 逐行读取原始标注数据
    os.system('pwd')
    for line in pd.read_excel('LLaMA-Factory/data/标题改写数据集.xlsx', sheet_name='卫生巾').to_dict('records'):
        yield line
    for line in pd.read_excel('LLaMA-Factory/data/标题改写数据集.xlsx', sheet_name='啤酒').to_dict('records'):
        yield line


cate_conf = {
    '卫生巾': ['品牌', '适用时间', '材质', '类型', '商品主体', '长度', '包装规格'],
    '啤酒': ['品牌', '系列', '度数', '商品主体(含工艺)', '包装规格']
}

def _gen_prompt_output(field_dict):
    cate_name, title, new_title = field_dict['cate_name'], field_dict['title'], field_dict['new_title']
    break_line = '\n'
    prompt = f"""
你是一个零售商品标题生成器，可以根据已有的标题和CPV图谱信息，生成一条标准化的标题。

已知如下2种信息：

1. 商品的{len(cate_conf[cate_name])}个属性（可能有错误或遗漏）
{break_line.join(['{}：{}'.format(k, field_dict[k]) for k in cate_conf[cate_name]])}
2. 商品原始标题：{title}

根据上面的商品属性和商品原始标题，生成新的标题，思考步骤为：
1. 根据商品属性，按照顺序拼接其取值，忽略取值为“无”的属性，卫生巾的包装规格前面可能需要加一个空格，得到新标题；
2. 商品属性可能会有错误或者遗漏，请根据商品原始标题修正新标题，查缺补漏改错；
3. 再次修正新标题，保证其语义通顺易懂，精简不重复；


请严格遵循以下输出格式回答：
新标题为：<新标题>

现在你的回答是：
""".strip()
    output = f"新标题为：{new_title}"
    return prompt, output


def main():
    samples = []
    for line in _read_data():
        prompt, output = _gen_prompt_output(line)
        samples.append({
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": output}
            ]
        })
    random.shuffle(samples)
    train_samples, dev_samples = samples[:int(len(samples) * 0.8)], samples[int(len(samples) * 0.8):]
    with open('LLaMA-Factory/data/标题-train.json', 'w') as f:
        f.write(json.dumps(train_samples, ensure_ascii=False, indent=2))
    with open('LLaMA-Factory/data/标题-dev.json', 'w') as f:
        f.write(json.dumps(dev_samples, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    # for line in _read_data():
            # print(line)
            # prompt, output = _gen_prompt_output(line)
            # print(prompt)
            # print(output)
            # exit()
    main()

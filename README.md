# cotgen

## prompt++
- cot_to_sana.py: 把 geneval 的简单的 prompt 转化为复杂的 cot prompt 和可组合的 prompt
- geneval_cot_metadata.jsonl: 用 cot_to_sana.py 生成的 geneval 的复杂 prompts
- dpgbench_cot_metadata.jsonl: 用 cot_to_sana.py 生成的 dpgbench 的复杂 prompts

## prompt2image
- generate_from_prompts.py: 这是从https://github.com/djghosh13/geneval/blob/main/generation/diffusers_generate.py改过来的，生成 geneval/dpgbench 原本的 prompts 对应的图像
- generate_from_cot.py: 从 qwen 生成的复杂 prompt 当中进行图像生成
- generate_from_cot_labels.py: 从 qwen 生成的 labels 当中进行多个图像生成以待拼接

## results
- cot_geneval_results.jsonl: geneval 上，从 qwen 生成的复杂 prompt 得到的图像的 judge 结果
- vanilla_geneval_results.jsonl: geneval 上，原始 prompt 得到的图像 judge 结果
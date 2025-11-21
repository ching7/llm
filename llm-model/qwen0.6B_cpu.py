"""
Qwen 0.6B 模型 - CPU 版本
使用 CPU 进行推理，稳定可靠，适合所有环境
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# 使用 CPU 设备
device = "cpu"
print(f"💻 使用 CPU 设备（稳定可靠）")

# 使用 ModelScope 下载的模型路径
model_path = os.path.expanduser("~/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B")  

print("\n正在加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print("正在加载模型到 CPU...")
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    trust_remote_code=True,
    torch_dtype=torch.float32  # CPU 使用 float32
)
model = model.to(device)
model.eval()

prompt = "用一个生活中的例子说明 attention 是什么："
print(f"\n输入提示: {prompt}\n")

inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# 设置 pad_token（如果不存在）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("正在生成回复（CPU 模式，可能需要一些时间）...")
with torch.no_grad():
    out = model.generate(
        **inputs, 
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    print("✅ 生成完成！")

print("\n" + "="*50)
print("生成结果:")
print("="*50)
result = tokenizer.decode(out[0], skip_special_tokens=True)
print(result)
print("="*50)


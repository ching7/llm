#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM SFT 模型评估脚本
用于评估微调后的模型性能
支持与训练脚本相同的数据格式
"""

import os
import json
import torch
import argparse
import platform
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

def is_apple_silicon():
    """检测是否是 Apple Silicon (M1/M2/M3 等)"""
    try:
        machine = platform.machine()
        if machine == 'arm64':
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return True
        return False
    except:
        return False

def is_intel_mac():
    """检测是否是 Intel Mac"""
    try:
        machine = platform.machine()
        return machine == 'x86_64'
    except:
        return False

def load_dataset(data_path):
    """加载评估数据集，支持json和jsonl格式"""
    data = []
    if data_path.endswith('.jsonl'):
        # jsonl格式：每行一个json对象
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    else:
        # json格式：整个文件是一个json数组或对象
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    return data

def format_prompt(example):
    """格式化评估提示（与训练脚本格式一致）"""
    query = example.get("query", "")
    language = example.get("tag", "")
    
    # 构建对话格式（与训练脚本一致）
    prompt = f"""
        ### 输入:
        {query}

        ### language:
        {language}
        
        ### 输出:
"""
    
    return prompt

def generate_response(model, tokenizer, prompt, device, max_new_tokens=200):
    """生成模型响应"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解码完整响应
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # 提取输出部分（从 "### 输出:" 之后的内容）
    if "### 输出:" in full_response:
        response = full_response.split("### 输出:")[-1].strip()
        # 移除可能的 EOS token
        response = response.replace(tokenizer.eos_token, "").strip()
    else:
        # 如果没有找到标记，返回完整响应（去除输入部分）
        response = full_response.replace(prompt.strip(), "").strip()
    
    return response

def evaluate_model(model_path, data_path, device=None, max_new_tokens=200):
    """评估模型"""
    # 自动检测设备（与训练脚本一致）
    if device is None:
        if is_intel_mac():
            device = "cpu"
            if hasattr(torch.backends, "mps"):
                torch.backends.mps.enabled = False
            print(f"💻 检测到 Intel 芯片，使用 CPU 设备进行评估")
        elif is_apple_silicon() and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            print(f"💻 检测到 Apple Silicon，使用 MPS (Metal) 设备进行评估")
        else:
            device = "cpu"
            if hasattr(torch.backends, "mps"):
                torch.backends.mps.enabled = False
            print(f"💻 使用 {device} 设备进行评估")
    else:
        print(f"💻 使用 {device} 设备进行评估")
    
    print(f"📦 加载模型: {model_path}")
    
    # 加载分词器
    print("\n正在加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # 设置pad_token（如果不存在）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    print("正在加载模型...")
    # 根据设备选择数据类型
    if device == "mps":
        dtype = torch.float16
        print("📊 使用 float16 数据类型以节省内存")
    else:
        dtype = torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype
    )
    
    # 移动到设备
    if device != "cpu":
        model = model.to(device)
    else:
        model = model.cpu()
        # 确保所有参数都在 CPU 上
        for param in model.parameters():
            if param.device.type != "cpu":
                param.data = param.data.cpu()
    
    model.eval()
    
    # 清理缓存
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    
    # 加载数据集
    print("正在加载评估数据集...")
    data = load_dataset(data_path)
    
    # 评估结果
    results = []
    
    print(f"\n开始评估，共 {len(data)} 个样本...")
    
    for i, example in enumerate(data):
        print(f"\n{'='*60}")
        print(f"📝 评估样本 {i+1}/{len(data)}")
        print(f"{'='*60}")
        
        # 格式化提示
        prompt = format_prompt(example)
        query = example.get("query", "")
        language = example.get("tag", "")
        
        print(f"\n📥 输入:")
        print(f"  Query: {query}")
        print(f"  Language: {language}")
        
        # 生成响应
        response = generate_response(model, tokenizer, prompt, device, max_new_tokens=max_new_tokens)
        print(f"\n🤖 模型输出:")
        print(f"  {response}")
        
        # 期望输出
        expected_output = example.get("response", "")
        print(f"\n✅ 期望输出:")
        print(f"  {expected_output}")
        
        # 简单相似度检查（可选）
        similarity = "✓" if expected_output.strip() in response or response.strip() in expected_output else "✗"
        print(f"\n📊 匹配度: {similarity}")
        
        # 保存结果
        results.append({
            "query": query,
            "tag": language,
            "expected_response": expected_output,
            "model_response": response,
            "match": similarity == "✓"
        })
    
    return results

def save_results(results, output_path):
    """保存评估结果"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n📁 评估结果已保存到: {output_path}")

def calculate_metrics(results):
    """计算评估指标"""
    total = len(results)
    matched = sum(1 for r in results if r.get("match", False))
    match_rate = (matched / total * 100) if total > 0 else 0
    
    return {
        "total_samples": total,
        "matched_samples": matched,
        "match_rate": f"{match_rate:.2f}%"
    }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LLM SFT 模型评估脚本")
    parser.add_argument("--model_path", type=str, default="../outputs/sft_results/final_model", 
                       help="微调后的模型路径")
    parser.add_argument("--data_path", type=str, default="../data/self_cognition.jsonl", 
                       help="评估数据集路径（支持 json 和 jsonl 格式）")
    parser.add_argument("--output_path", type=str, default="../outputs/evaluation_results.json", 
                       help="评估结果保存路径")
    parser.add_argument("--device", type=str, default=None, 
                       help="使用的设备（cpu/mps/cuda），默认自动检测")
    parser.add_argument("--max_new_tokens", type=int, default=200, 
                       help="生成的最大token数")
    
    args = parser.parse_args()
    
    # 确保路径是绝对路径
    if not os.path.isabs(args.model_path):
        args.model_path = os.path.join(os.path.dirname(__file__), args.model_path)
    if not os.path.isabs(args.data_path):
        args.data_path = os.path.join(os.path.dirname(__file__), args.data_path)
    if not os.path.isabs(args.output_path):
        args.output_path = os.path.join(os.path.dirname(__file__), args.output_path)
    
    # 评估模型
    results = evaluate_model(args.model_path, args.data_path, args.device, args.max_new_tokens)
    
    # 计算指标
    metrics = calculate_metrics(results)
    
    # 保存评估结果
    save_results(results, args.output_path)
    
    # 打印总结
    print(f"\n{'='*60}")
    print(f"📊 评估总结")
    print(f"{'='*60}")
    print(f"总样本数: {metrics['total_samples']}")
    print(f"匹配样本数: {metrics['matched_samples']}")
    print(f"匹配率: {metrics['match_rate']}")
    print(f"{'='*60}")
    print(f"\n🎉 评估完成！")
    print(f"📁 详细结果已保存到: {args.output_path}")

if __name__ == "__main__":
    main()

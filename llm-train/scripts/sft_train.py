#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM SFT 微调脚本
支持 MPS (Metal) 和 CPU 进行模型监督微调
自动优化内存使用以适配 MPS 设备
"""

import os
import json
import torch
import gc
import platform
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

def is_apple_silicon():
    """检测是否是 Apple Silicon (M1/M2/M3 等)"""
    try:
        # 检查处理器架构
        machine = platform.machine()
        # Apple Silicon 的架构是 'arm64'
        if machine == 'arm64':
            # 进一步检查是否有 MPS 支持
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return True
        return False
    except:
        return False

def is_intel_mac():
    """检测是否是 Intel Mac"""
    try:
        machine = platform.machine()
        # Intel Mac 的架构是 'x86_64'
        return machine == 'x86_64'
    except:
        return False

# 检测芯片类型并设置 MPS
if is_intel_mac():
    # Intel Mac 不支持 MPS，显式禁用
    if hasattr(torch.backends, "mps"):
        torch.backends.mps.enabled = False
        print("🔧 Intel Mac：已禁用 MPS 支持")
elif is_apple_silicon():
    # Apple Silicon 可以使用 MPS，设置内存限制
    # 设置 MPS 内存限制（限制为系统内存的 80%，避免内存不足）
    # 值必须在 0.0-1.0 之间，0.8 表示限制为 80%
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.8"

def load_config(config_path):
    """加载配置文件"""
    # 确保路径是绝对路径
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 扩展模型路径中的波浪号
    if config["model_path"].startswith("~"):
        config["model_path"] = os.path.expanduser(config["model_path"])
    
    # 确保数据路径是绝对路径
    if not os.path.isabs(config["data_path"]):
        config["data_path"] = os.path.join(os.path.dirname(__file__), config["data_path"])
    
    # 确保输出目录是绝对路径
    if not os.path.isabs(config["output_dir"]):
        config["output_dir"] = os.path.join(os.path.dirname(__file__), config["output_dir"])
    
    return config

def load_dataset(data_path):
    """加载训练数据集，支持json和jsonl格式"""
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

def format_example(example, tokenizer):
    """格式化训练样本"""
    query = example.get("query", "")
    response = example.get("response", "")
    language = example.get("tag", "")

    # 构建对话格式
    prompt = f"""
        ### 输入:
        {query}

        ### language:
        {language}
        
        ### 输出:
        {response}
    """
    
    # 完整文本
    full_text = prompt + tokenizer.eos_token
    
    return {
        "prompt": prompt,
        "full_text": full_text
    }

def tokenize_function(examples, tokenizer):
    """分词处理"""
    tokenized = tokenizer(
        examples["full_text"],
        truncation=True,
        max_length=256,  # 减少序列长度以降低内存占用（从512降到256）
        padding="max_length"
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def main():
    """主函数"""
    # 加载配置
    config = load_config("../configs/sft_config.json")
    
    # 检测芯片类型并设置设备
    if is_intel_mac():
        # Intel Mac 不支持 MPS，强制使用 CPU
        device = "cpu"
        # 确保 MPS 已禁用（双重保险）
        if hasattr(torch.backends, "mps"):
            torch.backends.mps.enabled = False
        print(f"💻 检测到 Intel 芯片，使用 CPU 设备进行训练")
        print(f"ℹ️  Intel Mac 不支持 MPS (Metal) 加速，仅 Apple Silicon (M1/M2/M3) 支持")
    elif is_apple_silicon() and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Apple Silicon 可以使用 MPS
        device = "mps"
        print(f"💻 检测到 Apple Silicon，使用 MPS (Metal) 设备进行训练")
        print(f"⚠️  MPS 内存限制已设置为 80%")
    else:
        # 其他情况使用 CPU
        device = "cpu"
        # 确保 MPS 已禁用
        if hasattr(torch.backends, "mps"):
            torch.backends.mps.enabled = False
        print(f"💻 使用 {device} 设备进行训练")
    
    # 模型路径
    model_path = config["model_path"]
    print(f"📦 使用模型: {model_path}")
    
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
    # 根据设备选择数据类型：MPS 可以使用 float16 节省内存
    if device == "mps":
        # MPS 支持 float16，可以节省约 50% 内存
        dtype = torch.float16
        print("📊 使用 float16 数据类型以节省内存")
    else:
        dtype = torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype  # 注意：虽然警告说已弃用，但目前仍需要使用 torch_dtype
    )
    
    # 只有在非 CPU 设备时才移动模型
    if device != "cpu":
        model = model.to(device)
    
    # 清理缓存
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    
    # 加载数据集
    print("正在加载数据集...")
    raw_data = load_dataset(config["data_path"])
    
    # 格式化数据
    formatted_data = [format_example(example, tokenizer) for example in raw_data]
    
    # 转换为Dataset对象
    dataset = Dataset.from_list(formatted_data)
    
    # 分词处理
    print("正在处理训练数据...")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # 非掩码语言模型
    )
    
    # 训练参数
    # 根据设备调整 batch size（MPS 内存有限，使用更小的 batch size）
    batch_size = config["per_device_train_batch_size"]
    if device == "mps":
        # MPS 内存有限，确保 batch size 不超过 1
        batch_size = min(batch_size, 1)
        print(f"📦 MPS 设备：batch size 设置为 {batch_size}")
    
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        overwrite_output_dir=True,
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4 if device == "mps" else 1,  # MPS 使用梯度累积减少内存峰值
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        logging_dir=os.path.join(config["output_dir"], "logs"),
        logging_steps=config["logging_steps"],
        save_strategy="epoch",
        save_total_limit=2,
        fp16=(device == "mps"),  # MPS 可以使用 fp16 节省内存，CPU 不使用
        report_to="none",
        dataloader_num_workers=0,
        # MPS 内存优化选项
        dataloader_pin_memory=False,  # MPS 不需要 pin_memory
        remove_unused_columns=True,  # 移除未使用的列以节省内存
        # 明确指定不使用 MPS（如果 device 是 CPU）
        no_cuda=(device == "cpu"),  # CPU 训练时禁用 CUDA
    )
    
    # 初始化Trainer
    # 确保模型在正确的设备上
    if device == "cpu":
        # 强制模型在 CPU 上
        model = model.cpu()
        # 确保所有参数都在 CPU 上
        for param in model.parameters():
            if param.device.type != "cpu":
                param.data = param.data.cpu()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # 开始训练
    print("\n🚀 开始SFT微调训练...")
    try:
        trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e) or "MPS" in str(e):
            print("\n❌ MPS 内存不足！")
            print("💡 建议：")
            print("   1. 进一步减少序列长度（当前 256）")
            print("   2. 使用 CPU 训练（虽然慢但稳定）")
            print("   3. 使用 LoRA 等参数高效微调方法")
            raise
        else:
            raise
    finally:
        # 训练后清理内存
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    
    # 保存模型
    print("\n💾 保存微调后的模型...")
    trainer.save_model(os.path.join(config["output_dir"], "final_model"))
    tokenizer.save_pretrained(os.path.join(config["output_dir"], "final_model"))
    
    print("\n🎉 SFT微调训练完成！")
    print(f"📁 模型保存路径: {os.path.join(config['output_dir'], 'final_model')}")

if __name__ == "__main__":
    main()

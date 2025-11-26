# MPS 内存优化说明

## 概述

训练脚本已优化以支持 MPS (Metal Performance Shaders) 设备，并实施了多项内存限制措施以避免内存不足错误。

## 已实施的优化措施

### 1. **MPS 内存限制环境变量**

```python
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.8"
```

- 限制 MPS 使用系统内存的 80%
- 防止内存溢出导致系统不稳定
- 值范围：0.0-1.0（0.0 表示不限制，但可能不稳定）

### 2. **序列长度优化**

```python
max_length=256  # 从 512 减少到 256
```

- **内存节省**：约 50%（序列长度减半）
- **影响**：可能略微影响长文本处理能力
- **权衡**：对于大多数任务，256 长度已足够

### 3. **数据类型优化**

```python
dtype = torch.float16  # MPS 使用 float16
```

- **内存节省**：约 50%（相比 float32）
- **性能**：MPS 对 float16 支持良好
- **精度**：对大多数训练任务影响可忽略

### 4. **Batch Size 限制**

```python
batch_size = min(batch_size, 1)  # MPS 限制为 1
```

- **内存节省**：减少激活值内存占用
- **补偿**：通过梯度累积保持有效 batch size

### 5. **梯度累积**

```python
gradient_accumulation_steps=4  # MPS 使用梯度累积
```

- **效果**：有效 batch size = batch_size × gradient_accumulation_steps
- **内存节省**：减少前向传播的内存峰值
- **训练效果**：与直接使用大 batch size 效果相近

### 6. **混合精度训练**

```python
fp16=True  # MPS 启用 fp16
```

- **内存节省**：约 50%（激活值使用半精度）
- **性能**：MPS 对 fp16 支持良好，训练速度可能更快

### 7. **内存清理机制**

```python
torch.mps.empty_cache()  # 清理 MPS 缓存
gc.collect()  # Python 垃圾回收
```

- 在模型加载后清理
- 在训练完成后清理
- 防止内存碎片化

### 8. **其他优化选项**

```python
dataloader_pin_memory=False  # MPS 不需要 pin_memory
remove_unused_columns=True  # 移除未使用的列
```

## 内存占用估算

### 优化前（float32, 序列长度 512, batch_size=1）

- 模型参数：~1.2 GB
- 前向激活值：~3-4 GB
- 梯度：~1.2 GB
- 优化器状态：~2.4 GB
- **总计**：~8-9 GB ❌（超过 MPS 限制）

### 优化后（float16, 序列长度 256, batch_size=1, 梯度累积）

- 模型参数：~0.6 GB（float16）
- 前向激活值：~0.8-1.0 GB（float16 + 序列长度减半）
- 梯度：~0.6 GB（float16）
- 优化器状态：~1.2 GB（float16）
- **总计**：~3.2-3.4 GB ✅（在 MPS 限制内）

## 使用建议

### 如果仍然出现内存不足

1. **进一步减少序列长度**
   ```python
   max_length=128  # 从 256 进一步减少
   ```

2. **增加梯度累积步数**
   ```python
   gradient_accumulation_steps=8  # 从 4 增加到 8
   ```

3. **调整 MPS 内存限制**
   ```python
   os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.7"  # 更保守的限制
   ```

4. **使用 LoRA 微调**（推荐）
   - 可以进一步减少 50-70% 内存占用
   - 需要安装 `peft` 库

### 性能对比

| 配置 | 内存占用 | 训练速度 | 稳定性 |
|------|---------|---------|--------|
| CPU (float32) | ~8 GB | 慢 | 高 ✅ |
| MPS (float16, 优化) | ~3-4 GB | 快 | 中 ⚠️ |
| MPS (float16, LoRA) | ~1-2 GB | 快 | 高 ✅ |

## 错误处理

脚本已添加错误处理，如果出现内存不足错误，会提供明确的建议：

```python
except RuntimeError as e:
    if "out of memory" in str(e) or "MPS" in str(e):
        print("❌ MPS 内存不足！")
        print("💡 建议：...")
```

## 监控内存使用

可以在训练过程中监控 MPS 内存使用：

```python
# 在训练循环中添加
if device == "mps":
    allocated = torch.mps.current_allocated_memory() / 1024**3  # GB
    print(f"MPS 已分配内存: {allocated:.2f} GB")
```

## 总结

通过以上优化措施，训练脚本现在可以：
- ✅ 在 MPS 设备上运行
- ✅ 内存占用减少约 60-70%
- ✅ 保持训练效果（通过梯度累积）
- ✅ 自动处理内存不足错误

如果仍有问题，建议使用 LoRA 微调或切换到 CPU 训练。


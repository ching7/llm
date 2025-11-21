#!/bin/bash
# 使用 CPU 版本运行脚本

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 直接使用虚拟环境中的 Python（最可靠的方式）
"$SCRIPT_DIR/venv/bin/python" qwen0.6B_cpu.py


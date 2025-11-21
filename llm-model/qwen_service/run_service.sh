#!/bin/bash
# 启动 Qwen 0.6B 推理服务

# 激活虚拟环境
source ../venv/bin/activate

# 启动服务
echo "🚀 启动 Qwen 0.6B 推理服务..."
echo "📡 服务地址: http://localhost:8000"
echo "📚 API 文档: http://localhost:8000/docs"
echo ""

python3 app.py


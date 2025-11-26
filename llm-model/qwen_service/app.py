#!/Users/chenyanan/Downloads/gitproject/llm/llm-model/venv/bin/python3
"""
Qwen 0.6B 模型推理服务
基于 FastAPI 的持续推理服务，模型加载一次后持续提供服务
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import uvicorn
from typing import Optional

# 初始化 FastAPI 应用
app = FastAPI(
    title="Qwen 0.6B 推理服务",
    description="基于 Qwen 0.6B 模型的持续推理服务",
    version="1.0.0"
)

# 全局变量存储模型和分词器
model = None
tokenizer = None
device = "cpu"

# 请求模型
class InferenceRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    do_sample: Optional[bool] = True

# 响应模型
class InferenceResponse(BaseModel):
    result: str
    status: str
    message: str

@app.on_event("startup")
def load_model():
    """启动时加载模型（只加载一次）"""
    global model, tokenizer, device
    
    print("="*60)
    print("🚀 正在启动 Qwen 0.6B 推理服务...")
    print("="*60)
    
    device = "cpu"
    print(f"💻 使用设备: {device}")
    
    # 模型路径
    # model_path = os.path.expanduser("~/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B")
    model_path = "/Users/chenyanan/Downloads/gitproject/llm/llm-train/outputs/sft_results/final_model"
    
    print("\n📥 正在加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print("📥 正在加载模型（这可能需要一些时间）...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
    model = model.to(device)
    model.eval()
    
    # 设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ 模型加载完成！服务已就绪")
    print("="*60)
    print(f"📡 API 文档地址: http://localhost:8000/docs")
    print(f"🔗 健康检查: http://localhost:8000/health")
    print("="*60)

@app.get("/health")
def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device
    }

@app.post("/inference", response_model=InferenceResponse)
def inference(request: InferenceRequest):
    """
    推理接口
    
    参数:
    - prompt: 输入提示文本
    - max_new_tokens: 最大生成token数（默认200）
    - temperature: 温度参数，控制随机性（默认0.7）
    - top_p: 核采样参数（默认0.9）
    - do_sample: 是否启用采样（默认True）
    """
    global model, tokenizer, device
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="模型未加载，请稍后重试")
    
    try:
        # 编码输入
        inputs = tokenizer(request.prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 生成回复
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=request.do_sample,
                temperature=request.temperature,
                top_p=request.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # 解码输出
        result = tokenizer.decode(out[0], skip_special_tokens=True)
        
        return InferenceResponse(
            result=result,
            status="success",
            message="推理完成"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")

@app.get("/")
def root():
    """根路径，返回服务信息"""
    return {
        "service": "Qwen 0.6B 推理服务",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "inference": "/inference (POST)",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    # 启动服务
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # 生产环境建议设为 False
        log_level="info"
    )


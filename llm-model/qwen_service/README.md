# Qwen 0.6B 推理服务

基于 FastAPI 的持续推理服务，模型加载一次后持续提供服务。

## 功能特点

- ✅ 模型只加载一次，持续提供服务
- ✅ RESTful API 接口
- ✅ 自动生成 API 文档
- ✅ 支持自定义生成参数
- ✅ 健康检查接口

## 文件说明

- `app.py` - 主服务文件
- `run_service.sh` - 启动脚本
- `test_client.py` - 测试客户端
- `README.md` - 说明文档

## 使用方法

### 1. 启动服务

```bash
cd qwen_service
./run_service.sh
```

或者手动启动：

```bash
source ../venv/bin/activate
python3 app.py
```

服务启动后：
- 服务地址: http://localhost:8000
- API 文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health

### 2. 测试服务

使用测试客户端：

```bash
source ../venv/bin/activate
python3 test_client.py
```

或使用 curl：

```bash
# 健康检查
curl http://localhost:8000/health

# 推理请求
curl -X POST "http://localhost:8000/inference" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "用一个生活中的例子说明 attention 是什么：",
    "max_new_tokens": 200,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```

### 3. API 接口说明

#### POST /inference

推理接口

**请求体：**
```json
{
  "prompt": "输入提示文本",
  "max_new_tokens": 200,      // 可选，默认 200
  "temperature": 0.7,          // 可选，默认 0.7
  "top_p": 0.9,                // 可选，默认 0.9
  "do_sample": true            // 可选，默认 true
}
```

**响应：**
```json
{
  "result": "生成的文本",
  "status": "success",
  "message": "推理完成"
}
```

#### GET /health

健康检查接口

**响应：**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

#### GET /docs

自动生成的 API 文档（Swagger UI）

## 注意事项

1. 首次启动需要加载模型，可能需要一些时间
2. 服务运行后，模型会常驻内存
3. 默认端口为 8000，可在 `app.py` 中修改
4. 建议在生产环境中使用进程管理器（如 systemd、supervisor）

## 性能优化建议

1. 使用多进程部署（通过 uvicorn workers）
2. 添加请求队列和限流
3. 使用 GPU 版本（如果支持）
4. 添加缓存机制


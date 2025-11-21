"""
测试客户端 - 用于测试推理服务
"""
import requests
import json

# 服务地址
BASE_URL = "http://localhost:8000"

def test_health():
    """测试健康检查"""
    print("🔍 测试健康检查...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()

def test_inference(prompt: str):
    """测试推理接口"""
    print(f"💬 测试推理，提示: {prompt}")
    print("-" * 60)
    
    data = {
        "prompt": prompt,
        "max_new_tokens": 200,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True
    }
    
    response = requests.post(
        f"{BASE_URL}/inference",
        json=data
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ 推理成功")
        print(f"📝 结果: {result['result']}")
    else:
        print(f"❌ 推理失败: {response.status_code}")
        print(f"错误信息: {response.text}")
    
    print("-" * 60)
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Qwen 0.6B 推理服务测试客户端")
    print("=" * 60)
    print()
    
    # 测试健康检查
    try:
        test_health()
    except Exception as e:
        print(f"❌ 无法连接到服务: {e}")
        print("💡 请确保服务已启动: ./run_service.sh")
        exit(1)
    
    # 测试推理
    test_prompts = [
        "用一个生活中的例子说明 attention 是什么：",
        "解释一下什么是机器学习：",
        "写一首关于春天的短诗："
    ]
    
    for prompt in test_prompts:
        test_inference(prompt)
        input("按 Enter 继续下一个测试...")


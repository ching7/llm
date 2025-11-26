from vllm import LLM

llm = LLM("~/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B")  # 填你的本地模型目录也行
output = llm.generate("你好，请介绍一下 vLLM？")

print(output[0].outputs[0].text)
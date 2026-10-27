import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. 加载 .env 环境变量
load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("没有找到 DEEPSEEK_API_KEY")

# 2. 创建 DeepSeek 客户端
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
    timeout=60.0,
    max_retries=1,
)

# 3. 发送对话请求
response = client.chat.completions.create(
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"),
    messages=[
        {
            "role": "system",
            "content": "你是一名学术助手，请使用中文简洁回答。"
        },
        {
            "role": "user",
            "content": "请用一句话解释什么是RAG。"
        }
    ],
    extra_body={"thinking": {"type": "disabled"}},
    stream=False,
)

# 4. 获取模型返回的文本
answer = response.choices[0].message.content

print("DeepSeek 回答：")
print(answer)

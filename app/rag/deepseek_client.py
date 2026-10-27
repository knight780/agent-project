
import os

from dotenv import load_dotenv
from openai import OpenAI


def call_deepseek(prompt: str, *, system_prompt: str = "你是一名学术助手，请使用中文简洁回答。") -> str:
    """调用 DeepSeek API 并返回模型生成的文本。"""

    # 1. 检查 Prompt 是否为空
    if not prompt.strip():
        raise ValueError("Prompt 不能为空。")

    # 2. 加载环境变量
    load_dotenv()

    # 3. 读取 API Key 和模型名称
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise ValueError("没有找到 DEEPSEEK_API_KEY")

    model_name = os.getenv(
        "DEEPSEEK_MODEL", "deepseek-v4-pro"
    )

    # 4. 创建客户端
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=60.0,
        max_retries=1,
    )

    # 5. 调用 DeepSeek
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
        extra_body={"thinking": {"type": "disabled"}},
        stream=False,
    )

    # 6. 获取回答
    answer = response.choices[0].message.content

    # 7. 检查回答是否为空
    if not answer or not answer.strip():
        raise ValueError("模型返回的回答为空。")

    # 8. 返回回答
    return answer.strip()

from dotenv import load_dotenv
from typing import List, Dict
from openai import OpenAI

# 환경 변수 로드 (.env)
load_dotenv()

# OpenAI 클라이언트 초기화
client = OpenAI()

# 프롬프트 템플릿
PROMPT_TEMPLATE = "주제 {topic}에 대해 짧은 설명을 해주세요."

def call_chat_model(messages: List[Dict[str, str]]) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    return response.choices[0].message.content

def invoke_chain(topic: str) -> str:
    prompt = PROMPT_TEMPLATE.format(topic=topic)
    messages = [{"role": "user", "content": prompt}]
    return call_chat_model(messages)

if __name__ == "__main__":
    print(invoke_chain("더블딥"))

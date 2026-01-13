from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()  # 로컬에서는 경로 안 써도 됨 (.env를 자동으로 찾음)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "안녕하세요!"}]
)

print(response.choices[0].message.content)

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 환경 변수 로드 (.env)
load_dotenv()

# 2. LLM 모델 설정
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=200,
)

# 3. 프롬프트 템플릿 정의
prompt = ChatPromptTemplate.from_template(
    "주제 {topic}에 대해 짧은 설명을 해주세요."
)

# 4. 출력 파서 정의
output_parser = StrOutputParser()

# 5. LCEL 체인 구성
chain = prompt | model | output_parser

# 6. 실행
if __name__ == "__main__":
    result = chain.batch([{"topic": "더블딥"}, {"topic": "인플레이션"}])
    for i, r in enumerate(result, start=1):
        print(f"\n[{i}]")
        print(r)
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 환경 변수 로드
load_dotenv()

# 2. 모델 설정
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=200,
)

# 3. 주제 설명 프롬프트
prompt = ChatPromptTemplate.from_template(
    "주제 {topic}에 대해 짧은 설명을 해주세요."
)

# 4. 출력 파서
output_parser = StrOutputParser()

# 5. 1차 체인: 주제 → 설명
chain = prompt | model | StrOutputParser()

# 6. 번역 프롬프트
analysis_prompt = ChatPromptTemplate.from_template(
    "이 답변을 영어로 번역해 주세요:\n{answer}"
)

# 7. pipe()로 체인 합성
composed_chain = chain.pipe(
    lambda x: {"answer": x},   # str → {"answer": str}
    analysis_prompt,
    model,
    StrOutputParser(),
)

# 8. 실행
if __name__ == "__main__":
    result = composed_chain.invoke({"topic": "더블딥"})
    print(result)

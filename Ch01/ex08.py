from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# 1. 환경 변수 로드
load_dotenv()

# 2. 모델 설정
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=200,
)

# 3. 한국어 설명 프롬프트 체인
kor_prompt = ChatPromptTemplate.from_template(
    "주제 {topic}에 대해 짧은 설명을 해주세요."
)

kor_chain = (
    kor_prompt
    | model
    | StrOutputParser()
)

# 4. 영어 설명 프롬프트 체인
eng_prompt = ChatPromptTemplate.from_template(
    "주제 {topic}에 대해 짧게 영어로 설명해 주세요."
)

eng_chain = (
    eng_prompt
    | model
    | StrOutputParser()
)

# 5. 병렬 실행 체인 (RunnableParallel)
parallel_chain = RunnableParallel(
    kor=kor_chain,
    eng=eng_chain,
)

# 6. 실행
if __name__ == "__main__":
    result = parallel_chain.invoke({"topic": "더블딥"})

    print("한글 설명:", result["kor"])
    print("영어 설명:", result["eng"])

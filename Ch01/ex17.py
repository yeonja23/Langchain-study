from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers.json import SimpleJsonOutputParser

# 1) 환경변수 로드 (.env에 OPENAI_API_KEY 필요)
load_dotenv()

# 2) LLM 모델
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
)

# 3) JSON 형식 응답을 요구하는 프롬프트
json_prompt = PromptTemplate.from_template(
    "다음 질문에 대한 답변이 포함된 JSON 객체를 반환하십시오.\n"
    "질문: {question}"
)

# 4) JSON 파서 (검증 거의 없음, 그냥 JSON이면 OK)
json_parser = SimpleJsonOutputParser()

# 5) 체인 구성
json_chain = json_prompt | model | json_parser


if __name__ == "__main__":
    print("stream 결과:")
    chunks = list(
        json_chain.stream(
            {"question": "비트코인에 대한 짧은 한 문장 설명."}
        )
    )
    for chunk in chunks:
        print(chunk)

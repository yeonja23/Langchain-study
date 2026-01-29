from dotenv import load_dotenv

from langchain.output_parsers import RetryWithErrorOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

# 1) 환경변수 로드 (.env 에 OPENAI_API_KEY 있다고 가정)
load_dotenv()

# 2) "원래 파서" (JSON으로 파싱하고 싶다)
base_parser = JsonOutputParser()

# 3) "에러나면 다시 시도해주는 파서" (LLM을 이용해서 JSON 형태로 재작성 유도)
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
)

parser = RetryWithErrorOutputParser.from_llm(
    parser=base_parser,
    llm=model,
)

if __name__ == "__main__":
    # 4) 질문(프롬프트) + 일부러 "JSON이 아닌" 잘못된 모델 응답을 가정
    question = "가장 큰 대륙은?"
    # ai_response = "아시아입니다." # JSON 형식이 아닌 응답(일부러)
    ai_response = '{"answer": "아시아"}' # 성공

    # 5) parse_with_prompt: 응답 + 프롬프트를 함께 줘서,
    #    (1) 파싱 시도 -> 실패하면
    #    (2) LLM에게 '형식 맞춰 다시 써라'고 재시도 -> 성공하면 dict 반환
    try:
        result = parser.parse_with_prompt(ai_response, question)
        print("파싱 성공!")
        print(result)  # 예: {"continent": "아시아"}
    except Exception as e:
        print(f"오류 발생: {e}")

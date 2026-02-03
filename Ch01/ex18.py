from dotenv import load_dotenv

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field

# 1) 환경 변수 로드 (.env에 OPENAI_API_KEY 존재)
load_dotenv()


# 2) 원하는 출력 데이터 구조 정의 (Pydantic)
class FinancialAdvice(BaseModel):
    setup: str = Field(description="금융 조언 상황을 설정하기 위한 질문")
    advice: str = Field(description="질문을 해결하기 위한 금융 답변")


def main():
    # 3) OpenAI 모델 설정
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
    )

    # 4) JSON 출력 파서 (Pydantic 기반)
    parser = JsonOutputParser(pydantic_object=FinancialAdvice)

    # 5) 프롬프트 템플릿
    # - format_instructions: JSON 형식 강제 지침
    prompt = PromptTemplate(
        template=(
            "다음 금융 관련 질문에 답변해 주세요.\n"
            "{format_instructions}\n"
            "{query}\n"
        ),
        input_variables=["query"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        },
    )

    # 6) 체인 구성: 프롬프트 → 모델 → 파서
    chain = prompt | model | parser

    # 7) 실행
    result = chain.invoke(
        {
            "query": "부동산에 관련하여 금융 조언을 받을 수 있게 질문하라."
        }
    )

    # 8) 결과 출력
    print(result)
    print("\nsetup:", result["setup"])
    print("advice:", result["advice"])


if __name__ == "__main__":
    main()

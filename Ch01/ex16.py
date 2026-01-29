from dotenv import load_dotenv

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field, model_validator

# 1) 환경변수 로드
load_dotenv()


# 2) 원하는 출력 "데이터 구조" 정의 (Pydantic)
class FinancialAdvice(BaseModel):
    setup: str = Field(description="금융 조언 상황을 설정하기 위한 질문(물음표로 끝나야 함)")
    advice: str = Field(description="질문을 해결하기 위한 금융 답변")

    # Pydantic 검증 로직: setup이 ?로 끝나야 한다
    @model_validator(mode="before")
    @classmethod
    def question_ends_with_question_mark(cls, values: dict) -> dict:
        setup = values.get("setup", "")
        if not isinstance(setup, str) or not setup.endswith("?"):
            raise ValueError("잘못된 질문 형식입니다! setup은 '?'로 끝나야 합니다.")
        return values


def main():
    # 3) 모델
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
    )

    # 4) 파서 (LLM 출력 -> FinancialAdvice로 파싱/검증)
    parser = PydanticOutputParser(pydantic_object=FinancialAdvice)

    # 5) 프롬프트 템플릿 + format_instructions 삽입
    prompt = PromptTemplate(
        template=(
            "다음 금융 관련 질문에 답변해 주세요.\n"
            "{format_instructions}\n"
            "질문: {query}\n"
        ),
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # 6) 체인 구성 (프롬프트 -> 모델 -> 파서)
    chain = prompt | model | parser

    # 7) 실행
    try:
        result = chain.invoke({"query": "부동산 관련해서 금융 조언을 받을 수 있게 질문을 만들어줘"})
        # result는 FinancialAdvice 객체입니다.
        print("파싱 성공!")
        print(result)
        print("\nsetup:", result.setup)
        print("advice:", result.advice)

    except Exception as e:
        # 여기서 에러는 "파싱 실패/검증 실패" 같은 상황
        print(f"오류 발생: {e}")


if __name__ == "__main__":
    main()

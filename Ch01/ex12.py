from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

load_dotenv()

# 1) 예시 1개를 포맷하는 템플릿
example_prompt = PromptTemplate.from_template(
    "질문: {question}\n답변: {answer}"
)

# 2) 예시 데이터
examples = [
    {
        "question": "주식 투자와 예금 중 어느 것이 더 수익률이 높은가?",
        "answer": "일반적으로 주식 투자가 예금보다 수익률이 높습니다."
    },
    {
        "question": "부동산과 채권 중 어느 것이 더 안정적인 투자인가?",
        "answer": "일반적으로 채권이 부동산보다 안정적인 투자입니다."
    }
]

# 3) Few-shot 프롬프트 생성
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="질문: {input}",
    input_variables=["input"],
)

# 4) 프롬프트 확인 (LLM 호출 아님)
if __name__ == "__main__":
    result = prompt.invoke(
        {"input": "부동산 투자의 장점은 무엇인가?"}
    )
    print(result.to_string())

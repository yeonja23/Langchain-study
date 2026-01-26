from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

# 1) 예시 1개를 어떤 모양으로 찍을지 정의
example_prompt = PromptTemplate.from_template(
    "질문: {question}\n답변: {answer}"
)

# 2) 예시 데이터(질문/답변) 목록
examples = [
    {
        "question": "주식 투자와 예금 중 어느 것이 더 수익률이 높은가?",
        "answer": """
        후속 질문이 필요한가요: 네.
        후속 질문: 주식 투자의 평균 수익률은 얼마인가요?
        중간 답변: 주식 투자의 평균 수익률은 연 7%입니다.
        후속 질문: 예금의 평균 이자율은 얼마인가요?
        중간 답변: 예금의 평균 이자율은 연 1%입니다.
        따라서 최종 답변은: 주식 투자
        """
    },
    {
        "question": "부동산과 채권 중 어느 것이 더 안정적인 투자인가?",
        "answer": """
        후속 질문이 필요한가요: 네.
        후속 질문: 부동산 투자의 위험도는 어느 정도인가요?
        중간 답변: 부동산 투자의 위험도는 중간 수준입니다.
        후속 질문: 채권의 위험도는 어느 정도인가요?
        중간 답변: 채권의 위험도는 낮은 편입니다.
        따라서 최종 답변은: 채권
        """
    }
]

if __name__ == "__main__":
    # 3) 첫 번째 예시가 실제로 어떤 문자열이 되는지 확인
    print(example_prompt.invoke(examples[0]).to_string())

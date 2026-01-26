from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# 1) few-shot 예제 데이터
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

# 2) 예제 선택기 초기화
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=examples,
    embeddings=OpenAIEmbeddings(),
    vectorstore_cls=Chroma,
    k=1,
)

# 3) 입력 질문
question = "부동산 투자에 어떤 이점이 있나요?"

# 4) 가장 유사한 예제 선택
selected_examples = example_selector.select_examples(
    {"question": question}
)

# 5) 결과 출력
print(f"입력 질문: {question}")

for example in selected_examples:
    print("\n# 입력과 가장 유사한 예제:")
    for k, v in example.items():
        print(f"{k}: {v}")

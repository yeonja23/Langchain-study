from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

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

# 2) 예제 선택기 초기화 (질문과 유사한 예제 k개 자동 선택)
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=examples,
    embeddings=OpenAIEmbeddings(),
    vectorstore_cls=Chroma,
    k=1,
)

# 3) 예시 1개를 "질문/답변" 형식으로 포맷하는 템플릿
example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="질문: {question}\n답변: {answer}"
)

# 4) FewShotPromptTemplate: (유사한 예제 선택기 + prefix/suffix)로 최종 프롬프트 조립
prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="다음은 금융 관련 질문과 답변의 예입니다:",
    suffix="질문: {input}\n답변:",
    input_variables=["input"],
)

# 5) 모델 + 체인
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    max_tokens=300,
)

chain = prompt | model

# 6) 실행
if __name__ == "__main__":
    question = "부동산 투자에 어떤 이점이 있나요?"
    result = chain.invoke({"input": question})
    print(result.content)
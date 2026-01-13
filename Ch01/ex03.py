from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()  # .env 로드 (OPENAI_API_KEY)

# 1) 프롬프트 템플릿
prompt = ChatPromptTemplate.from_template(
    "주제 {topic}에 대해 짧은 설명을 해주세요."
)

# 2) 모델
model = ChatOpenAI(
    model="gpt-4o-mini",   # 교재의 text-davinci-002 대신 최신 chat 모델 사용
    temperature=0.7,       # 온도
    max_tokens=100,        # 최대 생성 토큰 수
    # top_p=1.0,           # 필요하면 추가 (기본값 보통 1.0)
    # frequency_penalty=0, # 필요하면 추가
    # presence_penalty=0,  # 필요하면 추가
)

# 3) 출력 파서
output_parser = StrOutputParser()

# 4) 체인 구성
chain = (
    {"topic": RunnablePassthrough()}
    | prompt
    | model
    | output_parser
)

if __name__ == "__main__":
    topic = input("설명할 주제를 입력하세요: ")
    result = chain.invoke(topic)
    print("\n[응답]")
    print(result)

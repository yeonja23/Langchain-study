from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=200,
)

prompt = ChatPromptTemplate.from_template(
    "주제 {topic}에 대해 짧은 설명을 해주세요."
)

output_parser = StrOutputParser()

chain = prompt | model | output_parser

if __name__ == "__main__":
    print("[스트리밍 시작]\n")

    for chunk in chain.stream({"topic": "더블딥"}):
        # chunk는 문자열 조각들이 순서대로 들어옵니다.
        print(chunk, end="", flush=True)

    print("\n\n[스트리밍 종료]")

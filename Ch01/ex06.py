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

analysis_prompt = ChatPromptTemplate.from_template(
    "이 답변을 영어로 번역해 주세요:\n{answer}"
)

composed_chain = (
    chain
    | (lambda x: {"answer": x})
    | analysis_prompt
    | model
    | StrOutputParser()
)

if __name__ == "__main__":
    result = composed_chain.invoke({"topic": "더블딥"})
    print(result)

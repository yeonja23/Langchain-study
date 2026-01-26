from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate

# 1. 환경 변수 로드 (이미 세팅돼 있으니 그대로)
load_dotenv()

# 2. 챗 프롬프트 템플릿 정의
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "당신은 유능한 금융 조언가입니다."),
    ("user", "주제 {topic}에 대해 금융 관련 조언을 해주세요.")
])

# 3. 프롬프트 템플릿 호출 (아직 LLM 호출 아님)
if __name__ == "__main__":
    prompt_value = prompt_template.invoke({"topic": "주식"})

    # 프롬프트가 어떻게 구성됐는지 확인
    print(prompt_value)

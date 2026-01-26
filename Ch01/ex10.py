from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

# 1. 환경 변수 로드
load_dotenv()

# 2. 메시지 자리 표시자를 포함한 챗 프롬프트 템플릿 정의
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "당신은 유능한 금융 조언가입니다."),
    MessagesPlaceholder("msgs")
])

# 3. 메시지 리스트를 msgs 자리에 전달하여 호출
if __name__ == "__main__":
    prompt_value = prompt_template.invoke({
        "msgs": [
            HumanMessage(content="안녕하세요!")
        ]
    })

    # 실제로 만들어진 메시지 구조 확인
    print(prompt_value)

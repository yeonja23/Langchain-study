from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 1) 환경변수 로드 (.env에 OPENAI_API_KEY가 있다고 가정)
load_dotenv()

def main():
    # 2) 모델 준비
    chat = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
    )

    # 3) 프롬프트 템플릿 정의
    # - system: 역할 부여
    # - placeholder: messages 자리에 "이전 대화 목록"을 그대로 끼워 넣음
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "당신은 금융 상담사입니다. 사용자에게 최선의 금융 조언을 제공합니다."),
            ("placeholder", "{messages}"),
        ]
    )

    # 4) 체인 구성 (프롬프트 -> 모델)
    chain = prompt | chat

    # 5) “이전 대화 이력”을 직접 messages로 전달
    #    여기서 ('human', ...), ('ai', ...)는 역할(role)입니다.
    ai_msg = chain.invoke(
        {
            "messages": [
                ("human", "저축을 늘리기 위해 무엇을 할 수 있나요?"),
                ("ai", "저축 목표를 설정하고, 매달 자동 이체로 일정 금액을 저축하세요."),
                ("human", "방금 뭐라고 했나요?"),
            ]
        }
    )

    # 6) 응답 출력
    print(ai_msg.content)

if __name__ == "__main__":
    main()

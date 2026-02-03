from dotenv import load_dotenv

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def main():
    # 1) 환경변수 로드 (.env의 OPENAI_API_KEY)
    load_dotenv()

    # 2) 모델 준비
    chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    # 3) 프롬프트 템플릿: system + placeholder(messages)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "당신은 금융 상담사입니다. 사용자에게 최선의 금융 조언을 제공합니다."),
            ("placeholder", "{messages}"),  # 여기로 대화 이력이 들어감
        ]
    )

    # 4) 체인 구성
    chain = prompt | chat

    # ChatMessageHistory로 대화 이력 관리
    chat_history = ChatMessageHistory()

    # 5) 대화 이력에 메시지 추가
    chat_history.add_user_message("저축을 늘리기 위해 무엇을 할 수 있나요?")
    chat_history.add_ai_message("저축 목표를 설정하고, 매달 자동 이체로 일정 금액을 저축하세요.")

    # 6) 새로운 질문을 추가하고, 전체 이력을 포함해서 다시 호출
    chat_history.add_user_message("방금 뭐라고 했나요?")

    ai_response = chain.invoke({"messages": chat_history.messages})

    # 7) 결과 출력
    print(ai_response.content)


if __name__ == "__main__":
    main()

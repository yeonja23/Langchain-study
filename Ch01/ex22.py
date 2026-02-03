from dotenv import load_dotenv

from langchain_core.messages import trim_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from langchain_community.chat_message_histories import ChatMessageHistory

from operator import itemgetter


# 1) 환경변수 로드 (.env에 OPENAI_API_KEY)
load_dotenv()


# 2) 메시지 트리밍 유틸리티 설정
# - strategy="last": 최근 메시지 중심으로 남김
# - max_tokens=2: "토큰" 기준으로 2만 남김
trimmer = trim_messages(strategy="last", max_tokens=2, token_counter=len)


# 3) 프롬프트
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 금융 상담사입니다. 모든 질문에 최선을 다해 답변하십시오."),
        ("placeholder", "{chat_history}"),  # 이전 대화 이력
        ("human", "{input}"),               # 사용자의 새로운 질문
    ]
)

# 4) 모델
chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# 6) “트리밍을 끼운” 체인 구성
# itemgetter("chat_history")로 입력 딕셔너리에서 대화이력 꺼냄
# 그 대화이력을 trimmer로 잘라서 prompt로 전달되도록 chat_history에 다시 넣어줌
chain_with_trimming = (
    RunnablePassthrough.assign(chat_history=itemgetter("chat_history") | trimmer)
    | prompt
    | chat
)


# 7) 대화 이력 저장소 (메모리)
chat_history = ChatMessageHistory()

# 8) RunnableWithMessageHistory로 “자동 저장/불러오기” 래핑
chain_with_trimmed_history = RunnableWithMessageHistory(
    chain_with_trimming,
    lambda session_id: chat_history,      # session_id에 해당하는 history 반환
    input_messages_key="input",           # 사용자의 입력이 들어있는 key
    history_messages_key="chat_history",  # 대화 이력이 들어갈 key
)


if __name__ == "__main__":
    # (1) 첫 질문
    res1 = chain_with_trimmed_history.invoke(
        {"input": "저는 5년 내에 집을 사기 위해 어떤 재정 계획을 세워야 하나요?"},
        {"configurable": {"session_id": "finance_session_1"}},
    )
    print(res1.content)

    # (2) 같은 session_id로 두 번째 질문 -> 이전 대화가 자동으로 포함됨(단, 트리밍됨)
    res2 = chain_with_trimmed_history.invoke(
        {"input": "내가 방금 뭐라고 했나요?"},
        {"configurable": {"session_id": "finance_session_1"}},
    )
    print(res2.content)

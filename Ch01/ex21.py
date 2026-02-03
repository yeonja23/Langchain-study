from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory

# 환경 변수 로드 (.env에 OPENAI_API_KEY)
load_dotenv()

# OpenAI 모델 설정
chat = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
)

# 시스템 메시지 + 대화 이력을 포함하는 프롬프트 템플릿
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 금융 상담사입니다. 모든 질문에 최선을 다해 답변하십시오."),
        ("placeholder", "{chat_history}"),  # 이전 대화 이력
        ("human", "{input}"),               # 사용자의 새로운 질문
    ]
)

# 프롬프트와 모델을 연결한 체인
chain = prompt | chat

# 대화 이력 저장소
chat_history = ChatMessageHistory()

# RunnableWithMessageHistory로 체인 감싸기
chain_with_message_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: chat_history,   # 세션 ID에 따른 대화 이력 반환
    input_messages_key="input",         # 사용자 입력 키
    history_messages_key="chat_history" # 대화 이력 키
)

# 첫 번째 질문
print("첫 번째 질문에 대한 답입니다.\n" \
"-------------------------")
response1 = chain_with_message_history.invoke(
    {"input": "저축을 늘리기 위해 무엇을 할 수 있나요?"},
    {"configurable": {"session_id": "unused"}},
)
print(response1.content)

# 두 번째 질문 (이전 대화 기억)
print("\n두 번째 질문에 대한 답입니다.\n" \
"-------------------------")

response2 = chain_with_message_history.invoke(
    {"input": "내가 방금 뭐라고 했나요?"},
    {"configurable": {"session_id": "unused"}},
)
print(response2.content)

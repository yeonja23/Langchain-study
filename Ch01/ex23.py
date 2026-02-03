from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 금융 상담사입니다. 모든 질문에 최선을 다해 답변하십시오."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)

chain = prompt | chat

# 메모리
chat_history = ChatMessageHistory()

# 자동 메모리 래핑
chain_with_message_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

def summarize_messages(chain_input):
    stored_messages = chat_history.messages
    if len(stored_messages) == 0:
        return False

    summarization_prompt = ChatPromptTemplate.from_messages(
        [
            ("placeholder", "{chat_history}"),
            ("user", "이전 대화를 요약해 주세요. 가능한 한 많은 세부 정보를 포함하십시오."),
        ]
    )

    summarization_chain = summarization_prompt | chat
    summary_message = summarization_chain.invoke({"chat_history": stored_messages})

    chat_history.clear()                 # 요약 후 이전 대화 삭제
    chat_history.add_message(summary_message)  # 요약 메시지만 추가

    return True

chain_with_summarization = (
    RunnablePassthrough.assign(messages_summarized=summarize_messages)
    | chain_with_message_history
)

# 먼저 대화 몇 번 쌓기
print(
    chain_with_summarization.invoke(
        {"input": "저축을 늘리기 위해 무엇을 할 수 있나요?"},
        {"configurable": {"session_id": "unused"}},
    ).content
)

print(
    chain_with_summarization.invoke(
        {"input": "구체적으로 자동이체는 어떻게 설정하는 게 좋을까요?"},
        {"configurable": {"session_id": "unused"}},
    ).content
)

# 요약된 대화를 바탕으로 새 질문
print(
    chain_with_summarization.invoke(
        {"input": "저에게 어떤 재정적 조언을 해주셨나요?"},
        {"configurable": {"session_id": "unused"}},
    ).content
)

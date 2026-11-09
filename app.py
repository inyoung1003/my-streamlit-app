
import json
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# 체인 캐싱
@st.cache_resource
def get_chain():
    s_msg = "당신은 주어진 규정을 기반으로 답변하는 규정 도우미입니다.\n\n[규정]\n\n{context}"
    h_msg = "{input}"
    messages = [("system", s_msg), ("human", h_msg)]
    prompt = ChatPromptTemplate.from_messages(messages)
    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True, temperature=0)
    return prompt | llm

# 파일 읽어오기
@st.cache_data
def load_docs(path="data/docs.jsonl"):
    data_list = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data_list.append(json.loads(line))
    return data_list

# 하나의 문자열로 변환
@st.cache_data
def format_docs(data_list):
    return "\n\n".join(
        f"[page={rec.get('metadata', {}).get('page', '?')}]\n{rec.get('page_content', '')}"
        for rec in data_list
    )

# 데이터 로드 및 모델 준비
raw_data = load_docs()
context = format_docs(raw_data)
chain = get_chain()

# 페이지 설정
st.set_page_config(page_title="RAG 챗봇", page_icon="🏢")
st.title("🏢 인사규정 챗봇")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 대화 출력
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# 사용자 입력
user_input = st.chat_input("메시지를 입력하세요.")

if user_input:
    # 사용자 메시지 출력
    st.chat_message("user").markdown(user_input)

    # 응답 출력
    with st.chat_message("assistant"):
        response_stream = chain.stream(
            {
                "input": user_input,
                "context": context
            }
        )
        full_response = st.write_stream(response_stream)

    # 대화 기록 저장
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.session_state.messages.append(AIMessage(content=full_response))


import os
import uuid
import base64
from typing import Literal

import streamlit as st
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# 상담 티켓 JSON 구조
class SupportTicket(BaseModel):
    product_model: str | None = Field(default=None, description="문의 대상 제품 모델명 (대화에 없으면 null)")
    category: Literal["제품 오류", "필터/소모품", "앱 연결", "사용 방법", "기타"] = Field(description="문의 유형")
    core_issue: str = Field(description="고객 문의의 핵심 내용을 한 문장으로 요약")
    sentiment: Literal["긍정", "중립", "부정"] = Field(description="상담 중 고객의 전반적인 감정")
    urgency: int = Field(description="처리 긴급도 (1~5점, 5점이 가장 긴급)", ge=1, le=5)
    resolved: bool = Field(description="상담 중 매뉴얼 안내만으로 문제가 해결되었는지 여부")
    needs_agent: bool = Field(description="상담원의 후속 연락이나 조치가 필요한지 여부")
    next_action: str = Field(description="상담원이 이어서 해야 할 조치 (필요 없으면 '없음')")

# 매뉴얼 읽기 (앱 시작 시 한 번만 실행)
@st.cache_resource
def load_manual():
    with open("manual.txt", "r", encoding="utf-8") as f:
        return f.read()

# 체인 생성 (앱 시작 시 한 번만 실행)
@st.cache_resource
def create_chains():
    # 상담 체인
    s_msg = """당신은 클린웨이브 고객센터의 상담원입니다.
아래 [제품 매뉴얼]에 있는 내용만 근거로 답변하세요.

[답변 규칙]
1. 친절하고 간결하게 3~5문장 이내로 답변합니다.
2. 조치 방법에 순서가 있으면 번호를 붙여 안내합니다.
3. 질문 내용이 매뉴얼에 없을 때만(수리비, 환불, 배송, 타사 제품 비교 등) 추측하지 말고 "담당 부서 확인 후 안내드리겠습니다."라고 답합니다.
   - 매뉴얼로 답할 수 있는 질문에는 이 문장을 붙이지 않습니다.
   - 한 질문에 두 가지가 섞여 있으면 매뉴얼에 있는 부분은 답하고, 없는 부분에 대해서만 이 문장을 씁니다.
4. 고객이 불편을 표현하면 먼저 공감하는 말을 한 문장 덧붙입니다.

[제품 매뉴얼]
{manual}"""
    messages = [("system", s_msg), MessagesPlaceholder("history"), ("human", "{question}")]
    chat_chain = ChatPromptTemplate.from_messages(messages) | ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    # 이미지 판독 체인
    s_msg = "당신은 가전제품 사진과 앱 화면을 판독하는 전문가입니다. 이미지에 보이는 사실만 객관적으로 기술하세요."
    h_msg = [
        {"type": "text", "text": "이미지에서 확인되는 제품 모델명, 화면에 표시된 코드나 문구, 켜져 있는 표시등과 그 색상을 항목별로 정리하세요. 원인 추측이나 해결 방법은 쓰지 마세요."},
        {"type": "image_url", "image_url": {"url": "{image_url}"}},
    ]
    image_chain = ChatPromptTemplate.from_messages([("system", s_msg), ("human", h_msg)]) | ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 티켓 생성 체인
    s_msg = "당신은 고객 상담 내용을 분석해 상담 티켓을 작성하는 CS 품질 관리 전문가입니다."
    h_msg = "다음 상담 대화를 분석해 상담 티켓을 작성하세요.\n\n{conversation}"
    ticket_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    ticket_chain = ChatPromptTemplate.from_messages([("system", s_msg), ("human", h_msg)]) | ticket_llm.with_structured_output(SupportTicket)

    return chat_chain, image_chain, ticket_chain

# 페이지 설정
st.set_page_config(page_title="클린웨이브 고객센터", page_icon="🌀")
st.title("🌀 클린웨이브 고객센터")
st.caption("AP-300 공기청정기 상담 챗봇")

manual = load_manual()
chat_chain, image_chain, ticket_chain = create_chains()

# 세션 상태 초기화
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.history = InMemoryChatMessageHistory()
    st.session_state.last_image_id = None
    st.session_state.ticket = None

# 대화를 기억하는 체인 (히스토리는 st.session_state에 저장)
chain = RunnableWithMessageHistory(
    chat_chain,
    lambda session_id: st.session_state.history,
    input_messages_key="question",
    history_messages_key="history"
)

# 사이드바: 이미지 첨부, 상담 종료
with st.sidebar:
    uploaded = st.file_uploader("제품 사진 / 화면 캡처 첨부", type=["png", "jpg", "jpeg"])
    if uploaded:
        st.image(uploaded)

    st.divider()
    if st.button("상담 종료 및 티켓 생성", type="primary"):
        conversation = ""
        for msg in st.session_state.history.messages:
            if msg.type == "human":
                conversation += f"고객: {msg.content}\n"
            else:
                conversation += f"상담원: {msg.content}\n"
        if conversation:
            with st.spinner("티켓 생성 중..."):
                st.session_state.ticket = ticket_chain.invoke({"conversation": conversation})

    if st.session_state.ticket:
        st.subheader("📋 상담 티켓")
        st.json(st.session_state.ticket.model_dump())

    if st.button("새 상담 시작"):
        st.session_state.clear()
        st.rerun()

# 기존 대화 출력
for msg in st.session_state.history.messages:
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# 사용자 입력
user_input = st.chat_input("문의 내용을 입력하세요...")

if user_input:
    question = user_input

    # 새로 첨부된 이미지가 있으면 판독 결과를 질문에 붙이기 (같은 이미지는 한 번만 판독)
    if uploaded and uploaded.file_id != st.session_state.last_image_id:
        b64 = base64.b64encode(uploaded.getvalue()).decode("utf-8")
        data_url = f"data:{uploaded.type};base64,{b64}"
        with st.spinner("이미지 판독 중..."):
            image_result = image_chain.invoke({"image_url": data_url})
        question = f"[고객 첨부 이미지 판독 결과]\n{image_result.content}\n\n[고객 질문]\n{user_input}"
        st.session_state.last_image_id = uploaded.file_id

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("답변 작성 중..."):
            cfg = {"configurable": {"session_id": st.session_state.session_id}}
            response = chain.invoke({"manual": manual, "question": question}, config=cfg)
        st.markdown(response.content)

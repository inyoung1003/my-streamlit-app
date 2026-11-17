
import os, io, datetime as dt
from operator import itemgetter
from typing import List

import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# PDF (reportlab 내장 한글 폰트 사용 → 글꼴 다운로드·설치 없음)
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from xml.sax.saxutils import escape

# PPT
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR

# ─────────────────────────────────────────── 페이지 설정 (반드시 첫 Streamlit 명령)
st.set_page_config(page_title="AIVLE School 학습 도우미", page_icon="📘", layout="wide")

# ─────────────────────────────────────────── 기본 설정
CHROMA_DIR = "chroma_db"          # Colab에서 만들 때와 같아야 함
COLLECTION = "aivle_docs"
FOLDER = "/content/drive/MyDrive/project04/"
PDF_PATH = FOLDER + "AIVLE School 백서.pdf"   # 벡터 DB가 비어 있을 때만 사용
MODEL = "gpt-4o-mini"

# API 키: 환경변수에 없으면 api_key.txt에서 읽기
if not os.environ.get("OPENAI_API_KEY") and os.path.exists(FOLDER + "api_key.txt"):
    for line in open(FOLDER + "api_key.txt"):
        if "=" in line:
            k, v = line.strip().split("=", 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")


# 문서 포맷팅 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ─────────────────────────────────────────── 벡터 저장소 / 검색기
@st.cache_resource
def get_vectorstore():
    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory=CHROMA_DIR,
    )
    # 저장된 데이터가 없으면 백서 PDF로 새로 만들기
    if vectorstore._collection.count() == 0 and os.path.exists(PDF_PATH):
        from langchain_pymupdf4llm import PyMuPDF4LLMLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        docs = [d for d in PyMuPDF4LLMLoader(PDF_PATH).load() if d.page_content.strip()]
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
        vectorstore.add_documents(chunks)
    return vectorstore


@st.cache_resource
def get_retriever():
    return get_vectorstore().as_retriever(search_kwargs={"k": 3})


# 체인 캐싱
@st.cache_resource
def get_chain():
    retriever = get_retriever()
    s_msg = "당신은 주어진 문서를 기반으로 답변하는 AIVLE School 학습 도우미입니다.\n\n[문서]\n\n{context}"
    h_msg = "{input}"
    messages = [("system", s_msg), ("human", h_msg)]
    prompt = ChatPromptTemplate.from_messages(messages)
    llm = ChatOpenAI(model=MODEL, streaming=True, temperature=0)
    rag_chain = (
        {
            "context": itemgetter("input") | retriever | format_docs,
            "input": itemgetter("input")
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def search_context(queries, k=4, max_chars=9000):
    """여러 질문으로 백서를 검색해 중복 없이 합친 근거 텍스트"""
    vs = get_vectorstore()
    seen, parts = set(), []
    for q in queries:
        if not q:
            continue
        for d in vs.similarity_search(q, k=k):
            if d.page_content[:200] not in seen:
                seen.add(d.page_content[:200])
                parts.append(d.page_content)
    return "\n\n---\n\n".join(parts)[:max_chars]


# ─────────────────────────────────────────── 구조화 출력 스키마
class TimeBlock(BaseModel):
    time: str = Field(description="시간대, 예: 19:30~21:00")
    activity: str = Field(description="할 일(짧게)")
    detail: str = Field(description="구체적인 실행 내용 한 문장")


class WeekPlan(BaseModel):
    week: int = Field(description="주차 번호(1부터)")
    theme: str = Field(description="이번 주 주제")
    goals: List[str] = Field(description="이번 주 목표 2~3개")
    tasks: List[str] = Field(description="이번 주 할 일 3~5개, 분량이 드러나게")
    checkpoint: str = Field(description="주말에 확인할 점검 기준 한 문장")


class StudyPlan(BaseModel):
    summary: str = Field(description="계획 요약 2~3문장")
    weekday_schedule: List[TimeBlock] = Field(description="평일 하루 시간표(교육 시간 포함, 시간순)")
    weekend_schedule: List[TimeBlock] = Field(description="주말 하루 시간표(시간순)")
    weekly_plan: List[WeekPlan] = Field(description="주차별 계획")
    tips: List[str] = Field(description="실천 팁 3~5개")


class KeyNumber(BaseModel):
    value: str = Field(description="입력에 실제로 있는 숫자·성과(예: '32%', '대상')")
    label: str = Field(description="그 숫자의 의미")


class Strength(BaseModel):
    title: str = Field(description="행동 중심 강점 이름")
    evidence: str = Field(description="근거 한 줄(프로젝트 이름이나 숫자)")


class ProjectSummary(BaseModel):
    project_name: str
    project_type: str
    period: str
    title_sentence: str = Field(description="결론이 드러나는 제목 한 문장")
    one_line_result: str = Field(description="대표 성과 한 줄")
    my_role: str = Field(description="맡은 범위")
    role_reason: str = Field(description="그 역할·방법을 고른 이유")
    tech_stack: List[str]
    problem: List[str] = Field(description="문제 정의 정확히 3줄")
    process: List[str] = Field(description="해결 과정 3~4줄, 막힌 지점-원인-바꾼 방법 포함")
    results: List[str] = Field(description="결과 1~3줄, 가능하면 전후 비교")
    retrospective: str = Field(description="회고 1~2문장")
    keywords: List[str] = Field(description="직무 키워드 3~5개")


class PortfolioDoc(BaseModel):
    slogan: str = Field(description="어떤 문제를 어떻게 푸는 사람인지 보여주는 슬로건 한 줄")
    key_numbers: List[KeyNumber] = Field(description="입력에 있는 대표 숫자 0~3개, 없으면 빈 리스트")
    strengths: List[Strength] = Field(description="핵심 강점 정확히 3개")
    projects: List[ProjectSummary]


# ─────────────────────────────────────────── PDF 만들기 (reportlab 내장 한글 폰트)
pdfmetrics.registerFont(UnicodeCIDFont("HYGothic-Medium"))
F = "HYGothic-Medium"
NAVY, ACCENT, LIGHT, GRAY = "1F3864", "2E75B6", "EEF3FA", "595959"
S = {
    "title": ParagraphStyle("t", fontName=F, fontSize=20, leading=26, textColor=colors.HexColor("#" + NAVY), spaceAfter=6),
    "sub": ParagraphStyle("s", fontName=F, fontSize=10.5, leading=15, textColor=colors.HexColor("#" + GRAY), spaceAfter=10),
    "h1": ParagraphStyle("h1", fontName=F, fontSize=14, leading=19, textColor=colors.HexColor("#" + NAVY), spaceBefore=12, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName=F, fontSize=11.5, leading=16, textColor=colors.HexColor("#" + ACCENT), spaceBefore=8, spaceAfter=3),
    "body": ParagraphStyle("b", fontName=F, fontSize=10, leading=15.5, spaceAfter=3),
    "bul": ParagraphStyle("bl", fontName=F, fontSize=10, leading=15, leftIndent=12, bulletIndent=2, spaceAfter=2),
    "cell": ParagraphStyle("c", fontName=F, fontSize=9, leading=13),
    "cellh": ParagraphStyle("ch", fontName=F, fontSize=9, leading=13, textColor=colors.white),
    "box": ParagraphStyle("x", fontName=F, fontSize=12.5, leading=18, textColor=colors.HexColor("#" + NAVY),
                          backColor=colors.HexColor("#" + LIGHT), borderPadding=8, spaceBefore=6, spaceAfter=14),
    "small": ParagraphStyle("sm", fontName=F, fontSize=8.5, leading=12, textColor=colors.HexColor("#" + GRAY)),
}
CW = A4[0] - 36 * mm


def _p(t, s="body"):
    return Paragraph(escape(str(t or "")).replace("\n", "<br/>"), S[s])


def _bullets(items):
    return [Paragraph(escape(str(t)), S["bul"], bulletText="•") for t in items if str(t).strip()]


def _table(header, rows, widths):
    t = Table([[_p(h, "cellh") for h in header]] + [[_p(c, "cell") for c in r] for r in rows],
              colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#" + NAVY)),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F8FC")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BFCBDA")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def _pdf(story, title):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title=title, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm)

    def footer(c, d):
        c.setFont(F, 8)
        c.drawCentredString(A4[0] / 2, 10 * mm, f"AIVLE School 학습 도우미 · {d.page}")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


def plan_pdf(plan, prof):
    st_ = [_p("개인 학습 계획", "title"),
           _p(f"{prof['track']} · {prof['stage']} · {prof['weeks']}주 · 작성일 {dt.date.today():%Y-%m-%d}", "sub"),
           _p(f"학습 목표: {prof['goal']}", "box"), _p("계획 요약", "h1"), _p(plan["summary"])]
    for key, title in (("weekday_schedule", "평일 하루 계획"), ("weekend_schedule", "주말 하루 계획")):
        st_ += [_p(title, "h1"), _table(["시간", "할 일", "세부 내용"],
                                        [[b["time"], b["activity"], b["detail"]] for b in plan[key]],
                                        [28 * mm, 45 * mm, CW - 73 * mm])]
    st_.append(_p("주간 계획", "h1"))
    for w in plan["weekly_plan"]:
        st_.append(KeepTogether([_p(f"{w['week']}주차 · {w['theme']}", "h2"), _p("목표")] + _bullets(w["goals"])
                                + [_p("할 일")] + _bullets(w["tasks"])
                                + [_p(f"점검 기준: {w['checkpoint']}", "small"), Spacer(1, 4)]))
    st_ += [_p("실천 팁", "h1")] + _bullets(plan["tips"])
    return _pdf(st_, "개인 학습 계획")


def portfolio_pdf(doc, author):
    st_ = [_p("프로젝트 정리 문서", "title"),
           _p(f"{author['name']} · {author['job']} · 작성일 {dt.date.today():%Y-%m-%d}", "sub"),
           _p(doc["slogan"], "box"), _p("한눈에 보기", "h1"),
           _table(["항목", "내용"], [
               ["슬로건", doc["slogan"]],
               ["대표 숫자", " / ".join(f"{k['value']} ({k['label']})" for k in doc["key_numbers"]) or "-"],
               ["핵심 강점", "\n".join(f"• {s['title']}: {s['evidence']}" for s in doc["strengths"])],
               ["대표 프로젝트", "\n".join(f"• {p['project_name']}: {p['one_line_result']}" for p in doc["projects"])],
               ["링크", "\n".join(author["links"]) or "-"]], [32 * mm, CW - 32 * mm])]
    for i, p in enumerate(doc["projects"], 1):
        st_ += [PageBreak(), _p(f"프로젝트 {i}. {p['project_name']}", "h1"), _p(p["title_sentence"], "box"),
                _table(["구분", "내용"], [["유형·기간", f"{p['project_type']} · {p['period']}"],
                                         ["내 역할", p["my_role"]], ["선택 이유", p["role_reason"]],
                                         ["사용 기술", ", ".join(p["tech_stack"]) or "-"]], [32 * mm, CW - 32 * mm]),
                _p("문제 정의", "h2")] + _bullets(p["problem"])
        st_ += [_p("해결 과정", "h2")] + _bullets(p["process"])
        st_ += [_p("결과", "h2")] + _bullets(p["results"])
        st_ += [_p("회고", "h2"), _p(p["retrospective"]), _p("키워드: " + " · ".join(p["keywords"]), "small")]
    return _pdf(st_, "프로젝트 정리 문서")


# ─────────────────────────────────────────── PPT 만들기 (PC에 설치된 맑은 고딕으로 표시)
def _rgb(h):
    return RGBColor.from_string(h)


def _box(s, x, y, w, h, fill=None, line=None):
    shp = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    if fill:
        shp.fill.solid(); shp.fill.fore_color.rgb = _rgb(fill)
    else:
        shp.fill.background()
    if line:
        shp.line.color.rgb = _rgb(line)
    else:
        shp.line.fill.background()
    shp.shadow.inherit = False


def _txt(s, x, y, w, h, paras, anchor=MSO_ANCHOR.TOP):
    tf = s.shapes.add_textbox(x, y, w, h).text_frame
    tf.word_wrap, tf.vertical_anchor = True, anchor
    for i, (t, size, bold, color, bullet) in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run()
        r.text = ("• " if bullet else "") + str(t)
        r.font.name, r.font.size, r.font.bold, r.font.color.rgb = "맑은 고딕", Pt(size), bold, _rgb(color)
        p.space_after = Pt(4)


def _sec(s, x, y, w, h, title, items, size=11):
    _box(s, x, y, w, h, fill="F7F9FC", line="D6DEEA")
    _box(s, x, y, Inches(0.06), h, fill=ACCENT)
    _txt(s, x + Inches(0.12), y + Inches(0.05), w - Inches(0.2), h - Inches(0.1),
         [(title, 13, True, NAVY, False)] + [(t, size, False, "262626", True) for t in items])


def portfolio_pptx(doc, author):
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    W, H, blank = prs.slide_width, prs.slide_height, prs.slide_layouts[6]

    s = prs.slides.add_slide(blank)                       # 표지
    _box(s, 0, 0, W, H, fill=NAVY)
    _txt(s, Inches(0.8), Inches(1.9), Inches(11.5), Inches(1.6), [("프로젝트 정리 문서", 40, True, "FFFFFF", False)], MSO_ANCHOR.BOTTOM)
    _txt(s, Inches(0.8), Inches(3.8), Inches(11.5), Inches(1.4), [(doc["slogan"], 20, False, "DCE6F2", False)])
    _txt(s, Inches(0.8), Inches(6.2), Inches(11.5), Inches(0.6),
         [(f"{author['name']}  |  {author['job']}  |  {dt.date.today():%Y.%m.%d}", 14, False, "BFD1EA", False)])

    s = prs.slides.add_slide(blank)                       # 한눈에 보기
    _box(s, 0, 0, W, Inches(0.9), fill=NAVY)
    _txt(s, Inches(0.5), Inches(0.15), Inches(12), Inches(0.6), [("한눈에 보기", 24, True, "FFFFFF", False)])
    _txt(s, Inches(0.5), Inches(1.1), Inches(12.3), Inches(0.8), [(doc["slogan"], 18, True, NAVY, False)])
    nums, top = doc["key_numbers"][:3], Inches(2.0)
    if nums:
        cw = Inches(12.3 / len(nums))
        for i, k in enumerate(nums):
            x = Inches(0.5) + cw * i
            _box(s, x + Inches(0.05), Inches(1.95), cw - Inches(0.1), Inches(1.35), fill=LIGHT)
            _txt(s, x + Inches(0.05), Inches(2.0), cw - Inches(0.1), Inches(0.8), [(k["value"], 30, True, ACCENT, False)], MSO_ANCHOR.MIDDLE)
            _txt(s, x + Inches(0.05), Inches(2.8), cw - Inches(0.1), Inches(0.45), [(k["label"], 12, False, GRAY, False)])
        top = Inches(3.5)
    hh = H - top - Inches(0.4)
    _sec(s, Inches(0.5), top, Inches(4.0), hh, "핵심 강점", [f"{x['title']}: {x['evidence']}" for x in doc["strengths"]])
    _sec(s, Inches(4.65), top, Inches(5.0), hh, "대표 프로젝트",
         [f"{p['project_name']}: {p['one_line_result']} (p.{i + 3})" for i, p in enumerate(doc["projects"])])
    _sec(s, Inches(9.8), top, Inches(3.0), hh, "링크", author["links"] or ["-"], 10)

    for i, p in enumerate(doc["projects"], 1):            # 프로젝트별 1장
        s = prs.slides.add_slide(blank)
        _box(s, 0, 0, W, Inches(1.25), fill=NAVY)
        _txt(s, Inches(0.5), Inches(0.1), Inches(12.3), Inches(0.5),
             [(f"PROJECT {i}  ·  {p['project_name']}  ·  {p['project_type']}  ·  {p['period']}", 13, False, "BFD1EA", False)])
        _txt(s, Inches(0.5), Inches(0.5), Inches(12.3), Inches(0.7), [(p["title_sentence"], 20, True, "FFFFFF", False)])
        _box(s, Inches(0.5), Inches(1.4), Inches(12.3), Inches(0.62), fill=LIGHT)
        _txt(s, Inches(0.6), Inches(1.42), Inches(12.1), Inches(0.6),
             [(f"내 역할: {p['my_role']}  |  선택 이유: {p['role_reason']}", 11, False, NAVY, False)], MSO_ANCHOR.MIDDLE)
        cw, top, h = Inches(6.05), Inches(2.2), Inches(2.45)
        _sec(s, Inches(0.5), top, cw, h, "문제 정의", p["problem"])
        _sec(s, Inches(6.75), top, cw, h, "해결 과정", p["process"])
        _sec(s, Inches(0.5), top + h + Inches(0.15), cw, Inches(2.0), "결과", p["results"])
        _sec(s, Inches(6.75), top + h + Inches(0.15), cw, Inches(2.0), "회고", [p["retrospective"]])
        _txt(s, Inches(0.5), H - Inches(0.5), Inches(12.3), Inches(0.4),
             [("기술: " + (", ".join(p["tech_stack"]) or "-") + "   |   키워드: " + " · ".join(p["keywords"]), 10, False, GRAY, False)])

    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


# ═══════════════════════════════════════════ 화면
st.title("📘 AIVLE School 학습 도우미")

if not os.environ.get("OPENAI_API_KEY"):
    st.error("OpenAI API 키가 없습니다. api_key.txt 위치를 확인하세요.")
    st.stop()

# RAG 체인 준비
chain = get_chain()
if get_vectorstore()._collection.count() == 0:
    st.warning("벡터 DB가 비어 있습니다. chroma_db 폴더 또는 백서 PDF 경로(PDF_PATH)를 확인하세요.")

tab_chat, tab_plan, tab_doc = st.tabs(["💬 학습 도우미", "📅 개인 학습 계획", "📝 AI 문서 작성"])

# ─────────────────────────────────────────── 탭 1. 채팅 (기존 코드)
with tab_chat:
    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 기존 대화 내용 출력
    for message in st.session_state.messages:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

    # 사용자 입력 처리
    user_input = st.chat_input("메시지를 입력하세요.")
    if user_input:
        st.chat_message("user").markdown(user_input)
        with st.chat_message("assistant"):
            response_stream = chain.stream({"input": user_input})
            full_response = st.write_stream(response_stream)
        st.session_state.messages.append(HumanMessage(content=user_input))
        st.session_state.messages.append(AIMessage(content=full_response))

# ─────────────────────────────────────────── 탭 2. 개인 학습 계획
with tab_plan:
    st.subheader("📅 개인 학습 계획 만들기")
    with st.form("plan_form"):
        c1, c2, c3 = st.columns(3)
        track = c1.radio("트랙", ["DX 컨설턴트", "AI 개발자"], horizontal=True)
        stage = c2.selectbox("현재 단계", ["교육 시작 전", "STEP1 (분석형·생성형 AI)", "STEP2 (Cloud·서비스 개발/제안)",
                                         "STEP3 (빅프로젝트)", "수료 전후 (취업 준비)"])
        weeks = c3.slider("계획 기간(주)", 1, 8, 4)
        goal = st.text_area("학습 목표 *", placeholder="예) 4주 안에 AICE Associate 합격, SQL 기본기 다지기")
        c4, c5 = st.columns(2)
        wd_h = c4.number_input("평일 개인 학습 가능 시간 (교육 시간 외, 시간/일)", 0.0, 6.0, 2.0, 0.5)
        we_h = c5.number_input("주말 학습 가능 시간 (시간/일)", 0.0, 12.0, 4.0, 0.5)
        focus = st.multiselect("함께 준비할 항목", ["코딩마스터스", "코딩테스트", "AICE", "AX챌린지 자격증", "SQLD",
                                                "빅데이터분석기사", "공모전", "스터디", "포트폴리오", "자기소개서·면접"])
        level = st.text_area("현재 수준·강점·보완점", placeholder="예) 비전공자, 파이썬 기초는 됨, SQL이 약함")
        ok = st.form_submit_button("📅 학습 계획 생성", type="primary")

    if ok and not goal.strip():
        st.warning("학습 목표를 입력해 주세요.")
    elif ok:
        prof = {"track": track, "stage": stage, "weeks": weeks, "goal": goal}
        with st.spinner("백서를 참고해 계획을 만드는 중..."):
            context = search_context(["개인 학습 전략 하루 루틴", f"{stage} 단계별 전략", "일일 타임테이블",
                                      f"{track} 트랙", goal, " ".join(focus), "배경별 팁"])
            prompt = ChatPromptTemplate.from_messages([
                ("system", """당신은 AIVLE School 학습 코치입니다. [백서 근거]와 학습자 정보로 실행 가능한 학습 계획을 만드세요.
규칙:
1. 평일 09:00~18:00은 교육 시간입니다(체크인 08:50~09:10, 체크아웃 17:50~18:10). 평일 시간표에 '교육 참여' 블록으로 넣고, 개인 학습은 교육 시간 밖에만 배치하세요.
2. 개인 학습 블록의 합은 평일 하루 {wd_h}시간, 주말 하루 {we_h}시간을 넘지 마세요.
3. 휴식·식사 시간을 넣어 무리하지 않게 하세요.
4. 주간 계획은 정확히 {weeks}개 주차로, 할 일에는 분량(문제 수·시간·회차)을 적으세요.
5. 백서에 없는 사실·날짜·수치는 만들지 마세요. 한국어로 작성하세요.

[백서 근거]
{context}"""),
                ("human", "트랙: {track}\n단계: {stage}\n목표: {goal}\n기간: {weeks}주\n준비 항목: {focus}\n현재 수준: {level}"),
            ])
            plan = (prompt | ChatOpenAI(model=MODEL, temperature=0.3).with_structured_output(StudyPlan)).invoke({
                "context": context, "track": track, "stage": stage, "goal": goal, "weeks": weeks,
                "wd_h": wd_h, "we_h": we_h, "focus": ", ".join(focus) or "없음", "level": level or "입력 없음",
            }).model_dump()
        st.session_state.plan, st.session_state.plan_prof = plan, prof
        st.session_state.plan_pdf = plan_pdf(plan, prof)

    if "plan" in st.session_state:
        plan, prof = st.session_state.plan, st.session_state.plan_prof
        cols = {"time": "시간", "activity": "할 일", "detail": "세부 내용"}
        st.divider()
        st.markdown(f"### 🎯 {prof['goal']}")
        st.info(plan["summary"])
        c1, c2 = st.columns(2)
        c1.markdown("#### 평일 하루 계획")
        c1.dataframe(pd.DataFrame(plan["weekday_schedule"]).rename(columns=cols), hide_index=True, use_container_width=True)
        c2.markdown("#### 주말 하루 계획")
        c2.dataframe(pd.DataFrame(plan["weekend_schedule"]).rename(columns=cols), hide_index=True, use_container_width=True)
        st.markdown("#### 주간 계획")
        for w in plan["weekly_plan"]:
            with st.expander(f"{w['week']}주차 · {w['theme']}", expanded=(w["week"] == 1)):
                st.markdown("**목표**\n" + "\n".join(f"- {g}" for g in w["goals"]))
                st.markdown("**할 일**\n" + "\n".join(f"- [ ] {t}" for t in w["tasks"]))
                st.caption(f"점검 기준: {w['checkpoint']}")
        st.markdown("#### 실천 팁\n" + "\n".join(f"- {t}" for t in plan["tips"]))
        st.download_button("⬇️ 학습 계획 PDF 다운로드", st.session_state.plan_pdf,
                           file_name=f"학습계획_{dt.date.today():%Y%m%d}.pdf", mime="application/pdf")

# ─────────────────────────────────────────── 탭 3. AI 문서 작성
with tab_doc:
    st.subheader("📝 프로젝트 정리 문서 만들기")
    st.caption("프로젝트를 하나씩 추가한 뒤 PDF·PPT로 정리합니다. (백서 9장 포트폴리오 작성 원칙 적용)")
    if "projects" not in st.session_state:
        st.session_state.projects = []

    with st.form("project_form", clear_on_submit=True):
        c1, c2, c3 = st.columns([2, 1, 1])
        name = c1.text_input("프로젝트 이름 *")
        ptype = c2.selectbox("유형", ["빅프로젝트", "미니프로젝트", "공모전", "해커톤", "스터디", "개인 프로젝트", "기타"])
        period = c3.text_input("기간", placeholder="2026.06~08 (8주)")
        c4, c5 = st.columns(2)
        team = c4.text_input("팀 구성", placeholder="6명 (DX 4, AI 2)")
        tech = c5.text_input("사용 기술·도구", placeholder="Python, Pandas, Streamlit")
        role = st.text_area("맡은 역할 *", placeholder="조장, 데이터 수집·분석, 최종 발표")
        task = st.text_area("과제 내용 (어떤 문제를 풀었나) *")
        process = st.text_area("수행 과정 (막힌 점과 해결 포함)")
        result = st.text_area("결과·성과 (숫자가 있으면 꼭 적기)")
        learned = st.text_area("과제를 통해 얻은 것 *")
        add = st.form_submit_button("➕ 프로젝트 추가")

    if add:
        if not all(x.strip() for x in (name, role, task, learned)):
            st.warning("필수 항목(*)을 모두 입력해 주세요.")
        else:
            st.session_state.projects.append({"프로젝트 이름": name, "유형": ptype, "기간": period, "팀 구성": team,
                                              "사용 기술": tech, "맡은 역할": role, "과제 내용": task,
                                              "수행 과정": process, "결과·성과": result, "얻은 것": learned})
            st.success(f"'{name}' 추가 완료")

    for i, p in enumerate(st.session_state.projects):
        c1, c2 = st.columns([10, 1])
        c1.markdown(f"**{i + 1}. {p['프로젝트 이름']}** · {p['유형']} · {p['기간']}")
        if c2.button("삭제", key=f"del_{i}"):
            st.session_state.projects.pop(i)
            st.rerun()

    st.divider()
    c1, c2 = st.columns(2)
    author_name = c1.text_input("이름", placeholder="홍길동")
    job = c2.text_input("지원 직무", value="DX 컨설턴트")
    links = st.text_input("링크 (쉼표로 구분)", placeholder="github.com/..., 블로그 주소")
    formats = st.multiselect("출력 형식", ["PDF", "PPT"], default=["PDF", "PPT"])

    if st.button("🪄 AI로 문서 작성", type="primary", disabled=not st.session_state.projects):
        author = {"name": author_name or "에이블러", "job": job,
                  "links": [x.strip() for x in links.split(",") if x.strip()]}
        projects_text = "\n\n".join(f"[프로젝트 {i + 1}]\n" + "\n".join(f"- {k}: {v or '입력 없음'}" for k, v in p.items())
                                    for i, p in enumerate(st.session_state.projects))
        with st.spinner("백서의 포트폴리오 원칙을 참고해 작성하는 중..."):
            context = search_context(["포트폴리오 작성 순서", "나를 차별화하는 구성 규칙 7가지",
                                      "자기소개 페이지 다섯 칸 슬로건", "포트폴리오 핵심 4가지 영역"])
            prompt = ChatPromptTemplate.from_messages([
                ("system", """당신은 AIVLE School 교육생의 프로젝트를 정리하는 포트폴리오 작성 도우미입니다.
[백서의 포트폴리오 원칙]을 따르되, 사실은 반드시 [사용자 입력]에서만 가져오세요.
규칙:
1. 입력에 없는 수치·성과·기술·역할을 만들지 마세요. key_numbers는 입력에 실제로 있는 숫자·수상만(없으면 빈 리스트).
2. 프로젝트마다 제목 한 문장 → 문제 정의 3줄 → 해결 과정 3~4줄 → 결과 → 회고 순서로 정리하세요.
3. 강점은 형용사 대신 일하는 방식과 근거(프로젝트 이름·숫자)로 쓰세요.
4. '{job}' 직무의 언어로, 간결한 개조식(~함, ~임)으로 쓰세요. 한 줄은 60자 안팎.
5. projects는 입력 순서대로 입력 개수만큼 만드세요.

[백서의 포트폴리오 원칙]
{context}"""),
                ("human", "[사용자 입력]\n지원 직무: {job}\n\n{projects}"),
            ])
            doc = (prompt | ChatOpenAI(model=MODEL, temperature=0.2).with_structured_output(PortfolioDoc)).invoke(
                {"context": context, "job": job, "projects": projects_text}).model_dump()
        st.session_state.portfolio = doc
        stamp = f"{dt.date.today():%Y%m%d}"
        st.session_state.files = {}
        if "PDF" in formats:
            st.session_state.files["PDF"] = (portfolio_pdf(doc, author), f"프로젝트정리_{stamp}.pdf", "application/pdf")
        if "PPT" in formats:
            st.session_state.files["PPT"] = (portfolio_pptx(doc, author), f"프로젝트정리_{stamp}.pptx",
                                             "application/vnd.openxmlformats-officedocument.presentationml.presentation")

    if "portfolio" in st.session_state:
        doc = st.session_state.portfolio
        st.success(f"💡 {doc['slogan']}")
        if doc["key_numbers"]:
            for col, k in zip(st.columns(len(doc["key_numbers"])), doc["key_numbers"]):
                col.metric(k["label"], k["value"])
        for p in doc["projects"]:
            with st.expander(f"📁 {p['project_name']} — {p['title_sentence']}"):
                st.markdown(f"**내 역할**: {p['my_role']}  \n**선택 이유**: {p['role_reason']}")
                for title, key in (("문제 정의", "problem"), ("해결 과정", "process"), ("결과", "results")):
                    st.markdown(f"**{title}**\n" + "\n".join(f"- {x}" for x in p[key]))
                st.markdown(f"**회고**: {p['retrospective']}")
        for col, (fmt, (data, fname, mime)) in zip(st.columns(max(len(st.session_state.files), 1)),
                                                   st.session_state.files.items()):
            col.download_button(f"⬇️ {fmt} 다운로드", data, file_name=fname, mime=mime)
        st.caption("※ AI가 정리한 초안입니다. 제출 전 내용을 꼭 확인하세요.")

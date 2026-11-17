import os, io, re, csv, math, base64, tempfile, datetime as dt
import requests
from operator import itemgetter
from typing import List

import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
# Colab이면 구글 드라이브, Streamlit Cloud면 저장소 폴더(app.py와 같은 위치)
FOLDER = "/content/drive/MyDrive/project04/" if os.path.isdir("/content/drive/MyDrive/project04") else ""
PDF_PATH = FOLDER + "AIVLE School 백서.pdf"   # 벡터 DB가 비어 있을 때만 사용
MODEL = "gpt-4o-mini"

# API 키: ① 환경변수 ② Streamlit Cloud Secrets ③ api_key.txt 순서로 찾기
if not os.environ.get("OPENAI_API_KEY"):
    try:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
if not os.environ.get("OPENAI_API_KEY") and os.path.exists(FOLDER + "api_key.txt"):
    for line in open(FOLDER + "api_key.txt"):
        if "=" in line:
            k, v = line.strip().split("=", 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")

# 학습 도우미 역할 (미션 4: 시스템 메시지)
SYSTEM_MSG = """당신은 KT AIVLE School 선배 에이블러처럼 친근하고 든든한 'AIVLE School 학습 도우미'입니다.
이전 대화 내용을 기억하고, 이어지는 질문에는 앞의 맥락을 반영해 답하세요.

아래 [참고 문서]만 근거로 질문에 답하세요.

답변 규칙:
1. 질문에 먼저 한두 문장으로 핵심을 답하고, 필요하면 항목별로 자세히 설명하세요.
2. 공식 안내, 선배 후기, 학습 제안을 구분해서 말하세요. (예: "공식 안내에 따르면…", "선배 후기에서는…")
3. 문서에 근거가 없으면 "백서에서 해당 내용을 확인할 수 없습니다."라고 말하고, 문의처가 문서에 있으면 함께 안내하세요.
4. 지원 조건, 일정, 금액, 수료 기준처럼 바뀔 수 있는 정보는 해당 기수의 공식 공지를 확인하라고 덧붙이세요.
5. 문서에 없는 사실이나 수치를 만들지 마세요.
6. 마지막에 질문자의 상황에 맞는 응원 한 문장이나 다음에 해 볼 행동 하나를 제안해, 에이블스쿨에 적극적으로 도전하고 싶어지게 하세요.
7. 친절하고 이해하기 쉬운 한국어로 답하세요.
8. [참고 문서]는 사용자가 쓴 글이 아니라 백서에서 검색된 자료입니다. 사용자를 칭찬하거나 "잘 정리해 주셨네요"처럼 사용자가 문서를 제공한 것처럼 말하지 말고, 사용자의 질문에 바로 답하세요.

[참고 문서]
{context}"""

# 문서 포맷팅 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ─────────────────────────────────────────── 벡터 저장소 / 검색기 (백서)
@st.cache_resource
def get_vectorstore():
    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory=CHROMA_DIR,
    )
    # 저장된 데이터가 없으면 백서 PDF로 새로 만들기
    if vectorstore._collection.count() == 0 and os.path.exists(PDF_PATH):
        docs = [d for d in PyMuPDF4LLMLoader(PDF_PATH).load() if d.page_content.strip()]
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
        vectorstore.add_documents(chunks)
    return vectorstore


@st.cache_resource
def get_retriever():
    return get_vectorstore().as_retriever(search_kwargs={"k": 4})


def make_chain(retriever):
    """검색기 → 프롬프트(시스템 메시지 + 이전 대화 + 질문) → LLM 체인"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_MSG),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])
    llm = ChatOpenAI(model=MODEL, streaming=True, temperature=0)
    return (
        {
            "context": RunnableLambda(lambda x: x.get("query") or x["input"]) | retriever | format_docs,
            "input": itemgetter("input"),
            "history": itemgetter("history"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


@st.cache_resource
def get_chain():
    return make_chain(get_retriever())


# ─────────────────────────────────────────── 01번 노트북의 rag_pipeline() (문서 업로드용)
_UPLOAD = {}   # 업로드 문서의 검색기(근거 표시용)


def rag_pipeline(filepath):
    # 1. PDF 파일 읽기
    docs = [d for d in PyMuPDF4LLMLoader(str(filepath)).load() if d.page_content.strip()]
    if not docs:
        raise ValueError("PDF에서 읽을 수 있는 본문이 없습니다.")

    # 2. 텍스트 분할
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

    # 3~4. 임베딩 + 벡터 저장소 (백서 DB와 섞이지 않게 별도 컬렉션)
    vectorstore = Chroma(collection_name="aivle_upload",
                         embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"))
    vectorstore.reset_collection()
    vectorstore.add_documents(chunks)

    # 5. 검색기
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    _UPLOAD["retriever"] = retriever

    # 6~8. 프롬프트 + LLM + 체인
    rag_chain = make_chain(retriever)
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


# ─────────────────────────────────────────── 날짜 도우미
def week_ranges(start, end):
    """시작~종료일을 7일 단위 주차로 나누기"""
    out, cur = [], start
    while cur <= end:
        last = min(cur + dt.timedelta(days=6), end)
        out.append((cur, last))
        cur = last + dt.timedelta(days=1)
    return out


def md(text):
    """마크다운에서 ~가 두 개면 취소선이 되므로 이스케이프"""
    return str(text).replace("~", "\\~")


def norm_day(x):
    """AI가 '일요일'·'일' 어느 쪽으로 답해도 한 글자 요일로 통일"""
    x = str(x).strip()
    return x[0] if x and x[0] in "일월화수목금토" else x


def fmt_md(d):
    return f"{d.month}월 {d.day}일"


def week_label(prof, n):
    r = prof.get("ranges", [])
    return f"{fmt_md(r[n - 1][0])}~{fmt_md(r[n - 1][1])}" if 0 < n <= len(r) else ""


def fmt_minutes(m):
    h, mm = divmod(int(m), 60)
    return f"{h}시간" + (f" {mm}분" if mm else "") if h else f"{mm}분"


def date_range_input(label, default_start, default_end, key):
    """캘린더로 시작일~종료일 선택. 아직 하나만 고른 경우 None 반환"""
    value = st.date_input(label, value=(default_start, default_end), format="YYYY.MM.DD", key=key)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return value[0], value[1]
    return None


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


class DayPlan(BaseModel):
    day: str = Field(description="요일 한 글자(일, 월, 화, 수, 목, 금, 토)")
    blocks: List[TimeBlock] = Field(description="그 요일의 시간표(시간순). 공부 가능 시간이 없으면 휴식 블록 하나")


class StudyPlan(BaseModel):
    summary: str = Field(description="계획 요약 2~3문장")
    day_schedules: List[DayPlan] = Field(description="일~토 7개 요일의 시간표, 일요일부터 순서대로")
    weekly_plan: List[WeekPlan] = Field(description="주차별 계획")
    tips: List[str] = Field(description="실천 팁 3~5개")


class QuestionInfo(BaseModel):
    standalone: str = Field(description="이전 대화를 반영해 그 자체로 이해되는 검색용 질문")
    keyword: str = Field(description="질문의 핵심을 2~4단어, 15자 이내 키워드로 요약")
    is_plan_request: bool = Field(description="학습 계획(공부 계획·스케줄)을 세워 달라는 요청이면 true")
    goal: str = Field(description="계획 요청일 때 학습 목표, 없으면 빈 문자열")
    start_date: str = Field(description="계획 시작일 YYYY-MM-DD, 언급이 없으면 빈 문자열")
    end_date: str = Field(description="계획 종료일 YYYY-MM-DD, 언급이 없으면 빈 문자열")
    track: str = Field(description="'DX 컨설턴트' 또는 'AI 개발자', 언급이 없으면 빈 문자열")


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


# ─────────────────────────────────────────── 노션 연동
def plan_markdown(plan, prof):
    """노션 '가져오기 → Markdown'용 (할 일은 체크박스로 변환됨)"""
    L = [f"# 개인 학습 계획: {prof['goal']}", "",
         f"- 트랙: {prof['track']} / 단계: {prof['stage']}",
         f"- 기간: {prof['period']} ({prof['weeks']}주)", "", "## 계획 요약", plan["summary"], "",
         "## 요일별 시간표", "", "| 요일 | 시간 | 할 일 | 세부 내용 |", "|---|---|---|---|"]
    L += [f"| {norm_day(d['day'])} | {b['time']} | {b['activity']} | {b['detail']} |"
          for d in plan["day_schedules"] for b in d["blocks"]]
    if prof.get("exceptions"):
        L += ["", "## 날짜별 예외"] + [f"- {e}" for e in prof["exceptions"]]
    L += ["", "## 주간 계획"]
    for w in plan["weekly_plan"]:
        L += ["", f"### {w['week']}주차 ({week_label(prof, w['week'])}) · {w['theme']}", "**목표**"]
        L += [f"- {g}" for g in w["goals"]] + ["**할 일**"] + [f"- [ ] {t}" for t in w["tasks"]]
        L += [f"> 점검 기준: {w['checkpoint']}"]
    L += ["", "## 실천 팁"] + [f"- {t}" for t in plan["tips"]]
    return "\n".join(L)


def plan_csv(plan, prof):
    """노션 '가져오기 → CSV'용: 주차별 할 일을 데이터베이스(체크리스트)로"""
    buf = io.StringIO()
    w_ = csv.writer(buf)
    w_.writerow(["할 일", "주차", "시작일", "종료일", "구분", "주제", "완료"])
    for w in plan["weekly_plan"]:
        r = prof["ranges"][w["week"] - 1] if 0 < w["week"] <= len(prof["ranges"]) else (None, None)
        a = r[0].isoformat() if r[0] else ""
        b = r[1].isoformat() if r[1] else ""
        for g in w["goals"]:
            w_.writerow([g, f"{w['week']}주차", a, b, "목표", w["theme"], "No"])
        for t in w["tasks"]:
            w_.writerow([t, f"{w['week']}주차", a, b, "할 일", w["theme"], "No"])
    return buf.getvalue().encode("utf-8-sig")


def _rt(text):
    return [{"type": "text", "text": {"content": str(text)[:1900]}}]


def plan_notion_blocks(plan, prof):
    B = lambda t, kind: {"object": "block", "type": kind, kind: {"rich_text": _rt(t)}}
    blocks = [B(f"{prof['track']} · {prof['stage']} · {prof['period']} ({prof['weeks']}주)", "paragraph"),
              B("계획 요약", "heading_2"), B(plan["summary"], "paragraph"), B("요일별 시간표", "heading_2")]
    for d in plan["day_schedules"]:
        blocks.append(B(f"{norm_day(d['day'])}요일", "heading_3"))
        blocks += [B(f"{b['time']} · {b['activity']} — {b['detail']}", "bulleted_list_item") for b in d["blocks"]]
    blocks.append(B("주간 계획", "heading_2"))
    for w in plan["weekly_plan"]:
        blocks.append(B(f"{w['week']}주차 ({week_label(prof, w['week'])}) · {w['theme']}", "heading_3"))
        blocks += [B(f"목표: {g}", "bulleted_list_item") for g in w["goals"]]
        blocks += [{"object": "block", "type": "to_do", "to_do": {"rich_text": _rt(t), "checked": False}}
                   for t in w["tasks"]]
        blocks.append(B(f"점검 기준: {w['checkpoint']}", "quote"))
    blocks.append(B("실천 팁", "heading_2"))
    blocks += [B(t, "bulleted_list_item") for t in plan["tips"]]
    return blocks


def notion_push(token, page_url, title, blocks):
    """노션 페이지 아래에 새 페이지로 학습 계획 만들기 (Notion API)"""
    ids = re.findall(r"[0-9a-fA-F]{32}", page_url.replace("-", ""))
    if not ids:
        raise ValueError("노션 페이지 링크에서 페이지 ID를 찾지 못했습니다.")
    headers = {"Authorization": f"Bearer {token}", "Notion-Version": "2022-06-28",
               "Content-Type": "application/json"}
    res = requests.post("https://api.notion.com/v1/pages", headers=headers, timeout=30, json={
        "parent": {"page_id": ids[-1]},
        "properties": {"title": {"title": _rt(title)}},
        "children": blocks[:100]})
    if res.status_code >= 300:
        raise RuntimeError(res.json().get("message", res.text))
    page = res.json()
    for i in range(100, len(blocks), 100):     # 노션은 한 번에 100블록까지
        r = requests.patch(f"https://api.notion.com/v1/blocks/{page['id']}/children", headers=headers,
                           timeout=30, json={"children": blocks[i:i + 100]})
        if r.status_code >= 300:
            raise RuntimeError(r.json().get("message", r.text))
    return page.get("url", "")


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
           _p(f"{prof['track']} · {prof['stage']} · {prof['period']} ({prof['weeks']}주) · 작성일 {dt.date.today():%Y-%m-%d}", "sub"),
           _p(f"학습 목표: {prof['goal']}", "box"), _p("계획 요약", "h1"), _p(plan["summary"])]
    st_ += [_p("요일별 시간표", "h1"), _table(["요일", "시간", "할 일", "세부 내용"],
                                          [[norm_day(d["day"]), b["time"], b["activity"], b["detail"]]
                                           for d in plan["day_schedules"] for b in d["blocks"]],
                                          [14 * mm, 28 * mm, 42 * mm, CW - 84 * mm])]
    if prof.get("exceptions"):
        st_ += [_p("날짜별 예외", "h1")] + _bullets(prof["exceptions"])
    st_.append(_p("주간 계획", "h1"))
    for w in plan["weekly_plan"]:
        st_.append(KeepTogether([_p(f"{w['week']}주차 ({week_label(prof, w['week'])}) · {w['theme']}", "h2"), _p("목표")] + _bullets(w["goals"])
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
st.markdown("""
<style>
/* ── 전체 톤 ───────────────────────────── */
:root { --navy:#1F3864; --blue:#2E75B6; --sky:#EAF3FF; --line:#E3E9F2; }
.stApp { background: linear-gradient(180deg, #F6F9FE 0%, #FFFFFF 38%); }
.block-container { padding-top: 2.2rem; max-width: 1180px; }
h1, h2, h3, h4 { color: var(--navy); letter-spacing: -0.2px; }

/* ── 헤더 배너 ─────────────────────────── */
.hero { background: linear-gradient(120deg, #1F3864 0%, #2E75B6 70%, #4A9BE0 100%); color:#fff;
        padding: 26px 32px; border-radius: 22px; margin-bottom: 18px;
        box-shadow: 0 10px 28px rgba(31,56,100,.18); position: relative; overflow: hidden; }
.hero:after { content:""; position:absolute; right:-60px; top:-60px; width:220px; height:220px;
              border-radius:50%; background: rgba(255,255,255,.08); }
.hero h1 { color:#fff; font-size: 1.95rem; margin: 0; font-weight: 800; }

/* ── 사이드바 ─────────────────────────── */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #F2F6FC 0%, #FFFFFF 100%);
                                   border-right: 1px solid var(--line); }
.side-title { font-size: 0.82rem; font-weight: 800; color: #6B7A90; letter-spacing: .4px;
              margin: 6px 0 6px 2px; text-transform: uppercase; }
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { padding: 0.35rem 0.6rem; min-height: 0;
        border-radius: 14px; border: 1.5px dashed #BFD3EE; background: #FFFFFF; }
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] svg { display: none; }
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] small { display: none; }
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] span { font-size: 0.8rem; }
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button { padding: 0.1rem 0.7rem; min-height: 0;
        font-size: 0.8rem; border-radius: 999px; }
/* 사이드바 메뉴 버튼 */
div[class*="st-key-navoff_"] button { background:#FFFFFF; border:1px solid var(--line); border-radius:14px;
        justify-content:flex-start; padding: .6rem .9rem; font-weight:600; color: var(--navy); }
div[class*="st-key-navoff_"] button:hover { border-color:#9DC3E6; background:#F5F9FF; }
div[class*="st-key-navon_"] button { background: linear-gradient(120deg,#1F3864,#2E75B6) !important; color:#fff !important;
        border:none !important; border-radius:14px; justify-content:flex-start; padding: .6rem .9rem;
        font-weight:700; box-shadow: 0 6px 16px rgba(46,117,182,.25); }
/* 최근 질문 키워드 */
.kw { display:inline-block; background:#FFFFFF; color:var(--navy); border:1px solid #CFE0F5; border-radius:999px;
      padding:4px 12px; margin:3px 4px 3px 0; font-size:0.8rem; box-shadow: 0 1px 3px rgba(31,56,100,.06); }

/* ── 버튼 · 입력 · 카드 ──────────────────── */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button { border-radius: 12px; font-weight: 600;
        transition: all .15s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(31,56,100,.10); }
button[kind="primary"], button[data-testid="stBaseButton-primary"], button[data-testid="stBaseButton-primaryFormSubmit"] {
        background: linear-gradient(120deg,#1F3864,#2E75B6) !important; border: none !important; }
[data-testid="stForm"] { background:#FFFFFF; border:1px solid var(--line); border-radius:18px; padding: 1.1rem 1.2rem;
        box-shadow: 0 6px 18px rgba(31,56,100,.06); }
.stTextInput input, .stTextArea textarea, .stDateInput input, .stNumberInput input { border-radius: 12px !important; }
div[data-baseweb="select"] > div { border-radius: 12px !important; }
[data-testid="stExpander"] { border-radius: 14px; border:1px solid var(--line); background:#FFFFFF; }
[data-testid="stAlert"] { border-radius: 14px; }
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }

/* ── 채팅 ─────────────────────────────── */
[data-testid="stChatMessage"] { background:#FFFFFF; border:1px solid var(--line); border-radius:18px;
        padding: .8rem 1rem; box-shadow: 0 2px 10px rgba(31,56,100,.05); margin-bottom: .6rem; }
div[class*="st-key-sugg_"] button { border-radius: 999px; font-size: 0.84rem; background:#FFFFFF;
        border:1px solid #CFE0F5; color: var(--navy); }
div[class*="st-key-sugg_"] button:hover { background: var(--sky); border-color:#9DC3E6; }
.sec-title { font-weight: 800; color: var(--navy); margin: .4rem 0 .4rem 0; }

/* ── 공부 가능 시간 (예비 방식) ─────────── */
div[class*="st-key-s_on_"] button { background:#C8F7FF !important; color:#1565C0 !important;
      border:1px solid #8FE3F2 !important; border-radius:999px !important; min-height: 2.1rem; }
div[class*="st-key-s_off_"] button { background:#FFD1C9 !important; color:#C62828 !important;
      border:1px solid #F5AFA2 !important; border-radius:999px !important; min-height: 2.1rem; }
div[class*="st-key-fx_"] button { background:#F9FFC6 !important; color:#8A6D00 !important;
      border:1px solid #E3EC8C !important; border-radius:999px !important; opacity:1 !important; min-height: 2.1rem; }
/* 날짜별 수정 달력 */
div[class*="st-key-dday_"] button { background:#FFFFFF; border:1px solid var(--line); border-radius:999px; min-height: 2.4rem; }
div[class*="st-key-dmod_"] button { background:#C8F7FF; border:1px solid #90CAF9; color:#1565C0; border-radius:999px; min-height: 2.4rem; }
div[class*="st-key-dsel_"] button { background:linear-gradient(120deg,#1F3864,#2E75B6) !important; color:#FFFFFF !important;
      border:none !important; border-radius:999px; min-height: 2.4rem; }
.legend { display:flex; flex-wrap:wrap; gap:8px; margin: 4px 0 18px 0; }
.legend span { display:inline-block; padding:4px 14px; border-radius:999px; font-size:0.82rem; font-weight:600; line-height:1.4; }
.slot-head { text-align:center; font-weight:700; color:var(--navy); }
.slot-time { font-size:0.85rem; color:#595959; padding-top:0.45rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>AIVLE School 학습 도우미</h1></div>', unsafe_allow_html=True)

if not os.environ.get("OPENAI_API_KEY"):
    st.error("OpenAI API 키가 없습니다. Streamlit Cloud의 Settings → Secrets에 OPENAI_API_KEY를 등록하세요.")
    st.stop()

# 기본 RAG 체인 (백서)
default_chain = get_chain()
if get_vectorstore()._collection.count() == 0:
    st.warning("벡터 DB가 비어 있습니다. chroma_db 폴더 또는 백서 PDF 경로(PDF_PATH)를 확인하세요.")

if "history" not in st.session_state:
    st.session_state.history = []   # [{"q", "kw", "a", "sources"}] 최신이 앞

# ─────────────────────────────────────────── 공부 가능 시간 (요일 × 시간)
DAYS = ["일", "월", "화", "수", "목", "금", "토"]
WEEKEND = {"일", "토"}
HOURS = list(range(8, 24))
CELL_STYLE = {"교육": "background-color:#F9FFC6;color:#8A6D00;",
              "가능": "background-color:#C8F7FF;color:#1565C0;",
              "불가": "background-color:#FFD1C9;color:#C62828;"}


def is_fixed(d, h):
    """평일 09:00~18:00은 교육 시간으로 고정"""
    return d not in WEEKEND and 9 <= h < 18


def slot_minutes(d, h):
    return 50 if (d not in WEEKEND and h == 8) else 60     # 평일 08:00~08:50


def day_of(date):
    return DAYS[(date.weekday() + 1) % 7]


if "avail" not in st.session_state:          # 고정 요일 시간표
    st.session_state.avail = {
        f"{d}-{h}": (h in (19, 20, 21)) if d not in WEEKEND else (h in (10, 11, 14, 15, 16, 17))
        for d in DAYS for h in HOURS if not is_fixed(d, h)}
if "date_avail" not in st.session_state:     # 날짜별 수정 {날짜: {시: bool}}
    st.session_state.date_avail = {}
if "grid_ver" not in st.session_state:
    st.session_state.grid_ver = 0


def is_on(d, h, date=None):
    if is_fixed(d, h):
        return False
    if date is not None and date.isoformat() in st.session_state.date_avail:
        return st.session_state.date_avail[date.isoformat()].get(h, False)
    return st.session_state.avail.get(f"{d}-{h}", False)


def set_on(d, h, value, date=None):
    if is_fixed(d, h):
        return
    if date is None:
        st.session_state.avail[f"{d}-{h}"] = value
    else:
        iso = date.isoformat()
        if iso not in st.session_state.date_avail:      # 처음 수정하면 고정 요일 시간표를 복사
            st.session_state.date_avail[iso] = {x: st.session_state.avail.get(f"{d}-{x}", False)
                                                for x in HOURS if not is_fixed(d, x)}
        st.session_state.date_avail[iso][h] = value


def day_ranges(d, date=None):
    """가능 시간을 이어진 구간으로 합치기. 예) ['08:00~08:50', '19:00~22:00']"""
    hours = [h for h in HOURS if is_on(d, h, date)]
    out, i = [], 0
    while i < len(hours):
        j = i
        while j + 1 < len(hours) and hours[j + 1] == hours[j] + 1:
            j += 1
        a, b = hours[i], hours[j]
        end = "08:50" if (d not in WEEKEND and b == 8) else f"{b + 1:02d}:00"
        out.append(f"{a:02d}:00~{end}")
        i = j + 1
    return out


def day_minutes(d, date=None):
    return sum(slot_minutes(d, h) for h in HOURS if is_on(d, h, date))


def availability_text():
    lines = []
    for d in DAYS:
        r = day_ranges(d)
        edu = " (09:00~18:00 교육 참여)" if d not in WEEKEND else ""
        lines.append(f"- {d}: " + (", ".join(r) + f" → 총 {fmt_minutes(day_minutes(d))}" if r else "공부 가능 시간 없음") + edu)
    return "\n".join(lines)


def exceptions_in(start, end):
    """계획 기간 안에서 고정 요일과 다르게 수정한 날짜 목록"""
    out = []
    for iso in sorted(st.session_state.date_avail):
        date = dt.date.fromisoformat(iso)
        if start <= date <= end:
            d = day_of(date)
            new, base = day_ranges(d, date), day_ranges(d)
            if new != base:
                out.append(f"{fmt_md(date)}({d}): " + (", ".join(new) if new else "공부 불가")
                           + f" (평소 {', '.join(base) if base else '공부 불가'})")
    return out


def period_dates():
    """현재 선택한 계획 기간의 날짜 목록 (최대 12주)"""
    value = st.session_state.get("plan_period")
    if isinstance(value, (tuple, list)) and len(value) == 2:
        a, b = value
    else:
        a = dt.date.today()
        b = a + dt.timedelta(days=27)
    b = min(b, a + dt.timedelta(days=7 * 12 - 1))
    return [a + dt.timedelta(days=i) for i in range((b - a).days + 1)]


def editable_grid(columns, key):
    """
    columns: [(열 이름, 요일, 날짜 또는 None)]
    표에서 칸을 드래그해 여러 칸을 선택한 뒤 '가능/불가로 칠하기' 버튼으로 한 번에 변경
    """
    df = pd.DataFrame(
        {label: ["교육" if is_fixed(d, h) else ("가능" if is_on(d, h, date) else "불가") for h in HOURS]
         for label, d, date in columns},
        index=[f"{h:02d}:00" for h in HOURS])
    styler = df.style
    styler = (styler.map if hasattr(styler, "map") else styler.applymap)(
        lambda v: CELL_STYLE.get(v, "") + "text-align:center;")
    cells = []
    try:
        event = st.dataframe(styler, key=f"{key}_{st.session_state.grid_ver}", on_select="rerun",
                             selection_mode="multi-cell", height=36 * (len(HOURS) + 1) + 4, width="stretch")
        sel = event["selection"] if isinstance(event, dict) else event.selection
        raw = sel.get("cells", []) if isinstance(sel, dict) else getattr(sel, "cells", [])
        for c in raw:
            r, col = (c.get("row"), c.get("column")) if isinstance(c, dict) else (c[0], c[1])
            if isinstance(col, int):
                col = df.columns[col]
            cells.append((int(r), col))
    except Exception:
        # 드래그 선택을 지원하지 않는 버전이면 칸을 하나씩 눌러 바꾸는 방식으로
        st.caption("칸을 눌러 가능 ↔ 불가를 바꿀 수 있어요.")
        head = st.columns([0.8] + [1] * len(columns))
        for j, (label, d, date) in enumerate(columns):
            head[j + 1].markdown(f"<div class='slot-head'>{label}</div>", unsafe_allow_html=True)
        for h in HOURS:
            row = st.columns([0.8] + [1] * len(columns))
            row[0].markdown(f"<div class='slot-time'>{h:02d}:00</div>", unsafe_allow_html=True)
            for j, (label, d, date) in enumerate(columns):
                if is_fixed(d, h):
                    row[j + 1].button("교육", key=f"fx_{key}_{j}_{h}", disabled=True, width="stretch")
                else:
                    on = is_on(d, h, date)
                    if row[j + 1].button("가능" if on else "불가", key=f"s_{'on' if on else 'off'}_{key}_{j}_{h}",
                                         width="stretch"):
                        set_on(d, h, not on, date)
                        st.rerun()
        return

    label_map = {label: (d, date) for label, d, date in columns}
    b1, b2 = st.columns(2)
    paint_on = b1.button("선택한 칸 가능으로 칠하기", key=f"on_{key}", disabled=not cells, width="stretch")
    paint_off = b2.button("선택한 칸 불가로 칠하기", key=f"off_{key}", disabled=not cells, width="stretch")
    if paint_on or paint_off:
        for r, col in cells:
            if col in label_map and 0 <= r < len(HOURS):
                d, date = label_map[col]
                set_on(d, HOURS[r], paint_on, date)
        st.session_state.grid_ver += 1      # 선택 초기화
        st.rerun()


# ─────────────────────────────────────────── 동그란 시간표 (드래그로 칠하기 · Custom Component v2)
GRID_CSS = """
:host, .ag-wrap { display:block; width:100%; }
.ag-wrap { font-family: inherit; user-select: none; -webkit-user-select: none; }
.ag-grid { display:grid; grid-template-columns: 54px repeat(var(--ncol), minmax(0, 1fr)); gap: 7px 8px;
           touch-action: none; padding: 4px 2px 8px 2px; }
.ag-head { text-align:center; font-weight:800; font-size:13px; color:#1F3864; background:#EEF4FC;
           border-radius:999px; padding:7px 0; }
.ag-head.wkd { background:#FDEFF4; color:#B03A5B; }
.ag-time { font-size:12px; color:#7A869A; display:flex; align-items:center; justify-content:flex-end; padding-right:4px; }
.ag-cell { height:32px; border-radius:999px; display:flex; align-items:center; justify-content:center;
           font-size:12px; font-weight:700; cursor:pointer; transition: transform .08s ease, box-shadow .15s ease; }
.ag-cell:hover { transform: scale(1.05); box-shadow: 0 3px 10px rgba(31,56,100,.12); }
.ag-cell.on    { background:#C8F7FF; color:#1565C0; border:1.5px solid #8FE3F2; }
.ag-cell.off   { background:#FFD1C9; color:#C62828; border:1.5px solid #F5AFA2; }
.ag-cell.fixed { background:#F9FFC6; color:#8A6D00; border:1.5px solid #E3EC8C; cursor:not-allowed; }
.ag-cell.fixed:hover { transform:none; box-shadow:none; }
"""

GRID_JS = """
export default function({ parentElement, data, setStateValue }) {
  const root = parentElement;
  let wrap = root.querySelector('.ag-wrap');
  if (!wrap) { wrap = document.createElement('div'); wrap.className = 'ag-wrap'; root.appendChild(wrap); }
  const cols = data.cols, hours = data.hours, weekend = data.weekend || [];
  const state = Object.assign({}, data.cells);           // "열-시": fixed | on | off
  const label = v => v === 'fixed' ? '교육' : (v === 'on' ? '가능' : '불가');
  wrap.style.setProperty('--ncol', cols.length);
  let html = '<div class="ag-grid"><div></div>';
  cols.forEach((c, i) => { html += `<div class="ag-head ${weekend.includes(i) ? 'wkd' : ''}">${c}</div>`; });
  hours.forEach(h => {
    html += `<div class="ag-time">${String(h).padStart(2, '0')}:00</div>`;
    cols.forEach((c, i) => { const k = `${i}-${h}`; html += `<div class="ag-cell ${state[k]}" data-k="${k}">${label(state[k])}</div>`; });
  });
  wrap.innerHTML = html + '</div>';

  let paint = null;                                       // 드래그 중 칠할 값: on | off
  const apply = el => {
    if (!el || !el.dataset || !el.dataset.k) return;
    const k = el.dataset.k;
    if (state[k] === 'fixed' || state[k] === paint) return;
    state[k] = paint; el.className = 'ag-cell ' + paint; el.textContent = label(paint);
  };
  const finish = () => {
    if (paint === null) return;
    paint = null;
    const cells = {};
    for (const k in state) if (state[k] !== 'fixed') cells[k] = state[k] === 'on';
    setStateValue('grid', { ts: Date.now(), cells });
  };
  wrap.onpointerdown = e => {
    const el = e.target.closest('.ag-cell');
    if (!el || el.classList.contains('fixed')) return;
    e.preventDefault();
    paint = state[el.dataset.k] === 'on' ? 'off' : 'on';   // 처음 누른 칸의 반대 값으로 칠하기
    apply(el);
  };
  wrap.onpointermove = e => {
    if (paint === null) return;
    const hit = (root.elementFromPoint ? root.elementFromPoint(e.clientX, e.clientY)
                                       : document.elementFromPoint(e.clientX, e.clientY));
    apply(hit && hit.closest ? hit.closest('.ag-cell') : null);
  };
  wrap.onpointerup = finish;
  wrap.onpointerleave = finish;
  wrap.onpointercancel = finish;
}
"""

try:
    AVAIL_GRID = st.components.v2.component("aivle_avail_grid", html="<div class='ag-wrap'></div>",
                                            css=GRID_CSS, js=GRID_JS)
except Exception:
    AVAIL_GRID = None      # 구버전 Streamlit → 표 방식 사용


def avail_grid(columns, key):
    """동그란 칸을 드래그해 가능/불가를 칠하는 시간표. 지원하지 않으면 표 방식(editable_grid)으로"""
    if AVAIL_GRID is None:
        return editable_grid(columns, key)

    # ① 직전에 칠한 결과를 먼저 반영 (그래야 다시 그릴 때 바뀐 색이 유지됨)
    prev = st.session_state.get(key)
    grid = getattr(prev, "grid", None) if prev is not None else None
    if grid is None and isinstance(prev, dict):
        grid = prev.get("grid")
    if isinstance(grid, dict) and grid.get("ts") and grid.get("ts") != st.session_state.get(f"{key}__ts"):
        st.session_state[f"{key}__ts"] = grid["ts"]
        for k, v in (grid.get("cells") or {}).items():
            ci, h = (int(x) for x in k.split("-"))
            if 0 <= ci < len(columns):
                _, d, date = columns[ci]
                set_on(d, h, bool(v), date)

    # ② 현재 상태로 그리기
    cells = {f"{ci}-{h}": ("fixed" if is_fixed(d, h) else ("on" if is_on(d, h, date) else "off"))
             for ci, (_, d, date) in enumerate(columns) for h in HOURS}
    try:
        AVAIL_GRID(key=key, data={"cols": [c[0] for c in columns], "hours": HOURS, "cells": cells,
                                  "weekend": [i for i, c in enumerate(columns) if c[1] in WEEKEND]},
                   on_grid_change=lambda: None, width="stretch", height=56 + 39 * len(HOURS))
    except Exception:
        editable_grid(columns, key + "_table")


# ─────────────────────────────────────────── 학습 계획 생성 (채팅·계획 공용)
def generate_plan(track, stage, goal, start, end, level):
    ranges = week_ranges(start, end)
    weeks = len(ranges)
    period_txt = f"{fmt_md(start)}~{fmt_md(end)}"
    week_txt = "\n".join(f"- {i}주차: {fmt_md(a)}~{fmt_md(b)}" for i, (a, b) in enumerate(ranges, 1))
    exceptions = exceptions_in(start, end)
    prof = {"track": track, "stage": stage, "goal": goal, "weeks": weeks, "period": period_txt, "ranges": ranges,
            "exceptions": exceptions}
    context = search_context(["개인 학습 전략 하루 루틴", f"{stage} 단계별 전략", "일일 타임테이블",
                              f"{track} 트랙", goal, "배경별 팁"])
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 AIVLE School 학습 코치입니다. [백서 근거]와 학습자 정보로 실행 가능한 학습 계획을 만드세요.
규칙:
1. 요일별 시간표는 일~토 7개 요일을 일요일부터 순서대로 만드세요.
2. 개인 학습 블록은 반드시 [공부 가능 시간] 안에만 배치하세요. 가능 시간이 없는 요일은 '휴식' 블록 하나만 넣으세요.
3. 평일(월~금)에는 '09:00~18:00 교육 참여' 블록을 넣으세요.
4. 가능 시간이 긴 날은 중간에 휴식을 넣고, 블록은 30분 또는 1시간 단위로 나누세요.
5. 주간 계획은 [주차 구분]대로 정확히 {weeks}개 주차로 만들고, 할 일에는 분량(문제 수·시간·회차)을 적으세요.
   [날짜별 예외]가 있는 주는 그날 늘어나거나 줄어든 공부 시간만큼 그 주 할 일 분량을 조정하세요.
6. 백서에 없는 사실·날짜·수치는 만들지 마세요. 한국어로 작성하세요.

[공부 가능 시간]
{avail}

[날짜별 예외] (요일별 시간표는 평소 기준, 아래 날짜만 다름)
{exceptions}

[주차 구분]
{week_txt}

[백서 근거]
{context}"""),
        ("human", "트랙: {track}\n단계: {stage}\n목표: {goal}\n계획 기간: {period} ({weeks}주)\n현재 수준: {level}"),
    ])
    plan = (prompt | ChatOpenAI(model=MODEL, temperature=0.3).with_structured_output(StudyPlan)).invoke({
        "context": context, "track": track, "stage": stage, "goal": goal, "weeks": weeks,
        "week_txt": week_txt, "period": period_txt, "avail": availability_text(), "level": level or "입력 없음",
        "exceptions": "\n".join(f"- {e}" for e in exceptions) or "없음",
    }).model_dump()
    st.session_state.plan, st.session_state.plan_prof = plan, prof
    st.session_state.plan_pdf = plan_pdf(plan, prof)
    st.session_state.pop("notion_url", None)
    return plan, prof


# ─────────────────────────────────────────── 질문 분석 (검색용 질문 · 키워드 · 계획 요청 여부)
def analyze_question(question, history):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "오늘은 {today}입니다. 이전 대화와 마지막 질문을 보고 지정된 항목을 채우세요. "
                   "standalone은 이전 대화를 반영해 그 자체로 이해되는 한국어 질문으로 쓰고, "
                   "날짜는 '다음 주부터 한 달'처럼 상대적으로 말해도 오늘 기준으로 계산하세요."),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])
    chain = prompt | ChatOpenAI(model=MODEL, temperature=0).with_structured_output(QuestionInfo)
    return chain.invoke({"input": question, "history": history, "today": dt.date.today().isoformat()})


def parse_date(s, default):
    try:
        return dt.date.fromisoformat(s)
    except Exception:
        return default


# ─────────────────────────────────────────── 페이지 (학습 도우미 / 개인 학습 계획 / AI 문서 작성)
if "page" not in st.session_state:
    st.session_state.page = "chat"


def go(page):
    """사이드바 메뉴: 같은 메뉴를 다시 누르면 학습 도우미로 돌아감"""
    st.session_state.page = "chat" if st.session_state.page == page else page


# ─────────────────────────────────────────── 사이드바 (문서 업로드 → 개인 학습 계획 → AI 문서 작성 → 최근 질문)
with st.sidebar:
    st.markdown("<div class='side-title'>📂 문서 업로드</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("PDF 업로드", type=["pdf"], label_visibility="collapsed")
    if uploaded is not None:
        file_id = f"{uploaded.name}_{uploaded.size}"
        if st.session_state.get("upload_id") != file_id:
            with st.spinner("문서를 읽는 중..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name
                try:
                    st.session_state.upload_chain = rag_pipeline(tmp_path)
                    st.session_state.upload_retriever = _UPLOAD["retriever"]
                    st.session_state.upload_id = file_id
                    st.session_state.upload_name = uploaded.name
                except Exception as e:
                    st.error(f"문서 처리 오류: {e}")
                finally:
                    os.remove(tmp_path)

    if "upload_chain" in st.session_state:
        st.success(f"📄 {st.session_state.upload_name}")
        if st.button("📘 기본 백서로 돌아가기", width="stretch"):
            for k in ("upload_chain", "upload_retriever", "upload_id", "upload_name"):
                st.session_state.pop(k, None)
            st.rerun()
    else:
        st.caption("📘 기본 문서: AIVLE School 백서")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    page_now = st.session_state.page
    st.button("📅 개인 학습 계획", key=f"nav{'on' if page_now == 'plan' else 'off'}_plan",
              on_click=go, args=("plan",), width="stretch")
    st.button("📝 AI 문서 작성", key=f"nav{'on' if page_now == 'doc' else 'off'}_doc",
              on_click=go, args=("doc",), width="stretch")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    recent_box = st.container()     # 최근 질문 (화면 맨 끝에서 채움 → 방금 질문까지 반영)
    st.caption("※ 지원 일정·금액 등은 기수마다 바뀔 수 있으니 공식 공지를 함께 확인하세요.")

chain = st.session_state.get("upload_chain", default_chain)
retriever = st.session_state.get("upload_retriever", get_retriever())

page = st.session_state.page
if page != "chat":
    st.button("← 학습 도우미로 돌아가기", key="back_chat", on_click=go, args=(page,))

# ─────────────────────────────────────────── 탭 1. 학습 도우미 (최근 대화가 위)
SUGGESTED = [
    "비전공자도 에이블스쿨을 따라갈 수 있을까요?",
    "AI 개발자 트랙과 DX 컨설턴트 트랙은 어떻게 다른가요?",
    "훈련장려금은 얼마나, 언제 받을 수 있나요?",
    "선배들이 강조한 빅프로젝트 수상 비결은?",
    "하루 일과는 어떻게 되나요?",
    "AICE 합격을 목표로 4주 학습 계획을 세워 줘",
]


def history_messages(limit=6):
    """최근 대화 몇 턴을 LLM 기억용 메시지로 (오래된 것부터)"""
    msgs = []
    for h in reversed(st.session_state.history[:limit]):
        msgs += [HumanMessage(content=h["q"]), AIMessage(content=h["a"])]
    return msgs


def show_sources(sources):
    if sources:
        with st.expander("📎 답변 근거 보기"):
            for i, s in enumerate(sources, 1):
                st.markdown(f"**근거 {i}**")
                st.caption(md(s))


if page == "chat":
    # ① 질문 입력 (맨 위)
    with st.form("ask_form", clear_on_submit=True):
        c1, c2 = st.columns([8, 1])
        typed = c1.text_input("질문", placeholder="예) 수료 기준이 어떻게 되나요? / 다음 주부터 4주 SQL 공부 계획 세워 줘",
                              label_visibility="collapsed")
        sent = c2.form_submit_button("보내기", type="primary", width="stretch")

    # ② 추천 질문 (항상 표시, 여러 번 눌러도 동작)
    st.markdown("<div class='sec-title'>💡 추천 질문</div>", unsafe_allow_html=True)
    clicked = None
    cols = st.columns(3)
    for i, q in enumerate(SUGGESTED):
        if cols[i % 3].button(q, key=f"sugg_{i}", width="stretch"):
            clicked = q

    question = clicked or (typed.strip() if sent and typed.strip() else None)

    # ③ 새 질문 → 답변을 맨 위에 바로 표시
    shown = 0
    if question:
        item_id = dt.datetime.now().strftime("%H%M%S%f")
        with st.chat_message("user", avatar="🙋"):
            st.markdown(md(question))
        with st.chat_message("assistant", avatar="📘"):
            thinking = st.empty()
            thinking.markdown("🤔 생각중입니다...")
            past = history_messages()
            info = analyze_question(question, past)
            sources = []

            if info.is_plan_request:
                # 학습 도우미 → 개인 학습 계획 연동
                today = dt.date.today()
                start = parse_date(info.start_date, today)
                end = parse_date(info.end_date, start + dt.timedelta(days=27))
                if end < start or (end - start).days > 7 * 12:
                    end = start + dt.timedelta(days=27)
                track = info.track if info.track in ("DX 컨설턴트", "AI 개발자") else st.session_state.get("plan_track", "DX 컨설턴트")
                plan, prof = generate_plan(track, st.session_state.get("plan_stage", "STEP1 (분석형·생성형 AI)"),
                                           info.goal or question, start, end, st.session_state.get("plan_level", ""))
                thinking.empty()
                answer = (f"📅 **학습 계획을 만들었어요!** 사이드바의 **'📅 개인 학습 계획'**에서 요일별 시간표와 "
                          f"주차별 할 일을 확인하고, PDF·노션으로 내보낼 수 있어요.\n\n"
                          f"- 목표: {prof['goal']}\n- 기간: {prof['period']} ({prof['weeks']}주)\n"
                          f"- 공부 가능 시간: 개인 학습 계획에서 선택한 시간 기준\n\n{plan['summary']}")
                st.markdown(md(answer))
                st.button("📅 학습 계획 보러 가기", key=f"goplan_{item_id}", on_click=go, args=("plan",))
            else:
                raw = []

                def stream():
                    first = True
                    for chunk in chain.stream({"input": question, "query": info.standalone or question,
                                               "history": past}):
                        if first:
                            thinking.empty()
                            first = False
                        raw.append(chunk)
                        yield md(chunk)
                st.write_stream(stream())
                answer = "".join(raw)
                sources = [d.page_content[:300] + ("…" if len(d.page_content) > 300 else "")
                           for d in retriever.invoke(info.standalone or question)]
                show_sources(sources)
        st.session_state.history.insert(0, {"id": item_id, "q": question, "kw": info.keyword or question[:15],
                                            "a": answer, "sources": sources, "plan": info.is_plan_request})
        shown = 1

    # ④ 이전 대화 (최신이 위)
    if st.session_state.history[shown:]:
        st.divider()
    for h in st.session_state.history[shown:]:
        with st.chat_message("user", avatar="🙋"):
            st.markdown(md(h["q"]))
        with st.chat_message("assistant", avatar="📘"):
            st.markdown(md(h["a"]))
            if h.get("plan"):
                st.button("📅 학습 계획 보러 가기", key=f"goplan_{h.get('id', '')}", on_click=go, args=("plan",))
            show_sources(h["sources"])

# ─────────────────────────────────────────── 탭 2. 개인 학습 계획
elif page == "plan":
    st.subheader("📅 개인 학습 계획 만들기")
    today = dt.date.today()

    # ① 계획 기간
    st.markdown("#### ① 계획 기간")
    date_range_input("계획 기간 (시작일 ~ 종료일)", today, today + dt.timedelta(days=27), "plan_period")

    # ② 공부 가능 시간 선택
    st.markdown("#### ② 공부할 수 있는 시간 선택")
    st.markdown(md("칸을 누르거나 **드래그하면 한 번에 칠해져요** (처음 누른 칸의 반대 값으로 칠함). "
                   "평일 09:00~18:00은 교육 시간으로 고정되고, 평일 08:00 칸은 08:00~08:50(체크인 전)입니다. "
                   "주말은 모든 시간을 선택할 수 있어요."))
    st.markdown("<div class='legend'><span style='background:#F9FFC6;color:#8A6D00'>교육</span>"
                "<span style='background:#C8F7FF;color:#1565C0'>가능</span>"
                "<span style='background:#FFD1C9;color:#C62828'>불가</span></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    mode = st.segmented_control("수정 방식", ["고정 요일", "날짜별 수정"], default="고정 요일",
                                key="avail_mode", label_visibility="collapsed") or "고정 요일"

    if mode == "고정 요일":
        st.caption("요일마다 반복되는 기본 시간표입니다. 요일(열)마다 따로 칠할 수 있어요.")
        avail_grid([(d, d, None) for d in DAYS], "week_grid")
        total = sum(day_minutes(d) for d in DAYS)
        st.caption("주간 공부 가능 시간: **" + fmt_minutes(total) + "**  ·  " +
                   "  ".join(f"{d} {fmt_minutes(day_minutes(d))}" for d in DAYS if day_minutes(d)))
    else:
        st.caption("특정 날짜만 다르게 바꿀 수 있어요. 날짜를 고른 뒤 그날의 시간표를 칠하세요. "
                   "(하늘색 날짜 = 수정한 날)")
        dates = period_dates()
        sel = st.session_state.get("sel_date")
        head = st.columns(7)
        for j, d in enumerate(DAYS):
            head[j].markdown(f"<div class='slot-head'>{d}</div>", unsafe_allow_html=True)
        week = [None] * ((dates[0].weekday() + 1) % 7)
        rows = []
        for date in dates:
            week.append(date)
            if len(week) == 7:
                rows.append(week)
                week = []
        if week:
            rows.append(week + [None] * (7 - len(week)))
        for wk in rows:
            cols = st.columns(7)
            for j, date in enumerate(wk):
                if date is None:
                    continue
                iso = date.isoformat()
                prefix = "dsel" if iso == sel else ("dmod" if iso in st.session_state.date_avail else "dday")
                if cols[j].button(f"{date.month}.{date.day}", key=f"{prefix}_{iso}", width="stretch"):
                    st.session_state.sel_date = iso
                    st.rerun()

        if sel and dt.date.fromisoformat(sel) in dates:
            date = dt.date.fromisoformat(sel)
            d = day_of(date)
            st.markdown(f"**{fmt_md(date)} ({d}) 시간표**")
            avail_grid([(f"{date.month}.{date.day}({d})", d, date)], f"date_grid_{sel}")
            st.caption(f"이날 공부 가능 시간: **{fmt_minutes(day_minutes(d, date))}** "
                       f"(평소 {d}요일 {fmt_minutes(day_minutes(d))})")
            if sel in st.session_state.date_avail and st.button("↩️ 이 날짜를 고정 요일 시간표로 되돌리기"):
                st.session_state.date_avail.pop(sel, None)
                st.session_state.grid_ver += 1
                st.rerun()
        else:
            st.info("위 달력에서 수정할 날짜를 눌러 주세요.")

    # ③ 계획 정보 입력
    st.markdown("#### ③ 계획 정보 입력")
    with st.form("plan_form"):
        c1, c2 = st.columns(2)
        track = c1.radio("트랙", ["DX 컨설턴트", "AI 개발자"], horizontal=True, key="plan_track")
        stage = c2.selectbox("현재 단계", ["교육 시작 전", "STEP1 (분석형·생성형 AI)", "STEP2 (Cloud·서비스 개발/제안)",
                                         "STEP3 (빅프로젝트)", "수료 전후 (취업 준비)"], key="plan_stage")
        goal = st.text_area("학습 목표 *", placeholder="예) AICE Associate 합격, SQL 기본기 다지기")
        level = st.text_area("현재 수준·강점·보완점", placeholder="예) 비전공자, 파이썬 기초는 됨, SQL이 약함",
                             key="plan_level")
        ok = st.form_submit_button("📅 학습 계획 생성", type="primary")

    if ok:
        period = st.session_state.get("plan_period")
        if not goal.strip():
            st.warning("학습 목표를 입력해 주세요.")
        elif not (isinstance(period, (tuple, list)) and len(period) == 2):
            st.warning("계획 기간의 시작일과 종료일을 모두 선택해 주세요.")
        elif (period[1] - period[0]).days > 7 * 12:
            st.warning("계획 기간은 최대 12주까지 설정할 수 있어요.")
        elif sum(day_minutes(d) for d in DAYS) == 0 and not st.session_state.date_avail:
            st.warning("공부할 수 있는 시간을 한 칸 이상 '가능'으로 선택해 주세요.")
        else:
            with st.spinner("생각중입니다... 백서를 참고해 계획을 만들고 있어요."):
                generate_plan(track, stage, goal, period[0], period[1], level)

    # ④ 결과
    if "plan" in st.session_state:
        plan, prof = st.session_state.plan, st.session_state.plan_prof
        st.divider()
        st.markdown(f"### 🎯 {md(prof['goal'])}")
        st.caption(md(f"📆 {prof['period']} ({prof['weeks']}주) · {prof['track']} · {prof['stage']}"))
        st.info(md(plan["summary"]))
        if prof.get("exceptions"):
            st.markdown("**📌 날짜별 예외**\n" + "\n".join(f"- {md(e)}" for e in prof["exceptions"]))

        st.markdown("#### 요일별 시간표")
        day_tabs = st.tabs([f"{norm_day(d['day'])}요일" for d in plan["day_schedules"]])
        for t, d in zip(day_tabs, plan["day_schedules"]):
            t.dataframe(pd.DataFrame(d["blocks"]).rename(columns={"time": "시간", "activity": "할 일", "detail": "세부 내용"}),
                        hide_index=True, width="stretch")

        st.markdown("#### 주간 계획")
        for w in plan["weekly_plan"]:
            with st.expander(md(f"{w['week']}주차 ({week_label(prof, w['week'])}) · {w['theme']}"), expanded=(w["week"] == 1)):
                st.markdown("**목표**\n" + "\n".join(f"- {md(g)}" for g in w["goals"]))
                st.markdown("**할 일**\n" + "\n".join(f"- [ ] {md(t)}" for t in w["tasks"]))
                st.caption(md(f"점검 기준: {w['checkpoint']}"))
        st.markdown("#### 실천 팁\n" + "\n".join(f"- {md(t)}" for t in plan["tips"]))

        stamp = f"{dt.date.today():%Y%m%d}"
        st.download_button("⬇️ 학습 계획 PDF 다운로드", st.session_state.plan_pdf,
                           file_name=f"학습계획_{stamp}.pdf", mime="application/pdf")

        # ⑤ 노션 연동
        st.markdown("#### 🗂️ 노션으로 내보내기")
        n1, n2 = st.columns(2)
        n1.download_button("⬇️ 노션 페이지용 Markdown", plan_markdown(plan, prof).encode("utf-8"),
                           file_name=f"학습계획_{stamp}.md", mime="text/markdown", width="stretch")
        n2.download_button("⬇️ 노션 데이터베이스용 CSV", plan_csv(plan, prof),
                           file_name=f"학습계획_할일_{stamp}.csv", mime="text/csv", width="stretch")
        st.caption("노션에서 **설정 → 가져오기 → Markdown/CSV**로 불러오면 됩니다. "
                   "Markdown은 체크박스가 있는 페이지로, CSV는 주차·시작일·완료 열이 있는 할 일 데이터베이스로 만들어져요.")

        with st.expander("🔗 노션에 바로 보내기 (API 연동)"):
            st.caption("1) notion.so/my-integrations 에서 통합을 만들고 토큰을 복사합니다.  "
                       "2) 계획을 넣을 노션 페이지 오른쪽 위 ··· → 연결 → 만든 통합을 추가합니다.  "
                       "3) 그 페이지 링크를 아래에 붙여 넣습니다.")
            try:
                default_token = st.secrets.get("NOTION_TOKEN", "")
            except Exception:
                default_token = ""
            token = st.text_input("노션 통합 토큰", value=default_token, type="password",
                                  help="Streamlit Secrets에 NOTION_TOKEN으로 저장해 두면 자동으로 입력됩니다.")
            page_url = st.text_input("노션 페이지 링크", placeholder="https://www.notion.so/...")
            if st.button("📤 노션에 계획 보내기", type="primary", disabled=not (token and page_url)):
                try:
                    with st.spinner("노션에 페이지를 만드는 중..."):
                        st.session_state.notion_url = notion_push(
                            token, page_url, f"학습 계획 · {prof['goal'][:30]} ({prof['period']})",
                            plan_notion_blocks(plan, prof))
                except Exception as e:
                    st.error(f"노션 전송 실패: {e}")
            if st.session_state.get("notion_url"):
                st.success(f"노션에 학습 계획 페이지를 만들었어요 👉 [열기]({st.session_state.notion_url})")

# ─────────────────────────────────────────── 탭 3. AI 문서 작성
elif page == "doc":
    st.subheader("📝 프로젝트 정리 문서 만들기")
    st.caption("프로젝트를 하나씩 추가한 뒤 PDF·PPT로 정리합니다. (백서 9장 포트폴리오 작성 원칙 적용)")
    if "projects" not in st.session_state:
        st.session_state.projects = []

    today = dt.date.today()
    with st.form("project_form", clear_on_submit=True):
        c1, c2 = st.columns([2, 1])
        name = c1.text_input("프로젝트 이름 *")
        ptype = c2.selectbox("유형", ["빅프로젝트", "미니프로젝트", "공모전", "해커톤", "스터디", "개인 프로젝트", "기타"])
        proj_period = date_range_input("기간 (시작일 ~ 종료일)", today - dt.timedelta(days=55), today, "proj_period")
        c3, c4, c5 = st.columns([1.2, 0.8, 2])
        team_track = c3.selectbox("팀 트랙", ["DX 컨설턴트", "AI 개발자"])
        team_size = c4.number_input("인원", min_value=1, max_value=12, value=5, step=1)
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
        elif not proj_period:
            st.warning("기간의 시작일과 종료일을 모두 선택해 주세요.")
        else:
            a, b = proj_period
            weeks_n = max(1, math.ceil(((b - a).days + 1) / 7))
            period = f"{a:%Y.%m.%d}~{b:%Y.%m.%d} ({weeks_n}주)"
            team = f"{team_size}명 ({team_track} 트랙)"
            st.session_state.projects.append({"프로젝트 이름": name, "유형": ptype, "기간": period, "팀 구성": team,
                                              "사용 기술": tech, "맡은 역할": role, "과제 내용": task,
                                              "수행 과정": process, "결과·성과": result, "얻은 것": learned})
            st.success(f"'{name}' 추가 완료")

    for i, p in enumerate(st.session_state.projects):
        c1, c2 = st.columns([10, 1])
        c1.markdown(md(f"**{i + 1}. {p['프로젝트 이름']}** · {p['유형']} · {p['기간']} · {p['팀 구성']}"))
        if c2.button("삭제", key=f"del_{i}"):
            st.session_state.projects.pop(i)
            st.rerun()

    st.divider()
    c1, c2 = st.columns(2)
    author_name = c1.text_input("이름", placeholder="홍길동")
    job = c2.text_input("지원 직무", value="DX 컨설턴트")
    links = st.text_input("링크 (쉼표로 구분)", placeholder="github.com/..., 블로그 주소")
    formats = st.multiselect("출력 형식", ["PDF", "PPT"], default=["PDF", "PPT"])
    request = st.text_area("요청 사항", placeholder="예) 데이터 분석 역량이 잘 드러나게, 면접관이 1분 안에 읽을 수 있게 간결하게, "
                                                 "금융권 지원용 표현으로")

    if st.button("🪄 AI로 문서 작성", type="primary", disabled=not st.session_state.projects):
        author = {"name": author_name or "에이블러", "job": job,
                  "links": [x.strip() for x in links.split(",") if x.strip()]}
        projects_text = "\n\n".join(f"[프로젝트 {i + 1}]\n" + "\n".join(f"- {k}: {v or '입력 없음'}" for k, v in p.items())
                                    for i, p in enumerate(st.session_state.projects))
        with st.spinner("생각중입니다... 백서의 포트폴리오 원칙을 참고해 작성하고 있어요."):
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
5. projects는 입력 순서대로 입력 개수만큼 만들고, period는 입력한 기간을 그대로 쓰세요.
6. [요청 사항]이 있으면 문체·강조점·분량에 반영하세요. 단, 요청 사항이 1번 규칙(사실만 사용)과 충돌하면 1번을 지키세요.

[요청 사항]
{request}

[백서의 포트폴리오 원칙]
{context}"""),
                ("human", "[사용자 입력]\n지원 직무: {job}\n\n{projects}"),
            ])
            doc = (prompt | ChatOpenAI(model=MODEL, temperature=0.2).with_structured_output(PortfolioDoc)).invoke(
                {"context": context, "job": job, "projects": projects_text,
                 "request": request.strip() or "없음"}).model_dump()
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
        st.success(f"💡 {md(doc['slogan'])}")
        if doc["key_numbers"]:
            for col, k in zip(st.columns(len(doc["key_numbers"])), doc["key_numbers"]):
                col.metric(k["label"], k["value"])
        for p in doc["projects"]:
            with st.expander(md(f"📁 {p['project_name']} — {p['title_sentence']}")):
                st.markdown(md(f"**내 역할**: {p['my_role']}  \n**선택 이유**: {p['role_reason']}"))
                for title, key in (("문제 정의", "problem"), ("해결 과정", "process"), ("결과", "results")):
                    st.markdown(f"**{title}**\n" + "\n".join(f"- {md(x)}" for x in p[key]))
                st.markdown(md(f"**회고**: {p['retrospective']}"))
        for col, (fmt, (data, fname, mime)) in zip(st.columns(max(len(st.session_state.files), 1)),
                                                   st.session_state.files.items()):
            col.download_button(f"⬇️ {fmt} 다운로드", data, file_name=fname, mime=mime)
        st.caption("※ AI가 정리한 초안입니다. 제출 전 내용을 꼭 확인하세요.")

# ─────────────────────────────────────────── 사이드바: 최근 질문 3개 (키워드) · 대화 삭제
with recent_box:
    st.markdown("<div class='side-title'>🕘 최근 질문</div>", unsafe_allow_html=True)
    if st.session_state.history:
        st.markdown("".join(f"<span class='kw'>{escape(h.get('kw') or h['q'][:15])}</span>"
                            for h in st.session_state.history[:3]), unsafe_allow_html=True)
        if st.button("🗑️ 대화 삭제", width="stretch"):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("아직 질문이 없어요.")

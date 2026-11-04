# AI 개인비서 '에이블' (Streamlit)
# 실행: streamlit run app.py
# 같은 폴더에 필요한 파일:
#   - api_key.txt : OPENAI_API_KEY=..., GMAIL_ADDRESS=..., GMAIL_APP_PASSWORD=..., (선택) MY_EMAIL=...
#   - meeting.txt : (선택) 회의 분석 탭에서 기본으로 불러올 회의록
#
# 화면 구성
#   메인 탭  : 🤖 에이블 비서 (음성/텍스트 대화, 파일 첨부, 이메일 발송 도구, 음성 답변, 멀티턴)
#   다른 탭  : 📊 대시보드(캘린더·주요 일정·해야 할 일) / 🎙️ 회의 분석(실시간 녹음·음성 파일·텍스트 파일)
#              / 📥 메일 비서(메일 발송·받은 메일 요약)
#   사이드바 : 앞으로 3일 할 일 / 최근 회의록 요약 / 최근 받은 메일 5개 요약 (눌러서 펼치기)
import os
import re
import json
import uuid
import html
import email
import shutil
import smtplib
import imaplib
import tempfile
import subprocess
import calendar as cal
import datetime as dt
from pathlib import Path
from email import policy
from email.mime.text import MIMEText
from email.utils import parseaddr, parsedate_to_datetime

import streamlit as st
from openai import OpenAI

# 페이지 설정
st.set_page_config(page_title="음성 AI 서비스", page_icon="🤖", layout="wide")
st.title("음성 인식 AI 서비스")
st.write("음성으로 질문하시면 답변해 드립니다.")

# ─────────────────────────────────────────────
# 기본 설정
# ─────────────────────────────────────────────
# 노트북 셀에서 실행해도 오류가 나지 않도록 __file__ 이 없으면 현재 폴더 사용
BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
DATA_FILE = BASE / "able_data.json"
MODEL = "gpt-4o-mini"
KST = dt.timezone(dt.timedelta(hours=9))
WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]
PRIORITIES = ["높음", "보통", "낮음"]
PRIORITY_ORDER = {p: i for i, p in enumerate(PRIORITIES)}
PRIORITY_ICON = {"높음": "🔴", "보통": "🟡", "낮음": "🟢"}

st.set_page_config(page_title="AI 개인비서 에이블", page_icon="🤖", layout="wide")


# ─────────────────────────────────────────────
# API 키 읽기 (노트북과 같은 api_key.txt 형식)
# ─────────────────────────────────────────────
def load_keys():
    key_file = BASE / "api_key.txt"
    if not key_file.exists():
        return
    for line in key_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_keys()


def get_client():
    # 키가 없으면 앱 전체를 멈추지 않고, 부른 곳에서 오류 메시지를 보여 주도록 예외를 냄
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY가 없습니다. api_key.txt를 확인하세요.")
    return OpenAI()


def gmail_ready():
    return bool(os.getenv("GMAIL_ADDRESS") and os.getenv("GMAIL_APP_PASSWORD"))


# ─────────────────────────────────────────────
# 데이터 저장/불러오기
# ─────────────────────────────────────────────
DATA_DEFAULTS = {"todos": [], "events": [], "meeting_notes": [], "mail_summaries": {},
                 "processed_mails": [], "settings": {}}


def load_data():
    d = {}
    if DATA_FILE.exists():
        try:
            d = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        except Exception:
            d = {}
    for k, v in DATA_DEFAULTS.items():
        d.setdefault(k, json.loads(json.dumps(v)))
    return d


def save_data():
    DATA_FILE.write_text(json.dumps(st.session_state.data, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_note(n):
    # 예전 버전(회의 녹음 탭)에서 저장한 회의록도 새 형식으로 읽을 수 있게 빈 항목을 채움
    n.setdefault("id", uuid.uuid4().hex[:8])
    n.setdefault("title", "회의")
    n.setdefault("when", "")
    n.setdefault("attendees", "")
    n.setdefault("source", "이전 버전 회의록")
    n.setdefault("text", n.get("transcript", ""))
    n.setdefault("minutes", "")
    for k in ("summary", "decisions", "kim_tasks", "schedules"):
        if not isinstance(n.get(k), list):
            n[k] = []
    for t in n["kim_tasks"]:
        for k, v in {"title": "", "detail": "", "requester": "", "due_text": "", "due_type": "none",
                     "due_date": "", "priority": "보통", "evidence": "", "status": "nodue", "todo_id": ""}.items():
            t.setdefault(k, v)
    for sc in n["schedules"]:
        for k, v in {"title": "", "when_text": "", "date_type": "none", "date": "", "evidence": "",
                     "verified": True}.items():
            sc.setdefault(k, v)
    return n


if "data" not in st.session_state:
    st.session_state.data = load_data()
data = st.session_state.data
for k, v in DATA_DEFAULTS.items():
    data.setdefault(k, json.loads(json.dumps(v)))
data["meeting_notes"] = [normalize_note(x) for x in data["meeting_notes"] if isinstance(x, dict)]

today = dt.datetime.now(KST).date()   # 한국 시간 기준 (Colab 서버는 UTC)


def now_kst():
    return dt.datetime.now(KST).strftime("%Y-%m-%d %H:%M")


def my_email():
    # 내 이메일 주소: 메일 비서에서 저장한 주소 → api_key.txt의 MY_EMAIL → 발송용 Gmail 주소 순서로 항상 적용
    return (data["settings"].get("my_email") or os.getenv("MY_EMAIL") or os.getenv("GMAIL_ADDRESS") or "").strip()

def parse_date(s):
    try:
        return dt.date.fromisoformat(str(s)[:10])
    except Exception:
        return None


def add_todo(title, due="", priority="보통", detail="", source="직접 입력"):
    item = {
        "id": uuid.uuid4().hex[:8],
        "title": title.strip(),
        "detail": detail.strip(),
        "due": due if parse_date(due) else "",
        "priority": priority if priority in PRIORITIES else "보통",
        "done": False,
        "source": source,
    }
    data["todos"].append(item)
    save_data()
    return item


def add_event(title, date, note="", source="직접 입력"):
    if not parse_date(date):
        return {"status": "failed", "error": "날짜 형식은 YYYY-MM-DD 입니다."}
    item = {
        "id": uuid.uuid4().hex[:8],
        "title": title.strip(),
        "date": date[:10],
        "note": note.strip(),
        "source": source,
    }
    data["events"].append(item)
    save_data()
    return item


def set_todo_done(todo_id, done=True):
    for t in data["todos"]:
        if t["id"] == todo_id:
            t["done"] = done
            save_data()
            return {"status": "success", "todo": t["title"]}
    return {"status": "failed", "error": f"id {todo_id} 를 찾을 수 없습니다."}


def delete_item(kind, item_id):
    data[kind] = [x for x in data[kind] if x["id"] != item_id]
    save_data()


def d_day(d):
    diff = (d - today).days
    if diff == 0:
        return "D-Day"
    return f"D-{diff}" if diff > 0 else f"D+{-diff}"


def fmt_date(d):
    return f"{d.month}/{d.day}({WEEKDAYS[d.weekday()]})"



# ─────────────────────────────────────────────
# 이메일 보내기 / 받기
# ─────────────────────────────────────────────
def send_email(to, subject, body, reply_to_id=""):
    sender = os.getenv("GMAIL_ADDRESS")
    password = os.getenv("GMAIL_APP_PASSWORD")
    if not sender or not password:
        return {"status": "failed", "error": "GMAIL_ADDRESS / GMAIL_APP_PASSWORD 가 설정되지 않았습니다."}
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to
    if reply_to_id and reply_to_id.startswith("<"):   # 답장이면 원래 메일과 같은 대화로 묶기
        msg["In-Reply-To"] = reply_to_id
        msg["References"] = reply_to_id
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.starttls()
            s.login(sender, password)
            s.sendmail(sender, to, msg.as_string())
        return {"status": "success"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


def mail_text(msg):
    # 메일 본문을 글자만 뽑기 (html이면 태그 제거)
    try:
        part = msg.get_body(preferencelist=("plain", "html"))
        if part is None:
            return ""
        body = part.get_content()
        if part.get_content_type() == "text/html":
            body = re.sub(r"<(style|script).*?</\1>", " ", body, flags=re.S | re.I)
            body = re.sub(r"<[^>]+>", " ", body)
        return re.sub(r"\s+", " ", body).strip()
    except Exception:
        return ""


def parse_mail(raw_bytes, flags_line):
    msg = email.message_from_bytes(raw_bytes, policy=policy.default)
    try:
        a = msg["From"].addresses[0]
        name, addr = a.display_name, a.addr_spec
    except Exception:
        name, addr = parseaddr(str(msg.get("From", "")))
    try:
        reply = (msg["Reply-To"] or msg["From"]).addresses[0].addr_spec
    except Exception:
        reply = addr
    try:
        when = parsedate_to_datetime(msg["Date"]).astimezone(KST).strftime("%m/%d %H:%M")
    except Exception:
        when = ""
    subject = str(msg.get("Subject", "")).strip() or "(제목 없음)"
    body = mail_text(msg)[:2500]
    return {
        "subject": subject, "from": name or addr, "addr": reply, "date": when,
        "unread": b"\\Seen" not in flags_line,
        "msg_id": str(msg.get("Message-ID", "")).strip() or f"{subject}|{when}",
        "body": body,
    }


@st.cache_data(ttl=120, show_spinner=False)   # 2분 동안은 다시 불러오지 않음
def fetch_mails(n=5):
    try:
        with imaplib.IMAP4_SSL("imap.gmail.com") as imap:
            imap.login(os.getenv("GMAIL_ADDRESS"), os.getenv("GMAIL_APP_PASSWORD"))
            imap.select("INBOX", readonly=True)          # 읽기 전용: 읽음 표시가 바뀌지 않음
            _, found = imap.search(None, "ALL")
            mails = []
            for mid in found[0].split()[-n:][::-1]:      # 최신 n개
                _, parts = imap.fetch(mid, "(FLAGS BODY.PEEK[])")
                for p in parts:
                    if isinstance(p, tuple):
                        mails.append(parse_mail(p[1], p[0]))
                        break
            return {"mails": mails}
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────
# 음성 (STT / TTS)
# ─────────────────────────────────────────────
AUDIO_TYPES = ["m4a", "mp3", "wav", "webm", "mp4", "mpeg", "mpga", "ogg"]
STT_LIMIT = 24 * 1024 * 1024   # 음성 인식 API 파일 한도(25MB)보다 조금 작게


def shrink_audio(audio_bytes, name):
    # 25MB가 넘는 긴 녹음은 ffmpeg로 음성용 저용량 mp3로 변환 (Colab에는 ffmpeg가 기본 설치)
    if len(audio_bytes) <= STT_LIMIT or not shutil.which("ffmpeg"):
        return audio_bytes, name
    ext = name.rsplit(".", 1)[-1] if "." in name else "m4a"
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in." + ext)
        dst = os.path.join(tmp, "out.mp3")
        with open(src, "wb") as f:
            f.write(audio_bytes)
        subprocess.run(["ffmpeg", "-y", "-i", src, "-ac", "1", "-ar", "16000", "-b:a", "32k", dst],
                       capture_output=True)
        if os.path.exists(dst):
            with open(dst, "rb") as f:
                return f.read(), "meeting.mp3"
    return audio_bytes, name


def speech_to_text(audio_bytes, name="voice.wav", hint=None):
    kwargs = {"model": "whisper-1", "file": (name, audio_bytes), "language": "ko"}
    if hint:
        kwargs["prompt"] = hint
    return get_client().audio.transcriptions.create(**kwargs).text


def text_to_speech(text, voice="alloy"):
    return get_client().audio.speech.create(model="gpt-4o-mini-tts", voice=voice, input=text[:4000]).content


# ─────────────────────────────────────────────
# 지어내지 않기: 근거 문장 확인 + 김 대리 업무 반영 규칙
# ─────────────────────────────────────────────
def _squash(s):
    return re.sub(r"[\s\"'“”‘’.,!?…·:;()\[\]-]", "", s or "")


def found_in(evidence, source):
    # AI가 제시한 근거 문장이 실제 원문에 있는지 확인 (띄어쓰기·문장부호 차이는 무시)
    e, s = _squash(evidence), _squash(source)
    if len(e) < 4:
        return False
    return e in s or (len(e) >= 12 and (e[:12] in s or e[-12:] in s))


TASK_ITEM = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "업무 이름 (짧게)"},
        "detail": {"type": "string", "description": "원문에 있는 세부 내용만"},
        "requester": {"type": "string", "description": "업무를 맡긴 사람, 원문에 없으면 빈 문자열"},
        "due_text": {"type": "string", "description": "기한 표현 원문 그대로, 없으면 빈 문자열"},
        "due_type": {"type": "string", "enum": ["exact", "approximate", "none"]},
        "due_date": {"type": "string", "description": "due_type이 exact일 때만 YYYY-MM-DD, 아니면 빈 문자열"},
        "priority": {"type": "string", "enum": PRIORITIES},
        "evidence": {"type": "string", "description": "원문에서 그대로 복사한 근거 문장 1개"},
    },
    "required": ["title", "detail", "requester", "due_text", "due_type", "due_date", "priority", "evidence"],
    "additionalProperties": False,
}

TASK_RULES = [
    "- 원문에 실제로 있는 내용만 씁니다. 추측하거나 지어내지 않고, 모르면 빈 문자열이나 빈 배열로 둡니다.",
    "- kim_tasks에는 '김대리/김 대리'에게 맡겨졌다고 원문에 나온 업무만 넣습니다. 다른 사람 업무나 팀 전체 논의는 넣지 않습니다.",
    "- 같은 업무가 여러 번 나오면 하나로 합치고 세부 내용을 모읍니다.",
    "- evidence에는 그 업무가 김 대리에게 맡겨졌다는 근거 문장을 원문에서 글자 그대로 한 문장 복사합니다.",
    "- due_type: 날짜나 요일이 구체적이면 exact(예: 11월 14일까지, 이번 주 금요일까지),",
    "  '다음 주까지', '다음 달 초', '빨리', '출시 2주 전'처럼 대략적이면 approximate, 기한 언급이 없으면 none.",
    "- due_text는 원문 표현 그대로, due_date는 exact일 때만 오늘 기준 YYYY-MM-DD로 바꿔 씁니다.",
    "- priority: 원문에 '중요', '꼭', '빨리', '급하게' 같은 표현이 있으면 높음, 없으면 보통.",
]


def apply_kim_tasks(tasks, source_text, source_label):
    # 김 대리 업무만 할 일에 추가. 정확한 날짜가 있는 업무만 캘린더(마감일)에 올리고,
    # 대략적인 기한은 날짜 없이 추가해 분석 화면에서 확인하게 함. 근거가 원문에 없으면 추가하지 않음.
    undone_titles = {t["title"].strip(): t["id"] for t in data["todos"] if not t["done"]}
    out = []
    for t in tasks:
        t = dict(t)
        title = t["title"].strip()
        if not title:
            continue
        if not found_in(t["evidence"], source_text):
            t.update(status="unverified", todo_id="")
        elif title in undone_titles:
            t.update(status="duplicate", todo_id=undone_titles[title])
        else:
            exact = t["due_type"] == "exact" and parse_date(t["due_date"])
            detail = t["detail"].strip()
            if t["requester"]:
                detail += f" / 요청: {t['requester']}"
            if t["due_text"] and not exact:
                detail += f" / 기한(대략): {t['due_text']}"
            item = add_todo(title, t["due_date"] if exact else "", t["priority"], detail.strip(" /"), source_label)
            undone_titles[title] = item["id"]
            t.update(status="calendar" if exact else ("approximate" if t["due_type"] == "approximate" else "nodue"),
                     todo_id=item["id"])
        out.append(t)
    return out


def task_status_line(t):
    d = parse_date(t.get("due_date"))
    return {
        "calendar": f"📅 캘린더에 추가됨 · 마감 {fmt_date(d) if d else ''}",
        "approximate": f"🕓 대략적인 기한 '{t['due_text']}' — 캘린더에는 넣지 않고 할 일 목록에만 추가했어요",
        "nodue": "📝 기한 언급 없음 — 할 일 목록에만 추가했어요",
        "duplicate": "↔️ 이미 할 일 목록에 있는 업무예요",
        "unverified": "⚠️ 원문에서 근거 문장을 찾지 못해 추가하지 않았어요",
    }.get(t.get("status"), "")


def status_report_text():
    lines = [f"[{today.isoformat()}] 에이블 업무 현황", ""]
    undone = sorted([t for t in data["todos"] if not t["done"]],
                    key=lambda t: (t["due"] or "9999", PRIORITY_ORDER[t["priority"]]))
    lines.append(f"■ 해야 할 일 ({len(undone)}건)")
    for t in undone:
        due = f" (마감 {t['due']}, {d_day(parse_date(t['due']))})" if t["due"] else ""
        lines.append(f" - [{t['priority']}] {t['title']}{due}")
        if t["detail"]:
            lines.append(f"     {t['detail']}")
    upcoming = sorted([e for e in data["events"] if parse_date(e["date"]) and parse_date(e["date"]) >= today],
                      key=lambda e: e["date"])
    lines += ["", f"■ 다가오는 일정 ({len(upcoming)}건)"]
    for e in upcoming:
        lines.append(f" - {e['date']} {e['title']} ({d_day(parse_date(e['date']))})")
    return "\n".join(lines)

# ─────────────────────────────────────────────
# 캘린더 렌더링
# ─────────────────────────────────────────────
CALENDAR_CSS = [
    ".able-cal {width:100%; border-collapse:collapse; table-layout:fixed;}",
    ".able-cal th {font-size:.8rem; font-weight:600; padding:6px 0; opacity:.75; text-align:center;}",
    ".able-cal td {vertical-align:top; height:96px; padding:4px; border:1px solid rgba(128,128,128,.22);}",
    ".able-cal .num {font-size:.8rem; font-weight:600; display:inline-block; min-width:22px; text-align:center;}",
    ".able-cal .other {opacity:.35;}",
    ".able-cal .today .num {background:#ff4b4b; color:#fff; border-radius:999px;}",
    ".able-cal .sun {color:#e5484d;} .able-cal .sat {color:#3e7bfa;}",
    ".able-cal .pill {display:block; font-size:.7rem; line-height:1.35; margin-top:2px; padding:1px 4px; border-radius:4px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}",
    ".able-cal .pill.event {background:rgba(62,123,250,.16); border-left:3px solid #3e7bfa;}",
    ".able-cal .pill.todo  {background:rgba(245,158,11,.18); border-left:3px solid #f59e0b;}",
    ".able-cal .pill.over  {background:rgba(229,72,77,.16); border-left:3px solid #e5484d;}",
    ".able-cal .pill.done  {text-decoration:line-through; opacity:.45;}",
    ".able-cal .more {font-size:.68rem; opacity:.6; margin-top:2px;}",
    ".legend span {font-size:.75rem; margin-right:12px;}",
]
st.markdown("<style>" + " ".join(CALENDAR_CSS) + "</style>", unsafe_allow_html=True)


def items_by_date():
    m = {}
    for e in data["events"]:
        d = parse_date(e["date"])
        if d:
            m.setdefault(d, []).append(("event", e["title"]))
    for t in data["todos"]:
        d = parse_date(t["due"])
        if d:
            kind = "done" if t["done"] else ("over" if d < today else "todo")
            m.setdefault(d, []).append((f"todo {kind}", "마감: " + t["title"]))
    return m


# 캘린더 셀 스타일 (클릭형 캘린더용)
st.markdown(
    "<style>"
    ".cal-head {text-align:center; font-size:.8rem; font-weight:600; opacity:.8; padding-bottom:2px;}"
    ".cal-cell {height:54px; overflow:hidden; margin-top:-10px;}"
    ".cal-cell.other {opacity:.35;}"
    ".st-key-able_cal button {padding:0 4px; min-height:32px;}"
    ".st-key-able_cal [data-testid='stHorizontalBlock'] {gap:.35rem;}"
    "</style>",
    unsafe_allow_html=True,
)


def _toggle_from_dialog(todo_id):
    done = st.session_state[f"dlg_chk_{todo_id}"]
    set_todo_done(todo_id, done)
    st.session_state[f"chk_{todo_id}"] = done  # 대시보드 목록의 체크박스도 같이 맞춤


@st.dialog("📅 하루 일정", width="large")
def day_dialog(d):
    iso = d.isoformat()
    st.markdown(f"### {d.year}년 {d.month}월 {d.day}일 ({WEEKDAYS[d.weekday()]}) · {d_day(d)}")

    # --- 일정 ---
    events = [e for e in data["events"] if e["date"] == iso]
    st.markdown(f"#### 📌 일정 ({len(events)}건)")
    if not events:
        st.caption("등록된 일정이 없습니다.")
    for e in events:
        a, b = st.columns([7, 1])
        a.markdown(f"**{e['title']}**")
        if e["note"]:
            a.caption(e["note"])
        if b.button("🗑️", key=f"dlg_del_ev_{e['id']}", help="일정 삭제"):
            delete_item("events", e["id"])
            st.rerun(scope="fragment")

    # --- 이 날 마감인 할 일 ---
    todos = [t for t in data["todos"] if t["due"] == iso]
    st.markdown(f"#### ⏰ 이 날 마감인 할 일 ({len(todos)}건)")
    if not todos:
        st.caption("이 날 마감인 할 일이 없습니다.")
    for t in todos:
        a, b, c = st.columns([0.6, 6.4, 1])
        a.checkbox("완료", value=t["done"], key=f"dlg_chk_{t['id']}",
                   on_change=_toggle_from_dialog, args=(t["id"],), label_visibility="collapsed")
        title = f"~~{t['title']}~~" if t["done"] else f"**{t['title']}**"
        b.markdown(f"{PRIORITY_ICON[t['priority']]} {title}")
        if t["detail"]:
            b.caption(t["detail"])
        if c.button("🗑️", key=f"dlg_del_td_{t['id']}", help="할 일 삭제"):
            delete_item("todos", t["id"])
            st.rerun(scope="fragment")

    # --- 추가 ---
    st.markdown("#### ➕ 이 날에 추가")
    with st.form(f"dlg_add_{iso}", clear_on_submit=True):
        kind = st.radio("종류", ["일정", "할 일(마감)"], horizontal=True)
        title = st.text_input("제목")
        c1, c2 = st.columns([1, 2])
        prio = c1.selectbox("우선순위 (할 일만)", PRIORITIES, index=1)
        note = c2.text_input("메모 / 세부 내용")
        if st.form_submit_button("추가", type="primary"):
            if not title.strip():
                st.warning("제목을 입력하세요.")
            else:
                if kind == "일정":
                    add_event(title, iso, note)
                else:
                    add_todo(title, iso, prio, note)
                st.rerun(scope="fragment")

    st.divider()
    if st.button("닫고 캘린더에 반영", type="primary", use_container_width=True):
        st.rerun()


def render_calendar(year, month):
    m = items_by_date()
    weeks = cal.Calendar(firstweekday=6).monthdatescalendar(year, month)  # 일요일 시작

    head = st.columns(7)
    for i, w in enumerate(["일", "월", "화", "수", "목", "금", "토"]):
        color = "#e5484d" if i == 0 else "#3e7bfa" if i == 6 else "inherit"
        head[i].markdown(f'<div class="cal-head" style="color:{color}">{w}</div>',
                         unsafe_allow_html=True)

    with st.container(key="able_cal"):
        for week in weeks:
            cols = st.columns(7)
            for i, d in enumerate(week):
                with cols[i]:
                    items = m.get(d, [])
                    if st.button(str(d.day), key=f"day_{d.isoformat()}",
                                 type="primary" if d == today else "secondary",
                                 use_container_width=True,
                                 help=f"{d.month}월 {d.day}일 일정 보기·추가·삭제"):
                        day_dialog(d)
                    pills = ""
                    for kind, title in items[:2]:
                        t = html.escape(title)
                        pills += f'<span class="pill {kind}" title="{t}">{t}</span>'
                    if len(items) > 2:
                        pills += f'<div class="more">+{len(items) - 2}건</div>'
                    other = " other" if d.month != month else ""
                    st.markdown(f'<div class="able-cal cal-cell{other}">{pills}</div>',
                                unsafe_allow_html=True)

    st.markdown(
        '<div class="legend"><span>🟦 일정</span><span>🟧 할 일 마감</span>'
        '<span>🟥 기한 지남</span><span>👆 날짜를 누르면 그 날 일정을 보고 추가·삭제할 수 있어요</span></div>',
        unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 화면
# ─────────────────────────────────────────────
if st.session_state.get("pending_toast"):
    st.toast(st.session_state.pop("pending_toast"), icon="✅")

st.title("🤖 AI 개인비서 에이블")
st.caption(f"김 대리님의 업무를 돕는 음성 AI 비서 · 오늘은 {today.year}년 {today.month}월 {today.day}일 "
           f"{WEEKDAYS[today.weekday()]}요일입니다.")

tab_chat, tab_dash, tab_meet, tab_mail = st.tabs(["🤖 에이블 비서", "📊 대시보드", "🎙️ 회의 분석", "📥 메일 비서"])

# ===== 대시보드 =====
with tab_dash:
    undone = [t for t in data["todos"] if not t["done"]]
    week_end = today + dt.timedelta(days=6 - today.weekday())
    due_this_week = [t for t in undone if (d := parse_date(t["due"])) and today <= d <= week_end]
    overdue = [t for t in undone if (d := parse_date(t["due"])) and d < today]
    today_events = [e for e in data["events"] if parse_date(e["date"]) == today]
    total = len(data["todos"])
    rate = round(100 * (total - len(undone)) / total) if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("오늘 일정", f"{len(today_events)}건")
    c2.metric("남은 할 일", f"{len(undone)}건",
              delta=f"기한 지남 {len(overdue)}건" if overdue else None, delta_color="inverse")
    c3.metric("이번 주 마감", f"{len(due_this_week)}건")
    c4.metric("완료율", f"{rate}%")

    left, right = st.columns([3, 2], gap="large")

    # --- 캘린더 ---
    with left:
        if "cal_ym" not in st.session_state:
            st.session_state.cal_ym = (today.year, today.month)
        y, mth = st.session_state.cal_ym
        n1, n2, n3, n4 = st.columns([1, 4, 1, 1])
        if n1.button("◀", key="prev_m"):
            st.session_state.cal_ym = (y - 1, 12) if mth == 1 else (y, mth - 1)
            st.rerun()
        n2.markdown(f"### 📅 {y}년 {mth}월")
        if n3.button("오늘", key="today_m"):
            st.session_state.cal_ym = (today.year, today.month)
            st.rerun()
        if n4.button("▶", key="next_m"):
            st.session_state.cal_ym = (y + 1, 1) if mth == 12 else (y, mth + 1)
            st.rerun()
        render_calendar(y, mth)

    # --- 주요 일정 ---
    with right:
        st.markdown("### ⭐ 주요 일정")
        horizon = today + dt.timedelta(days=30)
        upcoming = []
        for e in data["events"]:
            d = parse_date(e["date"])
            if d and today <= d <= horizon:
                upcoming.append((d, "📌", e["title"], e.get("note", ""), "events", e["id"]))
        for t in undone:
            d = parse_date(t["due"])
            if d and d <= horizon:
                upcoming.append((d, "⏰", f"[마감] {t['title']}", t["detail"], None, t["id"]))
        upcoming.sort(key=lambda x: x[0])

        if not upcoming:
            st.info("앞으로 30일 안에 예정된 일정이 없습니다.")
        for d, icon, title, note, kind, iid in upcoming:
            with st.container(border=True):
                a, b = st.columns([5, 1])
                badge = d_day(d)
                color = "red" if d <= today else ("orange" if (d - today).days <= 3 else "blue")
                a.markdown(f"{icon} **{title}**  \n:{color}[**{badge}**] · {fmt_date(d)}")
                if note:
                    a.caption(note)
                if kind and b.button("🗑️", key=f"del_ev_{iid}", help="일정 삭제"):
                    delete_item("events", iid)
                    st.rerun()

        with st.expander("➕ 일정 추가"):
            with st.form("add_event", clear_on_submit=True):
                et = st.text_input("일정 제목")
                ed = st.date_input("날짜", value=today)
                en = st.text_input("메모")
                if st.form_submit_button("추가") and et:
                    add_event(et, ed.isoformat(), en)
                    st.rerun()

    # --- 할 일 목록 ---
    st.markdown("### ✅ 해야 할 일")
    f1, f2 = st.columns([3, 2])
    view = f1.radio("보기", ["진행 중", "완료", "전체"], horizontal=True, label_visibility="collapsed")

    def toggle(todo_id):
        set_todo_done(todo_id, st.session_state[f"chk_{todo_id}"])

    todos = data["todos"]
    if view == "진행 중":
        todos = [t for t in todos if not t["done"]]
    elif view == "완료":
        todos = [t for t in todos if t["done"]]
    todos = sorted(todos, key=lambda t: (t["done"], t["due"] or "9999", PRIORITY_ORDER[t["priority"]]))

    if not todos:
        st.info("표시할 할 일이 없습니다. '🎙️ 회의 분석'이나 '📥 메일 비서'에서 업무를 가져오거나 직접 추가해 보세요.")
    for t in todos:
        d = parse_date(t["due"])
        with st.container(border=True):
            c0, c1, c2, c3 = st.columns([0.5, 6, 2, 0.7])
            c0.checkbox("완료", value=t["done"], key=f"chk_{t['id']}", on_change=toggle,
                        args=(t["id"],), label_visibility="collapsed")
            title = f"~~{t['title']}~~" if t["done"] else f"**{t['title']}**"
            c1.markdown(f"{PRIORITY_ICON[t['priority']]} {title}")
            if t["detail"]:
                c1.caption(t["detail"])
            if d:
                color = "gray" if t["done"] else ("red" if d < today else
                                                  "orange" if (d - today).days <= 3 else "blue")
                c2.markdown(f":{color}[{fmt_date(d)} · {d_day(d)}]")
            else:
                c2.caption("기한 없음")
            if c3.button("🗑️", key=f"del_td_{t['id']}", help="삭제"):
                delete_item("todos", t["id"])
                st.rerun()

    with st.expander("➕ 할 일 추가"):
        with st.form("add_todo", clear_on_submit=True):
            tt = st.text_input("할 일")
            a, b = st.columns(2)
            td = a.date_input("마감일", value=None)
            tp = b.selectbox("우선순위", PRIORITIES, index=1)
            tdt = st.text_input("세부 내용")
            if st.form_submit_button("추가") and tt:
                add_todo(tt, td.isoformat() if td else "", tp, tdt)
                st.rerun()

# ===== 에이블 비서 도구 (이메일 발송 · 할 일/일정 관리) =====
TOOLS = [
    {
        "type": "function", "name": "send_email",
        "description": "이메일을 발송합니다. 사용자가 이메일로 보내 달라고 명시적으로 요청할 때만 사용합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "수신자 이메일 주소"},
                "subject": {"type": "string", "description": "이메일 제목"},
                "body": {"type": "string", "description": "이메일 본문"},
            },
            "required": ["to", "subject", "body"], "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function", "name": "add_todo",
        "description": "대시보드의 할 일 목록에 새 할 일을 추가합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "할 일 제목"},
                "due": {"type": "string", "description": "마감일 YYYY-MM-DD, 없으면 빈 문자열"},
                "priority": {"type": "string", "enum": PRIORITIES},
                "detail": {"type": "string", "description": "세부 내용, 없으면 빈 문자열"},
            },
            "required": ["title", "due", "priority", "detail"], "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function", "name": "add_event",
        "description": "캘린더에 일정을 추가합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "일정 제목"},
                "date": {"type": "string", "description": "날짜 YYYY-MM-DD"},
                "note": {"type": "string", "description": "메모, 없으면 빈 문자열"},
            },
            "required": ["title", "date", "note"], "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function", "name": "complete_todo",
        "description": "할 일을 완료 처리합니다. id는 현재 할 일 목록에 있는 값을 사용합니다.",
        "parameters": {
            "type": "object",
            "properties": {"id": {"type": "string", "description": "할 일 id"}},
            "required": ["id"], "additionalProperties": False,
        },
        "strict": True,
    },
]

TOOL_LABEL = {"send_email": "📧 메일 발송", "add_todo": "✅ 할 일 추가",
              "add_event": "📅 일정 추가", "complete_todo": "☑️ 완료 처리"}


def run_tool(name, args):
    if name == "send_email":
        return send_email(**args)
    if name == "add_todo":
        return add_todo(args["title"], args["due"], args["priority"], args["detail"], "에이블")
    if name == "add_event":
        return add_event(args["title"], args["date"], args["note"], "에이블")
    if name == "complete_todo":
        return set_todo_done(args["id"], True)
    return {"status": "failed", "error": f"알 수 없는 도구: {name}"}



def latest_note():
    return data["meeting_notes"][0] if data["meeting_notes"] else None


def system_prompt():
    snapshot = {
        "todos": [{k: t[k] for k in ("id", "title", "due", "priority", "done", "detail")} for t in data["todos"]],
        "events": [{k: e[k] for k in ("id", "title", "date", "note")} for e in data["events"]],
    }
    lines = [
        "당신은 '김 대리'의 개인 비서 '에이블'입니다. 김 대리가 요청한 업무를 최선을 다해 처리합니다.",
        f"오늘 날짜: {today.isoformat()} ({WEEKDAYS[today.weekday()]}요일)",
        f"김 대리 이메일(‘내 메일로 보내줘’의 받는 사람): {my_email() or '(미설정 - 주소를 물어보세요)'}",
        "",
        "규칙",
        "- [현재 업무 데이터], [최근 회의], 첨부 파일, 이전 대화에 있는 내용만 근거로 답합니다.",
        "  근거가 없는 내용은 지어내지 말고 '확인되지 않는 내용입니다'라고 말합니다.",
        "- '다음 주 금요일', '모레' 같은 표현은 오늘 날짜 기준 YYYY-MM-DD로 바꿔 도구에 넘깁니다.",
        "- 할 일/일정 추가, 완료 처리를 요청하면 해당 도구를 사용합니다.",
        "- 이메일로 보내 달라는 명시적 요청이 있을 때만 send_email을 사용합니다.",
        "  받는 사람을 따로 말하지 않으면 김 대리 이메일로 보냅니다.",
        "- 도구를 사용한 뒤에는 무엇을 했는지 짧게 보고합니다.",
        "- 공손한 한국어 존댓말로 간결하게 답합니다.",
        "",
        "[현재 업무 데이터]",
        json.dumps(snapshot, ensure_ascii=False),
        "",
    ]
    note = latest_note()
    if note:
        lines += [f"[최근 회의] {note['title']} ({note['when']})", "요약:"]
        lines += [f"- {s}" for s in note["summary"]] or [note["minutes"][:3000]]
        lines += ["김 대리 업무:"] + [f"- {t['title']} (기한: {t['due_text'] or '언급 없음'})" for t in note["kim_tasks"]]
        lines += ["회의 원문:", note["text"][:6000]]
    else:
        lines.append("[최근 회의] (없음)")
    return "\n".join(lines)

# ─────────────────────────────────────────────
# 에이블 비서 (과제 8: 음성 인식 AI 개인비서)
#   텍스트 파일 첨부 · 실시간 녹음(STT) · 실행 버튼 · 스피너 · LLM 호출(도구 사용)
#   · 응답 화면 출력 · 응답 음성 즉시 재생(TTS) · 멀티턴 상태 관리(최근 10개)
# ─────────────────────────────────────────────
MAX_TURNS = 10   # 기억할 대화(질문+답변) 개수
VOICES = {"alloy": "기본", "nova": "밝은", "shimmer": "부드러운", "echo": "차분한", "onyx": "낮은"}
QUICK_ASKS = ["📝 최근 회의 내용 요약해줘", "✅ 나에게 배정된 업무만 알려줘", "📧 오늘 할 일을 내 메일로 보내줘"]


def clean_for_tts(text):
    # 마크다운 기호를 빼고 읽기 좋은 문장으로 (TTS가 '#', '*' 등을 읽지 않게)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)          # [링크](url) → 링크
    t = re.sub(r"^\s*\|?[-:| ]+\|?\s*$", "", t, flags=re.M)     # 표 구분선 제거
    t = re.sub(r"^\s*[-*+]\s+", "", t, flags=re.M)              # 목록 기호
    t = re.sub(r"[#*`>|_~]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def ask_assistant(question, file_name="", file_text=""):
    # 최근 대화(멀티턴) + 첨부 파일 + 업무 데이터로 LLM 호출, 필요하면 도구(이메일 등) 실행
    instructions = system_prompt() + "\n" + "\n".join([
        "",
        "[답변 방식]",
        "- 답변은 화면에 보여 주고 음성으로도 읽어 주므로, 핵심부터 간결하게 말합니다.",
        "- 표보다는 짧은 문단과 목록을 씁니다.",
        "- 이전 대화 내용을 기억하고 이어서 답합니다.",
        "",
        f"[첨부 파일: {file_name}]" if file_text else "[첨부 파일] (없음)",
        file_text[:12000] if file_text else "",
    ])
    inputs = []
    for t in st.session_state.chat[-MAX_TURNS:]:
        inputs.append({"role": "user", "content": t["q"]})
        inputs.append({"role": "assistant", "content": t["a"]})
    inputs.append({"role": "user", "content": question})

    client = get_client()
    actions = []
    for _ in range(5):  # 도구 호출은 최대 5회 반복
        resp = client.responses.create(model=MODEL, instructions=instructions, input=inputs, tools=TOOLS)
        calls = [i for i in resp.output if i.type == "function_call"]
        if not calls:
            return resp.output_text, actions
        inputs += resp.output
        for c in calls:
            args = json.loads(c.arguments)
            result = run_tool(c.name, args)
            ok = isinstance(result, dict) and result.get("status") != "failed"
            detail = f" → {args['to']} · '{args['subject']}'" if c.name == "send_email" else ""
            detail += f" → {args.get('title', '')}" if c.name in ("add_todo", "add_event") else ""
            actions.append({"ok": ok, "text": f"{TOOL_LABEL.get(c.name, c.name)} {'완료' if ok else '실패'}{detail}"
                                              + ("" if ok else f" ({result.get('error')})")})
            inputs.append({"type": "function_call_output", "call_id": c.call_id,
                           "output": json.dumps(result, ensure_ascii=False)})
    return resp.output_text or "요청을 처리했습니다.", actions


def set_quick(q):
    st.session_state.chat_quick = q


def history_text():
    lines = []
    for t in st.session_state.chat:
        lines += [f"[{t['time']}] 김 대리: {t['q']}", f"[{t['time']}] 에이블: {t['a']}", ""]
    return "\n".join(lines)


with tab_chat:
    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("chat_n", 0)       # 입력칸 초기화용 번호
    n = st.session_state.chat_n

    st.markdown("#### 🤖 무엇을 도와드릴까요, 김 대리님?")
    st.caption("말로 물어보거나 글로 입력하세요. 에이블이 답을 화면에 보여 주고 바로 읽어 드립니다. "
               "텍스트 파일을 올리면 그 내용을 참고해 답합니다.")

    with st.container(border=True):
        up_txt = st.file_uploader("📎 텍스트 파일 업로드 (회의록, 메모 등)", type=["txt", "md", "csv"], key="chat_file")
        file_name, file_text = "", ""
        if up_txt is not None:
            file_name = up_txt.name
            file_text = up_txt.getvalue().decode("utf-8", errors="ignore")
            st.caption(f"📎 **{file_name}** 첨부됨 · {len(file_text):,}자 — 이 내용을 참고해 답변합니다.")
            with st.expander("첨부 파일 미리보기"):
                st.text(file_text[:1500] + ("\n…" if len(file_text) > 1500 else ""))

        voice_in = None
        if hasattr(st, "audio_input"):
            voice_in = st.audio_input("🎙️ 무엇을 도와드릴까요? (마이크를 눌러 말씀하세요)", key=f"chat_mic_{n}")
        else:
            st.warning("실시간 녹음을 쓰려면 `!pip install -q -U streamlit`으로 업데이트하세요.")
        typed = st.text_input("⌨️ 또는 글로 입력", key=f"chat_text_{n}",
                              placeholder="예) 회의 내용 요약해서 내 메일로 보내줘")

        qc = st.columns(len(QUICK_ASKS))
        for col, qa in zip(qc, QUICK_ASKS):
            col.button(qa, key=f"quick_{qa}", on_click=set_quick, args=(qa,), use_container_width=True)

        b1, b2, b3, b4 = st.columns([1.1, 1, 1.4, 1.5])
        run_clicked = b1.button("▶ 실행", type="primary", use_container_width=True)
        if b2.button("🧹 새 대화", use_container_width=True):
            st.session_state.chat = []
            st.session_state.chat_n += 1
            st.rerun()
        tts_on = b3.toggle("🔊 답변 음성 재생", value=True, key="chat_tts")
        voice = b4.selectbox("목소리", list(VOICES), format_func=lambda v: f"{VOICES[v]} ({v})",
                             key="chat_voice", label_visibility="collapsed")

    quick = st.session_state.pop("chat_quick", None)
    if run_clicked or quick:
        question = quick.split(" ", 1)[1] if quick else typed.strip()
        if not quick and voice_in is None and not question:
            st.warning("음성을 녹음하거나 질문을 입력한 뒤 실행을 눌러 주세요.")
        else:
            done = False
            try:
                with st.spinner("답변을 준비하고 있습니다.."):
                    stt = None
                    if voice_in is not None and not quick:
                        stt = speech_to_text(voice_in.getvalue())          # 음성 → 텍스트 (STT)
                        question = (stt + " " + question).strip()
                    answer, actions = ask_assistant(question, file_name, file_text)   # LLM 호출
                    reply_audio = text_to_speech(clean_for_tts(answer), voice) if tts_on and answer else None
                st.session_state.chat.append({
                    "q": question, "stt": stt, "via": "voice" if stt else "text", "file": file_name,
                    "a": answer, "actions": actions, "audio": reply_audio,
                    "time": dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%H:%M"),
                })
                st.session_state.chat = st.session_state.chat[-MAX_TURNS:]   # 최근 10개만 저장
                st.session_state.chat_n += 1                                  # 입력칸 비우기
                st.session_state.chat_autoplay = True
                done = True
            except Exception as e:
                st.error(f"답변을 만들지 못했습니다: {e}")
            if done:
                st.rerun()   # try 밖에서 호출해야 화면 새로고침 신호가 막히지 않음

    chat = st.session_state.chat
    if not chat:
        with st.container(border=True):
            st.markdown("**에이블이 할 수 있는 일**")
            st.markdown("- 📝 첨부한 파일이나 최근 회의 내용 요약, 김 대리 업무 정리\n"
                        f"- 📧 요청하면 요약·정리 내용을 이메일로 발송 (내 주소: {my_email() or '메일 비서 탭에서 설정'})\n"
                        "- 📅 일정·할 일 추가, 완료 처리, 이번 주 마감 확인\n"
                        "- 💬 이전 대화를 기억해서 이어서 질문 가능 (최근 10개)")
    else:
        last = chat[-1]
        st.markdown("### 💬 AI 답변")
        with st.container(border=True):
            if last["via"] == "voice":
                st.caption("🎙️ 음성 인식 결과 (STT) — 말씀하신 내용이 맞는지 확인하세요")
                st.info(last["stt"])
            else:
                st.caption("⌨️ 질문")
                st.info(last["q"])
            if last["file"]:
                st.caption(f"📎 참고한 파일: {last['file']}")
            st.markdown(last["a"])
            for a in last["actions"]:
                (st.success if a["ok"] else st.error)(a["text"])
            if last["audio"]:
                st.audio(last["audio"], format="audio/mp3",
                         autoplay=st.session_state.pop("chat_autoplay", False))

        h1, h2 = st.columns([3, 1])
        h1.caption(f"🧠 기억 중인 대화 {len(chat)}/{MAX_TURNS} — 이어서 질문하면 앞의 대화를 참고해 답합니다.")
        h2.download_button("⬇️ 대화 저장", history_text(), file_name="에이블_대화기록.txt",
                           mime="text/plain", use_container_width=True)

        if len(chat) > 1:
            with st.expander(f"🗂️ 이전 대화 {len(chat) - 1}개 보기", expanded=False):
                for t in reversed(chat[:-1]):
                    with st.chat_message("user", avatar="🧑‍💼"):
                        st.markdown(("🎙️ " if t["via"] == "voice" else "") + t["q"])
                        st.caption(t["time"] + (f" · 📎 {t['file']}" if t["file"] else ""))
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(t["a"])
                        for a in t["actions"]:
                            st.caption(a["text"])
                        if t["audio"]:
                            st.audio(t["audio"], format="audio/mp3")



# ─────────────────────────────────────────────
# 🎙️ 회의 분석 (실시간 녹음 · 음성 파일 · 텍스트 파일)
# ─────────────────────────────────────────────
SCHEDULE_ITEM = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "when_text": {"type": "string", "description": "시점 표현 원문 그대로"},
        "date_type": {"type": "string", "enum": ["exact", "approximate", "none"]},
        "date": {"type": "string", "description": "exact일 때만 YYYY-MM-DD, 아니면 빈 문자열"},
        "evidence": {"type": "string", "description": "원문에서 그대로 복사한 근거 문장 1개"},
    },
    "required": ["title", "when_text", "date_type", "date", "evidence"],
    "additionalProperties": False,
}

MEETING_SCHEMA = {
    "type": "object",
    "properties": {
        "title_guess": {"type": "string", "description": "원문에 나온 회의 주제, 없으면 빈 문자열"},
        "summary": {"type": "array", "items": {"type": "string"}, "description": "안건별 핵심 3~6문장"},
        "decisions": {"type": "array", "items": {"type": "string"}, "description": "회의에서 확정된 사항"},
        "kim_tasks": {"type": "array", "items": TASK_ITEM},
        "schedules": {"type": "array", "items": SCHEDULE_ITEM,
                      "description": "김 대리 업무가 아닌, 회의에서 언급된 일정(출시, 광고 시작, 다음 회의 등)"},
    },
    "required": ["title_guess", "summary", "decisions", "kim_tasks", "schedules"],
    "additionalProperties": False,
}


def analyze_meeting(text, title="", attendees=""):
    instructions = "\n".join([
        "당신은 '김 대리'의 개인 비서 '에이블'입니다. 회의 원문을 분석합니다.",
        f"오늘 날짜: {today.isoformat()} ({WEEKDAYS[today.weekday()]}요일). 회의는 오늘 열렸다고 가정합니다.",
        "원문이 음성 인식 결과일 수 있어 오탈자가 있을 수 있습니다. 문맥상 분명한 것만 바로잡습니다.",
        "",
        "[작성 규칙]",
        *TASK_RULES,
        "- summary: 안건별 핵심을 3~6개 문장으로. decisions: 회의에서 확정된 사항만.",
        "- schedules: 김 대리 업무가 아닌 일정만. when_text는 원문 표현, date_type/date는 위 기한 규칙과 같게.",
    ])
    resp = get_client().responses.create(
        model=MODEL, instructions=instructions,
        input=f"[회의 제목] {title or '(입력 없음)'}\n[참석자] {attendees or '(입력 없음)'}\n\n[회의 원문]\n{text}",
        text={"format": {"type": "json_schema", "name": "meeting_analysis",
                         "schema": MEETING_SCHEMA, "strict": True}},
    )
    return json.loads(resp.output_text)


def note_markdown(n):
    if not n["summary"] and n["minutes"]:      # 예전 버전 회의록
        return n["minutes"]
    lines = [f"# 회의록 - {n['title']}", "",
             f"- 일시: {n['when']}", f"- 참석자: {n['attendees'] or '(입력 없음)'}", f"- 분석 방법: {n['source']}", "",
             "## 주요 논의 내용"] + [f"- {s}" for s in n["summary"]]
    lines += ["", "## 결정 사항"] + ([f"- {s}" for s in n["decisions"]] or ["- (없음)"])
    lines += ["", "## 김 대리 업무"]
    for t in n["kim_tasks"]:
        lines.append(f"- {t['title']} (기한: {t['due_text'] or '언급 없음'}"
                     + (f", 요청: {t['requester']}" if t["requester"] else "") + ")")
        if t["detail"]:
            lines.append(f"  - {t['detail']}")
    if not n["kim_tasks"]:
        lines.append("- (없음)")
    lines += ["", "## 회의에서 언급된 일정 (참고)"]
    lines += [f"- {s['title']}: {s['when_text'] or '시점 언급 없음'}" for s in n["schedules"]] or ["- (없음)"]
    return "\n".join(lines)


with tab_meet:
    st.markdown("회의를 녹음하거나 녹음·텍스트 파일을 올리면 에이블이 요약하고 **김 대리 업무만** 할 일에 올립니다.")
    st.caption("📅 정확한 날짜가 있는 김 대리 업무만 캘린더에 들어갑니다. 대략적인 기한이나 다른 일정은 이 화면에서만 보여 드려요. "
               "원문에서 근거를 찾지 못한 내용은 추가하지 않습니다.")

    with st.container(border=True):
        mode = st.radio("분석 방법", ["🎙️ 실시간 녹음", "📁 음성 파일 분석", "📄 텍스트 파일 분석"],
                        horizontal=True, key="meet_mode")
        src_audio, src_name, src_text = None, "", ""
        if mode.startswith("🎙️"):
            if hasattr(st, "audio_input"):
                rec = st.audio_input("마이크 버튼을 눌러 회의 녹음을 시작·종료하세요", key="meet_mic")
                if rec is not None:
                    src_audio, src_name = rec.getvalue(), "recording.wav"
            else:
                st.warning("실시간 녹음을 쓰려면 `!pip install -q -U streamlit`으로 업데이트하세요.")
        elif mode.startswith("📁"):
            up_a = st.file_uploader("회의 녹음 파일 (m4a, mp3, wav 등)", type=AUDIO_TYPES, key="meet_audio")
            if up_a is not None:
                src_audio, src_name = up_a.getvalue(), up_a.name
        else:
            up_t = st.file_uploader("회의록 텍스트 파일 (.txt)", type=["txt", "md"], key="meet_txt")
            default = ""
            if up_t is not None:
                default = up_t.getvalue().decode("utf-8", errors="ignore")
            elif (BASE / "meeting.txt").exists():
                default = (BASE / "meeting.txt").read_text(encoding="utf-8")
            src_text = st.text_area("회의록 내용 (직접 고치거나 붙여 넣어도 됩니다)", value=default, height=200,
                                    key=f"meet_text_{up_t.name if up_t else 'default'}")
        if src_audio:
            st.audio(src_audio)
            st.caption(f"{src_name} · {len(src_audio) / 1024 / 1024:.1f}MB")

        c1, c2 = st.columns(2)
        m_title = c1.text_input("회의 제목 (선택)", placeholder="예: 신제품 출시 마케팅 전략 회의")
        m_people = c2.text_input("참석자 (선택)", placeholder="예: 팀장, 박과장, 이과장, 최대리")
        run_meet = st.button("🔍 회의 분석하기", type="primary",
                             disabled=not (src_audio or src_text.strip()))

    if run_meet:
        done = False
        with st.status("회의를 분석하는 중...", expanded=True) as status:
            try:
                if src_audio:
                    st.write("① 음성을 글로 받아 적는 중 (회의 길이에 따라 1~3분)")
                    a_bytes, a_name = shrink_audio(src_audio, src_name)
                    if len(a_bytes) > 25 * 1024 * 1024:
                        raise ValueError("파일이 너무 큽니다. 25MB 이하로 나눠서 올려 주세요.")
                    hint = "회사 회의 녹음입니다." + (f" 참석자: {m_people}" if m_people else "")
                    text = speech_to_text(a_bytes, a_name, hint)
                else:
                    text = src_text
                st.write("② 회의 내용 분석 중")
                res = analyze_meeting(text, m_title, m_people)
                st.write("③ 김 대리 업무를 할 일·캘린더에 반영 중")
                tasks = apply_kim_tasks(res["kim_tasks"], text, "회의 분석")
                schedules = [dict(s, verified=found_in(s["evidence"], text)) for s in res["schedules"]]
                note = {
                    "id": uuid.uuid4().hex[:8],
                    "title": m_title.strip() or res["title_guess"].strip() or f"{today.isoformat()} 회의",
                    "when": now_kst(), "attendees": m_people.strip(),
                    "source": ("실시간 녹음" if mode.startswith("🎙") else "음성 파일") if src_audio else "텍스트 파일",
                    "text": text, "summary": res["summary"], "decisions": res["decisions"],
                    "kim_tasks": tasks, "schedules": schedules,
                }
                data["meeting_notes"].insert(0, note)
                del data["meeting_notes"][20:]          # 최근 20개까지만 보관
                save_data()
                n_cal = sum(t["status"] == "calendar" for t in tasks)
                n_add = sum(t["status"] in ("calendar", "approximate", "nodue") for t in tasks)
                status.update(label="분석 완료!", state="complete")
                done = True
            except Exception as e:
                status.update(label="처리 중 오류가 발생했습니다", state="error")
                st.error(f"오류 내용: {e}")
        if done:
            st.session_state.meet_view = note["id"]
            st.session_state.pending_toast = f"김 대리 업무 {n_add}건을 할 일에 추가했어요 (캘린더 {n_cal}건)."
            st.rerun()

    notes = data["meeting_notes"]
    if not notes:
        st.info("아직 분석한 회의가 없습니다. 위에서 녹음하거나 파일을 올려 분석해 보세요.")
    else:
        ids = [n["id"] for n in notes]
        cur_id = st.session_state.get("meet_view") if st.session_state.get("meet_view") in ids else ids[0]
        if len(notes) > 1:
            label = {n["id"]: f"{n['when']} · {n['title']}" for n in notes}
            cur_id = st.selectbox("📂 회의록 선택", ids, index=ids.index(cur_id), format_func=lambda i: label[i])
        st.session_state.meet_view = cur_id
        n = next(x for x in notes if x["id"] == cur_id)

        st.markdown(f"### 📄 {n['title']}")
        st.caption(f"{n['when']} · {n['source']} · 참석자: {n['attendees'] or '입력 없음'}")

        l, r = st.columns([3, 2], gap="large")
        with l:
            st.markdown("#### 📋 회의 요약")
            if not n["summary"] and n["minutes"]:
                st.caption("이전 버전에서 만든 회의록입니다. 새 형식으로 보려면 같은 회의를 다시 분석하세요.")
                st.markdown(n["minutes"])
            else:
                st.markdown("\n".join(f"- {s}" for s in n["summary"]) or "_(요약할 내용이 없습니다)_")
            st.markdown("#### ✅ 결정 사항")
            st.markdown("\n".join(f"- {s}" for s in n["decisions"]) or "_(확정된 사항이 없습니다)_")
        with r:
            st.markdown("#### 🗓️ 회의에서 언급된 일정")
            st.caption("참고용입니다. 캘린더에는 넣지 않았어요.")
            if not n["schedules"]:
                st.caption("언급된 일정이 없습니다.")
            for s in n["schedules"]:
                when = s["when_text"] or "시점 언급 없음"
                if s["date_type"] == "exact" and parse_date(s["date"]):
                    when += f" ({fmt_date(parse_date(s['date']))})"
                st.markdown(f"- **{s['title']}** · {when}" + ("" if s.get("verified", True) else " ⚠️근거 확인 필요"))

        st.markdown(f"#### 🙋 김 대리 업무 ({len(n['kim_tasks'])}건)")
        if not n["kim_tasks"]:
            st.caption("이 회의에서 김 대리에게 맡겨진 업무는 없습니다.")
        for i, t in enumerate(n["kim_tasks"]):
            with st.container(border=True):
                st.markdown(f"{PRIORITY_ICON.get(t['priority'], '')} **{t['title']}**"
                            + (f"  ·  요청: {t['requester']}" if t["requester"] else ""))
                if t["detail"]:
                    st.write(t["detail"])
                msg = task_status_line(t)
                {"calendar": st.success, "unverified": st.warning}.get(t["status"], st.info)(msg)
                if t["evidence"]:
                    st.caption(f"근거: “{t['evidence']}”")

                todo = next((x for x in data["todos"] if x["id"] == t.get("todo_id")), None)
                if t["status"] in ("approximate", "nodue") and todo and not todo["due"]:
                    a, b = st.columns([2, 2])
                    pick = a.date_input("마감일 정하기", value=None, key=f"pick_{n['id']}_{i}",
                                        label_visibility="collapsed")
                    if b.button("📅 이 날짜로 캘린더에 추가", key=f"setdue_{n['id']}_{i}", disabled=pick is None):
                        todo["due"] = pick.isoformat()
                        t.update(status="calendar", due_date=pick.isoformat())
                        save_data()
                        st.session_state.pending_toast = f"'{t['title']}' 마감일을 {fmt_date(pick)}로 정했어요."
                        st.rerun()
                if t["status"] == "unverified":
                    if st.button("그래도 할 일에 추가", key=f"force_{n['id']}_{i}"):
                        item = add_todo(t["title"], "", t["priority"], t["detail"], "회의 분석")
                        t.update(status="nodue", todo_id=item["id"])
                        save_data()
                        st.rerun()

        b1, b2, b3, b4 = st.columns(4)
        if b1.button("📧 내 메일로 보내기", use_container_width=True, disabled=not (my_email() and gmail_ready())):
            sr = send_email(my_email(), f"[에이블] 회의록 - {n['title']}", note_markdown(n))
            if sr["status"] == "success":
                st.success(f"{my_email()}로 보냈습니다.")
            else:
                st.error(f"발송 실패: {sr['error']}")
        b2.download_button("⬇️ 회의록 (.md)", note_markdown(n), file_name=f"회의록_{n['when'][:10]}.md",
                           mime="text/markdown", use_container_width=True)
        b3.download_button("⬇️ 원문 (.txt)", n["text"], file_name=f"회의원문_{n['when'][:10]}.txt",
                           mime="text/plain", use_container_width=True)
        if b4.button("🗑️ 이 회의록 삭제", use_container_width=True):
            added = {t.get("todo_id") for t in n["kim_tasks"] if t["status"] in ("calendar", "approximate", "nodue")}
            data["todos"] = [x for x in data["todos"] if x["id"] not in added]
            data["meeting_notes"] = [x for x in notes if x["id"] != n["id"]]
            save_data()
            st.session_state.pending_toast = "회의록과 이 회의에서 추가한 할 일을 지웠어요."
            st.rerun()
        with st.expander("🗣️ 회의 원문 / 녹취록 보기"):
            st.text(n["text"])


# ─────────────────────────────────────────────
# 📥 메일 비서 (메일 발송 · 받은 메일 요약)
# ─────────────────────────────────────────────
MAIL_CATEGORIES = ["업무 요청", "회의/일정", "공유/참고", "광고/알림"]
CATEGORY_ICON = {"업무 요청": "🔴", "회의/일정": "📅", "공유/참고": "📎", "광고/알림": "📢"}

MAIL_SCHEMA = {
    "type": "object",
    "properties": {
        "mails": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "category": {"type": "string", "enum": MAIL_CATEGORIES},
                    "summary": {"type": "string", "description": "메일 핵심 1~2문장"},
                    "kim_tasks": {"type": "array", "items": TASK_ITEM},
                    "needs_reply": {"type": "boolean"},
                    "reply_draft": {"type": "string"},
                },
                "required": ["index", "category", "summary", "kim_tasks", "needs_reply", "reply_draft"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["mails"],
    "additionalProperties": False,
}

SHORT_SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object",
        "properties": {"index": {"type": "integer"}, "summary": {"type": "string"}},
        "required": ["index", "summary"], "additionalProperties": False}}},
    "required": ["items"], "additionalProperties": False,
}

DRAFT_SCHEMA = {
    "type": "object",
    "properties": {"subject": {"type": "string"}, "body": {"type": "string"}},
    "required": ["subject", "body"], "additionalProperties": False,
}


def mail_payload(mails):
    return "\n\n".join(f"[메일 {i}]\n보낸 사람: {m['from']} <{m['addr']}>\n받은 시각: {m['date']}\n"
                       f"제목: {m['subject']}\n본문: {m['body']}" for i, m in enumerate(mails))


def analyze_mails(mails):
    instructions = "\n".join([
        "당신은 '김 대리'의 개인 비서 '에이블'입니다. 김 대리가 받은 메일을 분석합니다.",
        f"오늘 날짜: {today.isoformat()} ({WEEKDAYS[today.weekday()]}요일). 여기서 '원문'은 각 메일의 제목과 본문입니다.",
        "메일마다 index(입력 번호 그대로), category, summary(1~2문장), kim_tasks, needs_reply, reply_draft를 씁니다.",
        *TASK_RULES,
        "- 메일을 받은 김 대리에게 요청된 일이 kim_tasks입니다. 광고·자동 알림·단순 공유면 빈 배열.",
        "- needs_reply: 사람이 보낸 메일이고 회신이 필요하면 true, 광고·자동 알림이면 false.",
        "- reply_draft: needs_reply가 true일 때만 김 대리 입장의 공손한 답장. 메일에 없는 날짜·숫자·약속은 만들지 않고,",
        "  끝에 '김 대리 드림'. false면 빈 문자열.",
    ])
    resp = get_client().responses.create(
        model=MODEL, instructions=instructions, input=mail_payload(mails),
        text={"format": {"type": "json_schema", "name": "mail_analysis", "schema": MAIL_SCHEMA, "strict": True}},
    )
    return json.loads(resp.output_text)["mails"]


def ensure_mail_summaries(mails):
    # 사이드바용 한 줄 요약. 한 번 요약한 메일은 저장해 두고 다시 요약하지 않음
    cache = data["mail_summaries"]
    missing = [m for m in mails if m["msg_id"] not in cache]
    if missing:
        try:
            resp = get_client().responses.create(
                model=MODEL,
                instructions="각 메일을 본문에 있는 내용만으로 40자 이내 한 문장으로 요약합니다. "
                             "지어내지 않고, 광고·자동 알림이면 '광고/알림:'으로 시작합니다. index는 입력 번호 그대로.",
                input=mail_payload(missing),
                text={"format": {"type": "json_schema", "name": "mail_short", "schema": SHORT_SCHEMA, "strict": True}},
            )
            for it in json.loads(resp.output_text)["items"]:
                if 0 <= it["index"] < len(missing):
                    cache[missing[it["index"]]["msg_id"]] = it["summary"]
            for k in list(cache)[:-100]:      # 최근 100개까지만 보관
                del cache[k]
            save_data()
        except Exception:
            pass
    return {m["msg_id"]: cache.get(m["msg_id"]) or ("(본문 앞부분) " + m["body"][:60] if m["body"] else "(본문 없음)")
            for m in mails}


def draft_mail(points, situation):
    resp = get_client().responses.create(
        model=MODEL,
        instructions="\n".join([
            "당신은 '김 대리'의 개인 비서 '에이블'입니다. 김 대리 이름으로 보낼 업무 메일을 씁니다.",
            "- 김 대리가 준 요점에 있는 사실만 씁니다. 새로운 날짜·숫자·약속·이름을 만들지 않습니다.",
            "- 정중한 비즈니스 메일 형식(인사 → 본론 → 마무리), 끝에 '김 대리 드림'.",
            "- subject는 내용을 잘 나타내는 짧은 제목.",
        ]),
        input=f"[요점]\n{points}\n\n[받는 사람/상황]\n{situation or '(없음)'}",
        text={"format": {"type": "json_schema", "name": "mail_draft", "schema": DRAFT_SCHEMA, "strict": True}},
    )
    return json.loads(resp.output_text)


def fill_mail(kind):
    if kind == "status":
        st.session_state.mail_subject = f"[에이블] {today.isoformat()} 업무 현황"
        st.session_state.mail_body = status_report_text()
    elif kind == "meeting" and latest_note():
        n = latest_note()
        st.session_state.mail_subject = f"[에이블] 회의록 - {n['title']}"
        st.session_state.mail_body = note_markdown(n)
    elif kind == "clear":
        st.session_state.mail_subject = ""
        st.session_state.mail_body = ""
    st.session_state.mail_to = my_email()


with tab_mail:
    st.markdown("메일을 쓰고 보내거나, 받은 메일을 에이블이 요약하고 **김 대리에게 요청된 일**을 할 일에 올립니다.")

    with st.expander(f"⚙️ 내 이메일 주소: {my_email() or '미설정'}"):
        new_addr = st.text_input("내 이메일 주소", value=my_email(), key="set_my_email")
        st.caption("한 번 저장하면 메일 보내기의 기본 받는 사람, 회의록 메일, 에이블 비서의 ‘내 메일로 보내줘’에 항상 적용됩니다.")
        if st.button("저장", key="save_my_email"):
            if re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", new_addr.strip()):
                data["settings"]["my_email"] = new_addr.strip()
                save_data()
                st.session_state.mail_to = new_addr.strip()
                st.session_state.pending_toast = "내 이메일 주소를 저장했어요."
                st.rerun()
            else:
                st.error("이메일 주소 형식을 확인해 주세요.")

    if not gmail_ready():
        st.warning("api_key.txt에 GMAIL_ADDRESS, GMAIL_APP_PASSWORD가 있어야 메일을 보내고 받을 수 있습니다.")

    sub_send, sub_inbox = st.tabs(["✉️ 메일 보내기", "📥 받은 메일 요약"])

    with sub_send:
        st.session_state.setdefault("mail_to", my_email())
        if st.session_state.pop("mail_clear", False):
            fill_mail("clear")

        q1, q2, q3 = st.columns(3)
        q1.button("📊 오늘 업무 현황 불러오기", on_click=fill_mail, args=("status",), use_container_width=True)
        q2.button("📝 최근 회의록 불러오기", on_click=fill_mail, args=("meeting",), use_container_width=True,
                  disabled=latest_note() is None)
        q3.button("🧹 새로 쓰기", on_click=fill_mail, args=("clear",), use_container_width=True)

        with st.expander("🤖 에이블에게 메일 초안 부탁하기"):
            points = st.text_area("메일에 담을 요점", key="mail_points", height=100,
                                  placeholder="예) 경쟁사 분석 보고서 다음 주까지 공유 예정, 자료 요청 감사 인사")
            situation = st.text_input("받는 사람 / 상황 (선택)", key="mail_situation", placeholder="예) 팀장님께 보고")
            if st.button("✍️ 초안 작성", disabled=not points.strip()):
                try:
                    with st.spinner("초안을 쓰는 중..."):
                        dr = draft_mail(points, situation)
                    st.session_state.mail_subject = dr["subject"]
                    st.session_state.mail_body = dr["body"]
                    st.session_state.pending_toast = "초안을 아래에 채워 넣었어요. 확인 후 보내세요."
                    ok_draft = True
                except Exception as e:
                    st.error(f"초안을 만들지 못했습니다: {e}")
                    ok_draft = False
                if ok_draft:
                    st.rerun()

        to = st.text_input("받는 사람", key="mail_to")
        subject = st.text_input("제목", key="mail_subject")
        body = st.text_area("본문", key="mail_body", height=260)
        if st.button("📨 보내기", type="primary", disabled=not (to.strip() and subject.strip() and body.strip())):
            if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", to.strip()):
                st.error("받는 사람 이메일 주소 형식을 확인해 주세요.")
            else:
                sr = send_email(to.strip(), subject, body)
                if sr["status"] == "success":
                    st.session_state.mail_clear = True
                    st.session_state.pending_toast = f"{to.strip()}로 메일을 보냈어요."
                    st.rerun()
                else:
                    st.error(f"발송 실패: {sr['error']}")

    with sub_inbox:
        processed = data["processed_mails"]
        o1, o2 = st.columns([2, 3])
        n_mail = o1.select_slider("요약할 최근 메일 수", options=[5, 10, 15, 20], value=10)
        include_done = o2.checkbox("이미 분석한 메일도 다시 분석", value=False)

        if st.button("🤖 받은 메일 요약하기", type="primary", disabled=not gmail_ready()):
            fetch_mails.clear()
            done, msg = False, ""
            try:
                with st.spinner("메일을 불러와 분석하는 중..."):
                    got = fetch_mails(n_mail)
                    if "error" in got:
                        raise RuntimeError(f"메일을 불러오지 못했습니다: {got['error']}")
                    me = (os.getenv("GMAIL_ADDRESS") or "").lower()
                    targets = [m for m in got["mails"]
                               if m["addr"].lower() != me and (include_done or m["msg_id"] not in processed)]
                    results = []
                    if targets:
                        for a in analyze_mails(targets):
                            if 0 <= a["index"] < len(targets):
                                m = targets[a["index"]]
                                tasks = apply_kim_tasks(a["kim_tasks"], m["subject"] + " " + m["body"],
                                                        f"메일({m['from']})")
                                results.append({**m, **a, "kim_tasks": tasks})
                        for m in targets:
                            if m["msg_id"] not in processed:
                                processed.append(m["msg_id"])
                        del processed[:-500]
                        save_data()
                st.session_state.mail_result = {"mails": results, "sent": []}
                n_add = sum(t["status"] in ("calendar", "approximate", "nodue") for r in results for t in r["kim_tasks"])
                msg = f"메일 {len(results)}통 요약 · 할 일 {n_add}건 추가" if targets else "새로 분석할 메일이 없습니다."
                done = True
            except Exception as e:
                st.error(str(e))
            if done:
                st.session_state.pending_toast = msg
                st.rerun()

        res = st.session_state.get("mail_result")
        if res and res["mails"]:
            work = [m for m in res["mails"] if m["category"] != "광고/알림"]
            ads = [m for m in res["mails"] if m["category"] == "광고/알림"]
            for m in work:
                with st.container(border=True):
                    st.markdown(f"{CATEGORY_ICON[m['category']]} **{m['subject']}**  \n"
                                f":gray[{m['from']} · {m['date']} · {m['category']}]")
                    st.write(m["summary"])
                    for t in m["kim_tasks"]:
                        st.markdown(f"- **{t['title']}** — {task_status_line(t)}")
                    if m["needs_reply"]:
                        if m["msg_id"] in res["sent"]:
                            st.success("답장을 보냈습니다.")
                        else:
                            with st.expander(f"✉️ {m['from']}님께 보낼 답장 초안 보기·수정"):
                                draft = st.text_area("답장", value=m["reply_draft"], height=200,
                                                     key=f"reply_{m['msg_id']}", label_visibility="collapsed")
                                st.caption(f"받는 사람: {m['addr']} · 제목: Re: {m['subject']}")
                                if st.button("📨 이 내용으로 답장 보내기", key=f"send_{m['msg_id']}", type="primary"):
                                    subj = m["subject"] if m["subject"].lower().startswith("re:") else "Re: " + m["subject"]
                                    sr = send_email(m["addr"], subj, draft, m["msg_id"])
                                    if sr["status"] == "success":
                                        res["sent"].append(m["msg_id"])
                                        st.session_state.pending_toast = f"{m['from']}님께 답장을 보냈어요."
                                        st.rerun()
                                    else:
                                        st.error(f"발송 실패: {sr['error']}")
            if ads:
                with st.expander(f"📢 광고/알림 메일 {len(ads)}통"):
                    for m in ads:
                        st.markdown(f"- **{m['subject']}** · {m['from']} — {m['summary']}")


# ─────────────────────────────────────────────
# 사이드바 (챕터를 누르면 펼쳐짐)
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 에이블")
    st.caption("김 대리님의 AI 개인비서")

    # 1) 앞으로 3일 할 일
    undone_all = [t for t in data["todos"] if not t["done"]]
    overdue = sorted([t for t in undone_all if parse_date(t["due"]) and parse_date(t["due"]) < today],
                     key=lambda t: t["due"])
    days3 = [today + dt.timedelta(days=i) for i in range(3)]
    soon = {d: [t for t in undone_all if parse_date(t["due"]) == d] for d in days3}
    n_soon = sum(len(v) for v in soon.values())
    with st.expander(f"✅ 앞으로 3일 할 일 ({n_soon}건)"):
        if overdue:
            st.markdown("**⏰ 기한 지남**")
            for t in overdue:
                st.markdown(f"- {PRIORITY_ICON[t['priority']]} {t['title']} :red[{d_day(parse_date(t['due']))}]")
        for d, label in zip(days3, ["오늘", "내일", "모레"]):
            st.markdown(f"**{label} · {fmt_date(d)}**")
            if soon[d]:
                for t in sorted(soon[d], key=lambda x: PRIORITY_ORDER[x["priority"]]):
                    st.markdown(f"- {PRIORITY_ICON[t['priority']]} {t['title']}")
            else:
                st.caption("마감인 할 일 없음")
        no_due = sum(1 for t in undone_all if not t["due"])
        if no_due:
            st.caption(f"기한 없는 할 일 {no_due}건은 📊 대시보드에서 확인하세요.")

    # 2) 가장 최근 회의록 요약
    with st.expander("📝 최근 회의록 요약"):
        note = latest_note()
        if not note:
            st.caption("아직 분석한 회의가 없습니다.")
        else:
            st.markdown(f"**{note['title']}**")
            st.caption(f"{note['when']} · {note['source']}")
            if note["summary"]:
                st.markdown("\n".join(f"- {s}" for s in note["summary"][:5]))
            elif note["minutes"]:
                st.caption("이전 버전 회의록입니다. 🎙️ 회의 분석 탭에서 내용을 볼 수 있어요.")
            if note["kim_tasks"]:
                st.markdown("**김 대리 업무**")
                st.markdown("\n".join(f"- {t['title']} ({t['due_text'] or '기한 언급 없음'})"
                                      for t in note["kim_tasks"]))
            st.caption("자세한 내용은 🎙️ 회의 분석 탭에서 볼 수 있어요.")

    # 3) 최근 받은 메일 5개 요약
    with st.expander("📥 최근 받은 메일 5개"):
        if not gmail_ready():
            st.caption("api_key.txt에 GMAIL_ADDRESS, GMAIL_APP_PASSWORD가 있어야 볼 수 있습니다.")
        else:
            if st.button("🔄 새로고침", key="side_mail_refresh", use_container_width=True):
                fetch_mails.clear()
            got = fetch_mails(5)
            if "error" in got:
                st.caption(f"메일을 불러오지 못했습니다: {got['error']}")
            elif not got["mails"]:
                st.caption("받은편지함이 비어 있습니다.")
            else:
                sums = ensure_mail_summaries(got["mails"])
                for m in got["mails"]:
                    st.markdown(("🔵 " if m["unread"] else "") + f"**{m['subject'][:30]}**  \n"
                                f":gray[{m['from']} · {m['date']}]")
                    st.caption(sums[m["msg_id"]])
                st.caption("🔵 = 읽지 않은 메일")

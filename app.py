import os, io, math, base64, tempfile, datetime as dt
from operator import itemgetter
from typing import List

import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
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

[참고 문서]
{context}"""

# 문서 포맷팅 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ─────────────────────────────────────────── 로고 (이미지 파일 없이 코드에 포함)
LOGO_B64 = "iVBORw0KGgoAAAANSUhEUgAAAbAAAABiCAYAAAAxz4a6AABW3klEQVR42u29ebxdWVnn/V3Tns45d0hSldQAJYoird0OOIICikNL24izr23rCzgwiC10Qc2pWyNQAo4gOGA32r7t0KiorSIiKqCogLaztihQRVVSSe5whr33Gt8/9jknN6mkKlWV3NxUzpPP+ux7T849Z++191rPWs/ze34/wcJOaV98x8FnHXXtl7bEfXmmbem5v4zpT9+z9urfXvTOwha2sIWdfxOLLjjRvuB7v/dp7ergF7l87+V63ypD59g6dpTHrV7C8O5726WkPt7ec2jtQz/6o29d9NbCFrawhZ0/U4suOG6ff+3Lnz8clD9/xWd8xt7Dkxp0xlK1TKlLcjJWlvfohF4VSn3JEz73KSsffe/7fm/Rawtb2MIWtnBg59U++4aX/cfJoPrRlU964soxZyl6y2SmYrQ5QiZFimC9h9yQVWWRG/N5n/DZn7n1r+957/sXvbewhS1sYQsHdl7sK66+ulev9H578LgrD4ykwgmNNjneBZIL9Ht9pFY00SJ7OU30KGPUuGk/56rP/Kyfv/tP/mS46MWFLWxhC9tZk4sugCYTX99fXj2wNW7IiwqTFTS1o60tg/4yMUaG4xFBQJCgegW2V7Kh5SWTpep7Fz24sIUtbGELB3ZebHNj82ubYa2ecMVVbB05ho6AD6wMlvDOMRqNGCz1WOpXbG0cxafIum247MlPIi31v3XRgwtb2MIWtnBg58X27LnkiZdffjkf/ehHWV3eg21biqKgaRqca1ldXmIyGtM0Df1+n9ZZVFky8R6PePwTX/rSpUUvLmxhC1vYwoHtuDXRTTbGQ/orA1ywIAQhRVJKGGOw1lJVFSEElNRoY5jUNVoqCmj7Si36cWELW9jCFg5s583Z9u7GtzgiTbSITJNSwBiDFhLbWOpJi0iSetyQfGJ1sEQ7HBLHE/cXP/RDG4teXNjCFrawhQPb+U5I4f8urSzhZSTmGotHS8VkOEJGwVKvj0IiUJS6YqUYkPmAcYEwqX9t0YMLW9jCFrZwYOfFBkL9j/X77rOunuCDQ8hEItIvC5SQtK1DZBqT54gER+6+F1W3+I31LW3DOxY9uLCFLWxhCwd2Xuz3XvvD/0dujX921eSsFjmxbcE7stzQhJZNOyZkis1mglKKy/btpXKB+t77/vZDr3/9glJqYQtb2MIWDuz8mWqOvOzwP/zj//HHNuhJQa5gMt4kqkhvpUcSASUiwlkm99/Px/7mbz58adH/pkXPLWxhC1vYgonjvNrH3vsh+0lf+u9/fjzc+jdS8KmSRARkLnHJYW3DcmEY3n1PYmvjfStRPOc9r3rVxxY9t7CFLWxhLNjod4t97sFr/uMkpu/u793zRWPviqSRRipkY/+0asPPvOfgbW9Z9NLCFrawhS0c2K6157zylYOJ9E/0IaYlkf3z2++6a8F5uLCFLWxhC1vYwha2sIUtbGELW9jCFrawhS1sYQtb2MIWtrCFLWxhC1vYwha2sIUtbGELW9jCFrawhS1sYQtb2MIWtrCFLYxFHdgD7SnX3Px4dP5NLfZJRiO1FPd4O3nnh+54zR/uunNdW/tskWVfr6K8PPkkVa/4v8Ju/vr7blj7i8Xjdu6tB/tjkf2GVHJCJERSAoRGkKQQCmIU0kTnkynzFh+KKMgmdXud9/73d8t1FPCEoNUvSCHGWooghJRCCIAEiBiD0FpjrUMXeXLWKu89PqRnnOFXXNIrq1/33tVZXpCSa0NIJYKQaYP1DokQIqXYekthCmHyvNc0zU9Mmuand9t9z7Ls38rEW4RSQ4JHSoWUAiFkCCmK4LwsepVM3hMi0QaXW2ufdiafbYx5vkB8l0Q0iYiWCghpei+IAmTqjikhSBQh8XPOuTdeCGOmX/VfFJN/Qa71yDmXhIiAIqWU2MY6GKbUTQFg+38dt/mLKXY/Cjl1MdOHF0hSILa99WR/lOZ+SSKIpG2fG6f/JxvrtkIIX7PrHdiTX37tq1eu+qQXq0F/MHINEKhKw/q9Hycbbv3P/scPffd73/KW815g/LRXvnKwLsOPmCsf9+3ZYFXqqJjULbpnsMN16o985A2fmOQ173jta8cLN3PuTGt5bfDxVbPhItP2EQGz19P09SLPaFvL0urSHx1b33r6LrqUJwn4ezU9Z5FOnCWkEAghcDGipMDFhFICH9IZj+ncZP9onf3k2e9KCULovshIQUoJpRRaa+qmnb3towGu2m333Sj1hhjTi0mRPMvx1hGJCCGJKXbXJxUhBiJQFtV/mzST553ZM6VfQUx3AYiUEEKQYjhhZp11epgelwaD12wOh9deCGOmyLIbo3e3aa3xziEEhHgaz7TNk5xPS3AM2LuryXw/48ZX/n8rV1z2vf1BbzAabrKyNKAYlDTB8fjHP559V17xdR/vZ+9/2itfOTjf57qeh7ftefKn/L+D1VWZG0MdHCJXVEXO5VdeQXFgz4v/KYx+7Suuvrq3cDPnNKTw/b1e2U360ya3/Xx8spZUZUHTWhKwsbH1xVXFZbvqWk465zSdOBIQUsLHiNGKlBLyEYzk1tnbqrJCK92txPt9skwjJYTYTdTeB+qmJTMZUkiqsnp8Dp+0C+/8C6WUaKVpbQtS0Ov1iSmilaZX9Qixcy9VWRG9fTi7yJhSIsZITJEYI1IIlBBIAUoK5Lx192w8rtMFNGxSiAnvPTF1izslp216nUoqpJDzpqQ64zb7G7Htn5Q8qpZl2XBXs9F/2o0vf4k+sP/ryktWem1y7L9kH87WtNGS9QvWh0OGMWaf+MVf8OTDub/9vDraW6+5ZelJn/hlriqZWMdo0qDKjN5qj0k94vCRw+x93JXi8k978pd+VLXXL9zMObNPlVLuH4/rbod10hQyey0l8CHinCPPDFpJhIB2wpfupovpQmACKSVSSpSU6GnLtEYCMXZB0hghpYc9Z76zrusYY0QpwebmkBACMXY7ihATeZ4hAOssUkom9QSk/Lpddt+f5oOXs+uf7bSappn2o2Q8GVPkBWVRMqknNN6/5+Et+LsoWDf5yunzlE44nhQJuKAc2PYY3ukWQ2K66xfioTf5298z+/lM//ZMzFqb7VoH9sxXvOJAtu/ArZMszyZKIHsZR48eIsSWiaupg0MWGQz63Bcsk9Xl7/6sg9d81fk4189fe+nSsUzc2GQF4xCJwlBUJRjBkY2jKCMRWlATcFUuqquuetnnXHvtZy18zdm3Xq/3Ams9WaYfsHvZHj7sJrluMLXWkVJCC0mvKv/rbrqeEBIxTlf+JzXrPQlQ6rigRHz4cZ17ldbvTilhjKEoMkJIaC3RWqGVpGktSh2fJqqyAsQrdlXYWOpvMlIhUsIHjzGGqqy6n7XBOTeb9Bg3NWVR3PnId8Vi7rTSCYuiNG1MFxTxghk32081bXstRogpde2k54+HaCmEE34XKc2bBER85C0lkFLaXevAjph8zee9PcuX7GeznbBpxwwuWQYJ+/btZVRPCJmBIscKzfLlVxRbRfWTn7+2trTT51qr3udc8rjHy7r1kCR5WeJjBJnQmULniqrXY1w31CFhTVFOCnPLwt2cfRuPx1cf373IUzaluqa1xjqP0QpjDCklxpP6syp2Vxhxe748bWuZ1tOdkSfPs+k1P/zPLvPyLYlE01iaxpLnBu8jzof5TkNKSb/XxwfPpJ4gpLwEeNIu6qEXhhjIsqwLIbZtd54IQujChr2qi9xnJmPSNL/08Cb4KGYOKqZImubBujzk8Z2ymP8OeV5yIe7AZs/bPFw3vc5u7Kh5O934eqh2/O/Fo2paSrUrHdi/u/7gs/SVV35PVgwYHhuysmcfLZ4xlgkNdV1jpML0ehzeGEJQ5MUAdcm+K45G/+qdPt9KFSvaC4wF7QRN0zBqJrjWUpUFR9ePEGWkp0tWymXKwSp6Zc9/+ILrXvmchcs5i6tweKqWCqUE1nqSjyQfCaFrMUTStBGn4AQlu4R8SoSYyIxmAs/dLZNKN4mI46HE2aQJWO/RWpNnhqa1ZJl+JDswtsZbv5HleVJKoNTx8FieGazzFHmGdZ7xeERhcnKd4b2nyIpv2CX99NQQQ7ayvIK1dr7zUbIDn8wm2+FkjE8R6+wR4C8e/gzf/ZvtvsK0xdSFWmNM02O36RiPh+lCg6MfD40+9A7skbS07XNSEo+8RYEQqtiVDswtV29USyuorKLMK+rRGGMMXid0r8Q2E5aqHuvr66yu7iEXGjtxVHv2IVZXX/TFN930jJ08XxkYtUeHLGc9yrwgSkU16FNkmmYyor+6zMbmJiJEJsMWksIsr0i33HvDV629dGnhes6SA8uyF86S9N1EPJv4p2CIafhnthKc1M18ULXWUeRZt7KGN+ym0E6cTo7HQ1TphHBWax1KCrz3j/RrNp1zvzD73LZtOzTj9PPa1pIZPd/NWG9JJJxtbt0l4cNvyLOcjc2N42hD1QFbnHd47wkxYLQhMxkJXvtIFhMnAhBkB+CgAwjJ6S5MbQNxFEVxAY2emB7KuZ2cu9qeDzvTdnajEmH3hRA//eC1161cfsUTmxAIISGiIIsClWCSPJPUxbdD6yhUTmwd2gtMkASb6O3by2hl+U07ec6D2v6VGtVDvEMoQUgeHwLRBzJtcN6TFSUkgZaSFECYDC7Zc+W/tvr7Fq7nrNhya+1/ng2s7RP+iQ99twIMIaBmZSYxIkWXH5m+LnL4lF0BQjwFevJ4VQ5475HiBBBBemSOMr45RtBanwBKmKHrQgiIWV/RIdAEIu6GMGKM4WXOOaSQGG3mTkxMkXOJ1OXBvJvlwn750dyQ2Y59FkY8Dg5KJ4TgrPVcODmwKIVgep+Zh6OlfGCh8Ozasyybh1OllCf0+yMAE53CQXX9O1uMzI6zn0Hsrh3YZ9zwsiuaqlgbOidnqxeR4rYTlCQBIklUksgIKoFCIFP3M0LRVtmnfsbtawd36rx/58477819+HlvWza21rv4rJFEHzpnlSRITdISoUy3TdcK0R+QVgbXPu2aax6/8D+P2p5R5AVSSLyPJ4AOHmqQnMoCfN1uy0ucY3t3VRW+bR2nr1E9CcRA0kWWfcP5rfnTT00kr5QiyzKcd1OQCXPIfK/q4bwjz3KUVH8N/PNj8P6do93/qZCxXQSjaRv6vf48RKuVJqY47/ftEYNH0owxhBjmDtJ5R1mWKKVw3iG1StPUwe6wWuo3rF5+ZdYkCcF323MiUURIAhW7V5jWaappNjsRiQJ0jCQhUWVJXF26/gtvvenX//jgbR/akZvdNLdvjta/Kbt8/2r0Do3CAwKJTIkkBFZKRIzgPV4ELC29A5f3tsb/+iPsnrwLF2btl/i+tm27ECFTVoTZ/6WHjp4I5o/VzF4FvPpi6sO6aX5KG/VC7wJZpkk+zPuvc/QnwqKVVLTWXg3ccd4mWe+/OddGO++QQJEXWGvn6EOA8aTjDfDe42P4scVoOTvOtj8Y/OVwOPwbIWVcqgbZ5uY6e/budZvrG1ppE0OYM5WI0zA/JR6EEap1nrLs0bQ1RmmysmA8HqOk8nlWFFLK0a5xYJ+39oqvrlf3fU3UCiUMzrUUqGnqdLraiuL4TgyJTAGBIshpdXwXqUVmOZKUb2xsvRX4tztx/u+/6667n3zHwTca4g0IiRuPkVojhEKIaVIUQcQjkRAjRdnHyAQHLv2az7n+mq/68ztf81uLcfSI7EBmzLOcc8QUKTNDax1SPHpqImvtX10snRgjPydE/K4s08paj5HixFonujE2q4UKJJRUKz6GJwH/cD7OuSyr7/PWzkOEXT2bQqVuV9Dajj0kywuatgH4tcVw4QxrEGeUUA8INwIwGg5fE+H/I0bWN9cBOHL0KADOhrNyDuN6PN1NW3B2FjLuPt8eD6WfV/t3V1/ds+XyD1fLe4ix6zitBQhHlB6vIAmJDprM646TS0KUgigiIkUgklIgEdAzJ7Z376c/+eDBl+zUdZj1yavsPfd9rAyBKsumIU+JSBKZum231hpjJHmmSY1na3NCceml1Cu9H18MmUfY71J+rXWWsiznIY5H4rtEmjYx53J73kXWle+NMakZ7Py0E1ua8wMSYqDIim86X6jTpp7gg8d5R7/XRwiBdRalFK1tOxYIpWeT7oeA+xYj5myZyneFoz3fJ+BFXKOqPtEmgTY53jYIPEkEgohdKGjqBESSiNidcpCRID0Ij6A7dk6sy4Xp3gCzsvyqZ9/+2h3hbfs/r33teHlsX8LWVrSTBim7nSJJQhRT+HaAFDqn6xOD3hJ1TOi9K1d91q03vHwxKB7B8xPjLVJIrLWURUndtI86+SAEKCkvOoBNStye5/kJOYxtYdoTkGTWWpRUWNuel5pGY8w3ZCZDIMiznLZt8cEjEGit52jBEELHIqLUaxej5ezkwJASUxh10TuwL7vmmn/bW9nzPct7L6EYDBg3NVWWkVyLVx6vQle1nbrcV+J4nwUZ8SqRRGJG0p1ExItEEBKNIc96g8Pjyet36nrec/Otv94e3fgdETwpdSHNNCWwM16gXUQEB97RL0ps47HWY5aX2CzlnU9be+Xli6HzsOxTM5NdElOcEs7WVGXxoOFDeYYPvbOt0vDMi6w/f3G2A3soIJnWmhDnxc47jtq0zr3Me09RFDjnCCFQFiWQcK0lz3OcdxRFgUA0IYTfWAyXx56dVwe2Lvnv/Uv2DSZ1i3UOYwyTeozOM6Lo+EZkjKgUCRL8NHQYxHRnM+VFTiKSZAIJQUQCHQS/6C8h9q5+3RfddMfzd2xl6CYvopkk7S340IEExAyWKukKRiXROYzsoMmNjyxfdnl+v0w3Lx7JM7dMZ//JOktmMtq2pcxyJnUzL8g85cqSM2PSNtpQ5MW3XGRd+ldt6+4x08X1zImdqo7HOYdWGiGEKLLsm3f4PJ8ap050xneodcfAkemsy3+17byuLZF+B9hajJiHlwM71TCaFSIHF9JF7cCecvDGF4X9+z8r5gVGa0xKpNCi84w6OKqqQoZEGRKFlAQFbQ6jGBA6J7SeHI3QiomrEYWgTg2RADIRpWDoAtmB/Yz6xY992TXXLO/Edb3vjtd9JBw7epuajFnKJQqHDxaRSbwINCHQRo/HklVmOplKvJfky3u/5XNvuu4Zi+FzZua9u6GDznu0lATn0UIhkiQlyE0xxbKC1MUzAyShNElMnZgEqUDI47Aokbr6sBAcddt8925cccqpUwnTmpgZg7g8C6NZSfljwScCILREKDllnAgIkUgp4ILDGEWInpg8Ifo1dhS8UX4rgA2WGYMIxI5FPzict3M2jpgiCX7qUQbUhCShRDphCZSmmjzypByhFlxo4UKREnMqrO2k15yGlWS3lAnI80XW61eWXsXSAJsSUils25Jp06GbjKapLZNRjTKGpp2A8LimoV9USBeo8h6+9cQI/aUBh4/cT9mrULLLNyUB1dIAJyRmZam8P893TITvijq91m2s39cOt0ipc6g+BoSSZHneQba1YGu0SaENvaxCCEW5vG9pXYoF1JczS+LPdgXqpN1BSonc5N1qsZts/rjxzR9IIf82pOnkLE5kfZdSIrcRtVZVxZTA4zm7kzshHdc4A4x5ZFRSp8gp/pIQIh3fhaWuf0QXLUgklOiYOvI8I0ZQXeHdv92pa2/b9kVKHS8qnhWnb8/bFUXRhTi71ckfLkYMD5t/81QObIZC3S01xOflJO4X4bZ80FvOck2SgjTX55SQFCIpog3oLCeWBlFlhLqmsg6xvrmRRiP6RTdBee9pa8sTr/pkms0JGYpoHc63ON8S6hFJJpafcPnXPe11t+3IZPT2u+4aZj5cbydjl2UGrSXERPSJtrUobXApkec5JQpqi9IZXgmWDuz/9E9+2fc+bzGEHpIo8FuFEGK780riOPjAe4/1FiUUg7L3owAhxR+pqqrTO1LTEgckMR2Xe5Cy40e01k7F/vLv2ZUOXCmkELhpvdZZZH74ZyH4sJTyBMYJIcS0qEfMOQZtYynLfBrCkzv1zH5hlmVyLlaa0gl0W0iBkIK6qVldWQXBzy3Ch5x9ihiiuygd2JfccvBZy1dd8a0ik/gUEJmk8Q6dFx2LAopgAyvLqxRlyUePHKJNgV6SlJP2bWZr/WnZaDjZPHSIpUEPhcS1nvsPH6XMKmgje8o+RmnatqYa9MgHJUO82PLhTV+1Q4z177np1p+xG5t/gHfznEFKiarokVIX/tGZwTYttmmpqgq0hqJgz1WP/4Fnrq0dWAyVBy29fMmDMUX0eh37eJ7nrNfj3wIo4e3D0ZCqrOaOqlu9x3l9y8yJWednu7SvBJZ32+W7KdhCTHeQWsuz+dk/cHxHc3xRILZRBUUftu2IHFLK/7JDl/6f2qZBxOPyJSdTPXWlFIL1jXVCjD+9GCznxHVcnCjEDaP+m15dqaIUXTw9dchBhJpDzgmK+w8dwYbA0iUrhNjCaFine+998fuvvvpvs43NtxQkmtGY5arPcm8FHwV5XiKagB91aEahE5NUM0oW0yvJVpYu+7gd75j45QGlXjG8/37X2gZjDMbkhJBIgXmiWWcGk2eMhxOk0EStmBi98pHhkZsWg+TUpuCrH4wKSgjBZDJBCcW4mbwL2AAYw315VnzAWjud/MQpdZ5ijGRGUxQFrW0V8Ozd1gdmSuGTtp3zWbRfjTHO5WdmDmwGjgkhoJSiLHKs7ZjwnfNiJ8KIZV68+FSq1TPG/u15msxkAO9ejJiHtPhQauC7lSprRx3YZx28/tbBFVdcuT6ZIFSnDdO6FrSisS3GZKQExmT0BksURYFvLaltaQ8duvr9d955COCPblt7qR6ODvWV4dCh+5FG433EOyiyHAkEZ5E6MQo1bXIgFUsrewi9/gu+7KZbd0RQ8ndvWPsLsTn64b7ShBSpnWUybhB0mjguOpIGoRVN0xBCQGrD8qWXqr1PeuLzP+faly+EL09heZa/6KFUYH30SK3I8/wHt78nM+oHQwidfHqMU8JQOZ+cY4yElLDOY60lz/IHTJq7ZQcWY6QqC2LibLN9H0rwVzP9ppMZxcM0rNg0LTJ1Do2E2IEw4lPbthVKqhOUBWa76e35MACf0g8ueA0feS3kbq8h3tGTePZrX3tVvHTPKzfqhl5vCSUkQiaiYIoKE8QkuuJfobAu0IwnqNaih+N7/uKOu954wiQ2mXxnc+QYq6vLbDUNe/ZewmRSU/UHbG1tgYi46KHUyFJTbwyJDlavfHx1SKf/tWMr5cbfHo5u/utkMkKVOcvLKx1abjrwxm1DEz39pQFKaJpJy8Ra2jwrNqrsDhZ2si1775+thDgleCNNn6eq6mGdZdK2f8QJ+lfj3+gmuxMlISKJSJpzIoppGLK1LU3bfBGwf7ftwEJKTOoGoxUhJMxZrC0VglfHGGMI4QQW9tCN0BNm9V6vhxAQQ3zZub1m860zBpDTkTOn1ImDSKVx3v33xXB5dCKqp2Pkv+h2YH8/OfpDTZHnvT17cK1HJknyAZNnuNgdW+8QQnWw6DyjNDnZqEHdP/r2kz/vHWtrv5E37S/UoxFCwZEjR+n3+9y/foy9l+0nJA9EcqNIzrNU9bCtI+Ul5f5Lr/rcW2+4cSeu+wOvec1mvjW5TbTOCyLD8RBjOplzrTVZkRPpdgQhBPpVjxjBVH2qyy/7959x/SuevxhOx61Q5rkhhtNqDc3CXePJGGPMzwGbJ33EplL6N9VUzHK+6wqRGLudhdYdwGNra4jWEinBGPm1u6kfZruMzGj8NB/lXDiLn887vY/Sh+O5wZl+mlIK6z2FMRhjGG2NpqHGbpd0rq7ZB/eiEAJFXsx3WyfrpM2030IIDfCXixHzKJGI2zzVbKwpIS6uQubPv+Xg15T7LvmqrKrwPnYkvCEghUYgSVHMY/haSxAR3za4zSHZ4Y2f/uDtr37XKZfitn1hQZwYAf2iY6IWuWEYLEF1xY1x0mICeB9BG6wPiH4l2bfntqffduMTduL633Pw4Fv0cPRXsa7JM03jLRHwoSt0log5GjPYAFGQZQWiqIQ7sOcnnrJDNWwXgtngXmG0OaXaa0ppXqUjpKJx7pT1P0KJn3CxK3ifTdBpW9jE+0BMHTRdTtVpReLm3bQSng3ebod0xmGfh2OHszz7TTGFq7sQiB3JcZcDEwLrHLa188ktzw3AuZJY+UJAKqXmCFEt1VxN2xjTyTIq2ZEgkG5fjJaHJcj8gPDhXAj2JD29GGNXLHkxOLCnrH13tZGLnzb9fp7pfCoHLdA6I/qEsx35ZpZlaC2pmzFGSAaFwa0f27p0Yk8blvjVtbUNe+jw948PHUYRgIQqc2rf4MVUeC1A5iVJCKISXahSKsSeVe7D71jNVTlxL9j8l49uFEqBgrIsSTHSzBizlewEFgGlM9aPbtB4z+CKy5VdLdYWYwyAT+qV1ac5/9AI3hCDB/6AU9cRvbsbi4mqLOYIu16vR5gyp+SZwTo/vzdCygPAJ11Mnd229s0hJFof6PUqtJQ01pJmebApnD9TiuACbesQcE7QiL28/AbBlKy3q3/Ah9CFgoFm6ki7fBgoYxbUUZxT2EfgYnBgrrjsjsHjH7eXJMD6ThVVCmLoajd6RY9Ma4bDLRKRKjf0jOTwh/+Fom1e+va77ho+2Of/2U23/mS5OX5nJQUutrjUIo3uJqUoyIJEo4hS4qVEkohJME6RuLz87C+45cYX7URH//ltt33oCqF/qdncQMhAW9dkpqDs90gC/GSC9B6ZdeKXy/1lSl2xMZzge9WLv/iOtc/moqeO0t88qScdsWx6cK2vBA8Gn94q8vxXBv0lRnUDUiOEZDgaY3THbDFV752GriLWeqTkmy+yLv/DosjIMk1d17gYO/SjEOjp5OFCwIVAkWdICUWRyXMRRpy09cuzLEMhCDGgpMRMFaTLPCczeq77FSMfdc4twoecay6Bx7gD+/Lbb/+cMOh//5bt6pyqvEDpTibaJ1DKEHxXdFqUGYqEsA3DQ/ey5N3b/nztzreeyfesNO47N+++25WZohkNyZRECdVpbyWFmMljpoSPkUggCkW5dx91mb/2q2+//Yqd6OzeMK3JyXhsfCQXAmLsWLS9pzQZ/aogycR4MkEiUFKysrRKb2Vvdm9oL3pARwjxdmAqKf7gWGCt9c8+2GfVbftjw+Fwvmrv9XoYbQghYLSaf0eMETVlB5ZCvOQi6/JN7/z/tNYTYprn3mJKSNFRdBmlup1Za5EJmsaiJV9/ls/jqbNaSiEEWml8jF0eLsuo2xbrPMWUGSQz8ocW/uXcmpLK81h3YMey+LPl0jJlXlK3lq3xCKUELjpCBJNVuNZjm5ZeWSGiQ6dAjzQxw+EZx7DfdccdH+mn8ApVj9nXG2C2KTgH0bWO5SOR5IzHTIDOKC69tLo32Bt2orN/9661j4ujx67JJjXSBkLryJRm0K+IInJsc4O2rVle6ZCUwsPWkQ20NmSX7vv3n37r9RczoOPTY4pCTfMePIjichLgvX/vQ3zeu2dS6NZZJpNJJ1UuJc4HnA/kmUFKyUxiJIR0OTtImbQbLIn45m7CEuRZB9qYWT51+LP7kWUZvV5JiLz8LE9S31hmOSKljttwFr6chjTFFMwyCyO2Lr594WLOrTWueWzvwL7otltekvr9T1XKUOUFkUQQoZM8iYGIpHWeLMsps5K2Hnday96z9fF7X/fHr379hx7O9/3xtTf/sP3YoT8Owy2idRAFMUhCUkR0RzEjRMc/oyBXOTFALQRtv/juL7zxuufuRIe//9Y738D6xl/nSFSUxMZi64YoBVm/IDeK0DaURUae5+xZ3oP1EVP2kYOlN33+DjGJsOtIbOX3Zyabw7lPB6OOHWbqTs6IxDT9lNaaQX9AiIE86xxVWeTT1XzEh8ikbqYksjkSnn9xESbzbq27VV8IgdY5EuC3SfVKKRlUFU1rGY9rpMRxFqVoEnx/a1uyLJvrkmml5sAvNWNOUQJt9D8B/7xwMZzjMg7jH7NxzGddf/3+ozLdvnfPPtbrBu8CWb9ARY1rGvK8wAdN0zgKKSiynNaO0TGxfujeI3/96tcefCTfe4mU3zts3HuC0KXKs3lYaVb3Ewkk0ZGTtqOaPXv2UDdb5JfuU5O2fstXrb30Xb+19qPnnDctH7Xft37oyNvL/Zf2l8uKjXbEOhPyPKefDNEGlDRsbG2R90pciPSkIestm/Hm5Frgei6+wsoXOOdQUwcmpoHheErFh/iLZxQGMeqnnXPf2bQNVV4QQsCHSAjtlPdv2gRkuaauW4wR329dehkXlbhhfIMW8qU+RFaXlrrnUhusdxipaEPATSZdPVq3gDBZnn99XdfvPgtf/wVVWUWRkqybGqMNKSVc6LQC+1XFaPrdrQ+k4F/Pwjj3hfTu8cC/MZBXy8tisrmJg1RVldieP34YxeHbhxzGmDSZTEbAP+34DuywiXdm+y5daaMk1yUGjWs9iEQIjhC7mqfBYNCR8TYTZAxM1o+2ZUz/+ZF+7+/csPbB+r5DP2yIaDGjQxHHYTMpkaYhiLIqWF9fR0pNEJKlx12xerczO6La+gd33vn7sp78ip9MCG2DlrILzUylQXxrqduGwUofnyJZniO0wUvNho3f8bSbb378RTZevhqIszoUPU3eb3de4fiuTHKG9T/OuT+JMSYpJDFGnHeYaQ1YZjQJ6IjWwTYeKcG5hDF8wUW1eJD8Tx86eq2tUZc3bL1DTHOEZoqollJC6Grqmro+K/nCXJn/bK2VdVOTZznee1LowCQJGE8mZJmZkxr3er1fWbiXc2/9orxVI/5GKv3B0ebmB4RUH8i0+WA9mXzAO3faFmdHHz54UvtQCOGD0YcPhBA+MJlMPpjn+dU7HkL8rFsOPov9lz3P95aYBENyOSZmqCCIrSPPJCSHlorRaERVlSgRaEcbxOHRn/2zm27/7Ufz/R+67VXXtUcO3xODJWCJ3ncPe0pkWqIQSAEWT9YroY1or2m8Qq5c+q3PvO22Z+7EA1A6e5P2bdsES922VCmjjBk+CmKeIQpF41uQjjY2jEXAVTmrj3/85daUP3UxDZZeVbyIFKUkopRASojTn5USpBQpqxxtJCmlh1WgHhGvk9p0YlpC4n1XJB2cRwFpWicmBYgICghOfsdFFkZ8X4TUOI/JS1IX0gUkafpPbgNZdNRcxLOhaO2ie3GIHiUFwdmuVEZ1dWmZyaeLDDPLwb1jPB4fOqfOfIp+nS2MxUkh7BnF1W6hWuIME0lRdC3RMZmc7rpnZpsaSYIY0EIgUiQFhxbdGDldk9OjSBGRIkzb9p9TjEgBUsZ6xx3YMIU356urIkpNlIooFQiFTBKZmLNIh+ApyxLrGjIJ/ZTiUu3WzsY5DEL6tnZzvS2MwmSK8WRIkedEn7BtS7B+KsXikEmgVUZm+ixdelnvkHdvZkeEL+/4yNbdd9+ikmdpaQlcoh7WGJ0jpw4XEbuHapq/i0KS8hy1vPSsp62tPYeLiDqqe6C7XIdzjkxrfAj4GFBKMpk0MyaKX3p4H6/f6pyjaRt6vR5SCuJJI1ikE2s+JemFXHSoM3lnkRdM6glKzkOFD+inlBKESIwonZtHpWittX6qlGIuMhpTopwWsS8PlmldSz49Jykk3tq3sGMF5umh/s/xmM5JT4VVU0Km7fefE8iA52TLs/fPnJgQc2empu9X04WiOr66TDvqwJ56x43X7Dtw2SfKkJAxIFLAK49TEUlCRYFIEhEEZa9iWG9hfUuwLfHI+vV/fMcP3nM2zuM91629O65vvNXVE5SRSKOx4xoZEv3eAKUUKoERHQF+kgLvHEkK9j7uyk/5Nze+cm0nHoK/veu1r2o3tz62fv/96FyztDqgrmtkAhlBhuMrOhNAuYjUCrnck/WgePPFMHFmmXq2tZ6YIIRInhnyPMd5z57V1akacSLPDUJwD/CPD+8b7F8JISjygrZtCTHNw4anZuie0uko9dUX1S4sxv/ZtM3MsZxQuhC3JTG2CyG2rfueR5d7898SQopKdmHjBAQSISU2h5skoGkbqrIiphgC/NYu6a4gpcy4aJSN0oM69LlWnzg9+fZ2SrjZy3EnUYhPvf2Gq5oiP4gxIoYAMZJweOXxKoAUKAQCA1OtrqLI0CKRRuOP/OnNd77mbHbqlUFf3Rw6MnS2xmQKYwzBxQ5aHxKahELgoqV1lrwoCBEocwb7D1z/9Ftv2hkm+CPr36ld22wON6idJdMGEQVaaIgCEbuSABkTRE8g4I2kKcyBz7zj4NpFMDpenmW6Y8wAWuu6PAhwdH19XsDatg6j5CNK4CulDnrvsc5S5NmcXuohdiTfdpFtwv4auFcrTWvbmSrvAzLyM2kTKbuoLI8ijJgiL8q1kkopGtsxcFjvKfMcOZWjkkLOQpe/yS4QrpxSmql4lvVtuEC0WGY7rtlC5gFtvtCZclie/PtplKDPuQMbxvgj/csur+JM4oCIIBBFIIhIFJEkFQkFSWGTI/oGXdfOHzp61qHJv7W2tnUgyedRN84nz2Q8ZHV1lbpuybMSfESmiMkybHIECUHA2LU0RpmhljsC6PjA7a9+h7vv8P/OckUTWqoiQ/iIEF3EWHuB7qCUBNHtqj0RtbKEXe6/7Km333DVY3hsHAgufI63nkndoJWkV5U4H8i07hjZp2zpHbAg/tojy/HYXwghsGd1D01r0UqeCQDkm3ej0CXnVMYm+xEhBFJMxUBPdl5TIIdSijzPiRGKovjWR/h1zwS086ErUs4yQkrkxlC37Qmrd+dd8CntqrywlDLwmCZP7P6lbYTa23dYXWTrgaHEGc5wtiPbThYM23JxaQcd2BfecsPXpqXlrzgyqfGkKR9ZQoiAxIPwBJFwIhFF59oyozAxMP7I3W8+HVnvo7Xfv+7G/zW+5+O/pF2gPxiwORmRZyVNXZMrjW+63ZnKM0ZNjU0BXZQM9u4lrfS/9HPvOHjdTjwMjxP594/vP2KLTLG1tUGWZd3kkCQqgoqyc/5TZvQIOAGpVy5tmewnH8Pj5Nv6vcoV05IIpTqlAqMVznvc1HlNF7sf4ZHX//xjIvmtra2HKaypns1FZI21v+i863hFt007x8sNjk9mTdMiJbRN852PyFnm6ltnnJRadQXLWkqsc+TGTLXQKmKKlEUpgV/fNQzuCbz38rH8LERxyvrLE8KBZ9ZODM3PHd1O7cC+7NXXLIc9Sz9V7tlXDFZW6K+s4rwnyYSQoAiolAgi4qd8hAhB9C3u2JH6KlOdUxaMSy1Xt4cO+ygiY9fiQiCXGbmSRO+6nJNWiLwLbUafqOuGUFawd+UVT7/ttnPOWP+OO+742JL1r/XjIb1+yZYdE1JECYFJCpESloiXXZZTCUGeleiqT1pd+fLPvOP6//BYHCRScNdoPDFKKaTowofOuQ4yLTq2eGNMt/o7w+Ll04cE9TU+ePLMEE4TQjxZviXLzXdycdmHq6r6uzn7v3hgLmPGWG6k7Fj8u+noGQ/3i5wN35WmnJRzWi/SCVIyo3pCQjBp6jftto7SWkoe49QCAUFKgjQ7JkFEEpIgxa6F1LV0UgM5PzL9m5A6UEKK8yRYPOcObCtWr4n9/h6KjNHEsjkcgZKdPGDwiBiQUwG6JAQI2VXOH9tADYcv/K21tXMat/6jO++8tz+y37Nx+DB5v5gLSdZ1zdLSEkIkGtfgkz++qpCalOW0ZbY6zNOO7HA+eO3aDWZ9vCEEyEySou/Qb7J7EHwM+GkNG4BvHCEk8j0r2LJ6A4858Ab/LiVEnhnGk3qufTVzZjMl3rrp2Px78PZHB1LwvyKFpLXutCCOk7jgaNv2S4EDF5MHG08mP7I9cX+yU58pIqeUKKYLjKoovu1hsis8dabJJoToPg8IMZEZg4sRnyJGm2m6Qv8SO56affD/izEmHtNaYWleQhE5/vNc1ijFB2/TRdDs9+1/l0hn7J8elQP7yrW1z26r4gXjBBPr0Fm3te+kUTTee/p5iVYCQSImj8o0djwmH9e/+6GDr3nrTnT2H6+tvUU29bvsZExMnhBcl/vyHmLCTBPCQqQucItE5wVBZ4TlwbOecutNOwKbrtYnzx+uH8ETqPKMYB1JSUy/QiLItelCaEqzVFSk2nd8FFXvqi941W3XPKYGiOd5WnQr8CmXbjeRhdgRHStF09puxxTTH47hvkf5lf+SUvrLk3cUp0jQHyf57XYGX3WR7cJ+9fgio8tLJdEhE8M0CU/qwr3WdcAY78MLHs4XmLz8luSn9UXTZrRCK4kLfv48TGVzosf/PudTQOtkEEM3taYLCip1Cj2wBwJ0xDzHqZRiRgsxe4tUepsr44QWT9MS3Q7u5NyaFDKEkMQ5dWD3i/jaas8eXS0tUS0tM5o0LC8vs3H0GNF5jFRsjYb41mKEhLalHY8Q1treyF+3k3dopeX5bA7b1LYIcXylIIRAEFHTYp8kOqiubT1CafKVZfSlK6/+sh0QlXzPrbf+ihyNfo9gsW1NIJCU5OOH7yPXhra2ZEXBaDJm6+g6S0WFCFCtrHBY2Nu/8gfvuOwxhD588akgtjNH0lo3DysWveJnzsZXLi8NfkIrOWd1OJNV6FJ/cB0Xl91njH7XDI1ojMaHSDutzauKAjGVNckz09X6KPWwwoht27x0Wgx8gojibNUOkGc5zjuKPP9BFraTCMspR6h74QBWw/G24oJfSbCSYIWTWjqpRdKDtpDivkEc3HzOHNhn3XjNi6orLv8SXVQcOXyUtm3Zu3cvo60hqysrpJDo95eIUlH2BijreNzqPkQ9wd9//xvec9ttH9jJzn/fjTd+ZKW2tx5YWmIyHhJIJCG61Xykg6kT8SIQJBghyZKibizOZMuTx+3ZkTDdysRdVzQtwVmyXo6Pgcv2HyA6TzHlgdNas3d1DypBO27RynDJEz5R//P68AceCwOl0MUzgWz7buhU+l9ZlqEzHcbj5qzQB61vbb0txnhGUrNSSkIM1HX9ycAncnHVhL0lCYE2nTrz0qBPmedY75k0DVprtFS01lGVFSkllNIvPdPwYVUUJ+XU0ry4fLaYaW2LVIq6bf/7wrXs3G5zm/J5sw6bnLu2sc765jlxYM95zWsGcWnpVeOO+owrL78cYuLo/YcgRGSSlGWPzc0tpDJEEq5pOfqxu0nHNj78Nzff+vLzcSP+5Nob77z3b//uI/uXVzpWkOlgkAiIHTloVJ0sfbIREQRFViDLkqER3/i5tx/8j+f6HP/g4G1/Jg5vvraYliLUkxFt3SBiIvrjkPHJaIvhcMhyf8BwPGGrtSxd9biv/fyD13/JhT5grG++5+Rd18lgCiUF3nus9e+cPvBnZXchlfyQlOJMasfo9/r44JHwdVxUumzhN0II3jpLiImt4Yi6bdFSMsMuuBDmE19rW5RQz11dXV0+g9zi8+umPqEu6OQicqY5yhBCC/zVwsXsbMFySgljjLxg5VT+ZePQ67I9e5dlUXLs2DFGG1uI4Fkd9KmqCmst9aSlbh152Wc8rullBRViOBg333M+L/gA+Te2hw4PCbGDqZ8QmupCR0oIMm0wSAyaprFkg+XM94uf2olQYrY1ubU9enSjGY2oMkMvz9A6wzmHbRqUkAwGA8oyp42eouqTpKIVsmpXl3/iAh8nS1Kqb5k7r5ggphPyTx10PuFijEpz11neXqyl9NBVKCEERuNRt2sw5qUX2Vy2afLsdxKw79J9Hf3PVCXAzyROhKAqOnYTrTTOO7m+vvXFD00cLF+QmWx+r7uiaDEtzXnAoua2hVs5P6aUihekA3vWq277svKKy7/dSjBZQW9pwFKvj44RDaToSUKgdcbKyh7GdQOpewCPfOyj7/uT229/5/m84N8+eN2f+fs3fkY0LoSQjkM2RSJJUCRU6nJiKSWaxtIrKpJLVMuDSzd72dq5Psf33nXXMG0MX14l4XMlue+++0gpccm+/SihkQjGzRhdGobjLTyJvKjor6zSGvPEp9x28EUX8Nj46jBFrZ5uFzYrgFRKyuD5qTzX92stN6qquLeqivuUEvcZY05oeZbdm2fZvaXJ1nOdfayq+ncPllYO93qDj1dV/0iZF1u5ye83Rf5TMT10FFGpjlemyAtiiJcNsuxTL6YJrGmaNxhjOHz4yFRdXZ0gdtkhRJvt+m1Ca/28h/jYp4UQsM7OUabbyHHn4JkYIybLAvDLC1eyw8XL0zFpnTMXpAM76ps3FXtWc6kznAuUvT73338/Mib8uNv6hxSpnSOhMCqjLHscuv8we5b2PG83XPQeo24Ko/pub12XGBYQpQDZsV6I0DmvSdNB7aNPVBiST7il4rue8aqbPvdcn+MHb3/VzzSHjrxXJNh/5eUoYzh06BD9qsdwOOygq0qSVWUHlAmJYd1Q9PuIfv81X3377VdciINkUFYvLaYr8O07ru3OLKWEFMwmzCfEEPcVRbE8mTQHmqbZH0La773f773fH5zfH33Y750/4J0/4L1f8cFfOZlMrhhubV0yHo8va5pmb4xxIAT76qa95AwiiB3Ctt+nbVtijGZs7f9zkc1nv+WcawGqqsB6T2stGgjeI6Wcwcnp9/tIIQnBfx0Pwl5S5Pl3KKXQSp+Q85w9BymlKUtDIqXkgX9YuJVz7wfSA2shUy+vLrwd2OcdvP4V/csv+6RJSEhtsNYyGm2xum8Va1vKfsXEtuT9CpNnNE2DQjA+cpSe91f/0Q033LsrRt7a2pYZNy82jfWkRBIShIYkpgMnAJHV1WWOHDmCFhKBROUVvUv39zYkP78jjjbxsq3DhxgO1xnWQ5ZWlhlvjLlksA8lNKNRV3NXLQ/mlb9RS1K/Gtzt65svwIF0oGmaL7DOnuC8ZivxeRI5JoppeEoC3kfqyQTR8edhtHzABCi3PewSKPMMiBilESlincXZdr67OhXT+vajlLLr/2luQCl908U26xlj3pZlmnbSzPW5pJjqpzmHUQqjNMOtzansTQDU0zkt+rD9Ludd5/xSekCb12kKgXPuvyz8zk6LyorZeBQzYucLxoF95tq1nzDZu3xHKpYJMSMG0EqgBFjXosucoWugMIyDwwXXkW5ahx6O/+8HrrnpdbvpZrx37fr/HY4c/sVSGUga34IWFT4mhFJ4HKNmSK8qAIlThlYaJh7Kvfuf+PTX3HLOHcQ7Dx78UFZPXmfrESnTNDExyPfgtgImGoqsJKaE95aYulUvuUEtD3DL1Xd83toNn3OBjZHnxBQxWk2BNWleGEuMiNQRMBshaeoGIxWF6aDauTYURncaXr4LAW9vM/2hWfO2wQggeuRUAFUBWoi5BpgQHQH1rHXLmEiKgRh8pz9GQgjwwQutedrFNKE5534y+ITWBh8iWuo5a7wREhcCzttO+DJ5skwjRHzeadCHz5BAphTOWcS0xCXGRIgJU5iOhFt3ZNzA77LD1ElxG4efeBC+o47QGHUBbb+SopO6InFCnd1MJjlNeVhDCHjvH6qWe/c5MF8WbzJ795mtxlNWg+OCLel4/VSSsrvRyaPzDNtMcOvHXK+xL9iNN255U7xk/PH7xqm2aKERQpDnJdbaTixxqkHkUqQh4YRCKA2mwJW9q5+5du0Tz/U57nPcaprGXrLcx7mWxrYUVUlIETFN1swmV2IiRLAxsHTgsqzW4mcupAkxN9l1RZ7hfDitPIOY7qiKPMdP36eFwFoH0/7QUnJyJeXJhZXHf4knvmdb6HL7OQghOuQqAjlFQYYQkLLjvivLnOj5fy+yRfnvhxhcB6VfwkdP6zqH5VOknxcYrTt28pCIHUPH154qjLg0WP5PTMExqqMGw8dIPuXBnEwsea5obEBp3g98mPNcE/XgVGMy8VgHJV4oDuzzbrj6a2VZfEluMoSSTOoRQm5zXN2wRieFiSB8ZOLGaJViWt/4mfetrf3hbrwD7/6htY3ltnnewDuU8IwmI0aTIf1+v5MwQRAkWAVBdoz6EoGUit7q3r6r+j+8E+HObDi8bvPuj6GDRfVgKCa00uHwGJ/IopiLCYoAwie8i+iqetJTD974XReE88rzT/HOfkLbWtSDJKFmo6ZpW4SA1nt0ZijLAjulHHIdCd9Dtgc4NXGiblEnItrpxc1W4LMcjFKKGGEmmuG9Rxt1sXEj0uv13wiwOdwknyokG6VRCCZtg/WeNF3Vh5AYDHoADwgjbg43v8toQ5yGZ4WUmCnbSpFnHTFwG9AaEPz4bgyvbc8RXVCKzDxGk3cAT1lbq/zq0htWL70sGw7HlGWB1HI6kYOUmiTklKQxISOoFMlJjI8eOZbDK3ZzB7zrxht/qb737t/LkwVhKaqKrdGwK56c1YWZTkteiKn6aJIcXt8k9Zef/Zk74CD+9MY7Xt8btx/bW5YMJ5s4HfAqkkSHmNQRVJqqo4aESJqiKFnat9+EpfKW566trez6+HoI3yGFJM+zB1e7JRHo7k1RFBR5RtNamqaLyRdFjhRnOumcJPVwYpz/AcCRjqut+27nPMYoiiIjzw3OBZwLaM2XXEwTyGg8+hmfok/T+zHPOUqJAHJj0FohpSTLNMPhmDzPTi6leUae5S6lhJZd6DGlhJuS9lpr5yKaMUJw/M5uZKnYdhSPdUXmLuRzATgwl+pbxJ49lx0dT8iKLnRlbUOM/vjKI3UOTCRJElP579GQbGvr6vefY7LesxJKTO3zj9zzL1v9vqaNLVm/IohOHyxKSCIAAcmUYkgqVi97HL4a4Pbt+YnPuP2Gc474y7fq54/vvscpEREZJDUNV6TOeanEdNeokEKzNazZai16757LPpzqG9j1xbHx+pgibWs5FQ3qdv40AK0ko7qhcXYavhboTDNu2gfsmB5OC3RggUg6rll0kn7RbCcYXKBtLW3rZgzkGJ29iIvL/tJoo5RUbAw3qaqKNgbaGFBa007VA2YioWWRYVv7H7aHEXNlvrm1rZnVkfWrCiklvbJkJlFirZ+pBvwuj5738pxTLcGFkwN7zO7Avuiaaz4jDlZfKnoDgjJkVQ9rLUVREAnH8wUpIaIEoboEZwj4+4+88y9uvuOCoHl55y2v+egq4pq4vhWzJAhBgCpAdCwiKUTwASIIoUhKc2w4JhYFvshRg9X/dq7P8ffW1t4phls/W8oI3nbcjaKjw5QJVJzSLQmJUIre8gqyLPGZIQ6ql3zxHWufvYtvwdOAoJVGKfmgO6gZMagNkcGgR4xdsessjDdDCM7Cew+3pcScuiiEE/n3up2ZmBO2VmWB1mobtD4ihPjGi03o0vtwc4iBzGSMpmhQOYXQS6BXlZRFjrW+I2fuZp15GNEG9yLoqMGMUowmE1wItG3b5TynUH2lBM75H91NIcOTdcCO18HhuXgYpnanA6uL7K7lAwfykBS6KLl//RgqM4Tour8UHUuCSMyLDRvvsOOtjcGk/a8X0t340+vueJM+PP67ymX4SSQkTRAGFQ06GHQQyKAISeLQiLxk4hP91X34LH/W562tPfdcn+MeYW4O68cSkzE6xg6lN90xSAQqTcmIBaxvblL2+vgkGVyyvxxlYteSnhqpXqKUUj54fIgnFK6eULx8fAeUtNJpazjukv0h4UMCoTqVARcekN860zYl/iBACpBmOkUROdc6ikiM0UzqhugCmZJkSiKhU/xWp4eK85iUlI+/aLTBTgEcRmuklB0rhxSMJzVN05KpDkGolEIJnj9FHz5dAv1eRd10sjladv3rY0RrRZFnTCbNDH34B7spZPjAcPQ87KwX+6Pz6MCefsst36H27vvixiW8S2iT0R8soVTHQTfLEaSU5ryCiEiyDWE0/qH33/na/3OhdcZ+oZ9nDx3xpdAdfDslJAIjOliwIBJSxM2ZIhQpgS4rIfYu/fgzz3Gu6bfW1u4uJu72fkyYEJjH2rYlkEMSJCHJy5LNrRE+JqLKKPbtffrn3nHTN+3GCC5C/D/OOzKToWSnsJwebMEP0gUvT4ZlhBCEtfYM4BsP3VIHI5MJfnN7eGjWju/2phRKISIF5Lkhifi9F9k88g/Ou49rpZFCdIK2s9o7KcmMnjKnKIxROBcQUjxnStj8XKMVo/FkzrQxl86ZHpvWYowiy/QvA1u7uUZqmwMTC/dynhzYV62tLU0K8zq1vFyCQidF9Im2bbviUSnpWLvlXEsrqURbT8hduLdMkx+6EDvj16+77s+qZvzDuR2hpUPpSHATjAxYNySmGpNFQmxQEnQIxMYihGBS6AObfX3O2eD/6LqbD2abzeFSaKJ1hBDIqxIXPC51K14/rZ3K8xytMxrv8TpDray84fPX1pZ2U5/nKv/iGOOUqSEQYyI3WVc7BOjMELqi2Q4VCDfsMIHpG5Gd/lhM3e5wBihQ27galewmrmAdKaSvONdhRKVU1z9aExMPS4b9nEwkUt5lg++K6TmeQ5zVDcmptlun8wUiJqHgmxO8zPvQ8SlK0f1NisdBGwLMFCTjrX8ruxRPfnLYWUqdLmRI/JxE+TRAql0Noz9k61ttke0NSjEYDCAl2rpmaTCgLEvquib6DlLeTvW1SAHh2tqsb3znB659zSYXqL37phuujuvr9xjfMhkeY2WpomlGFFWOKiQ+1GQ6omUkOEsuNWVZIoqCseb5X/L6O59xrs+xqttvX//4oWbP6j7yvOC+++7DVAVZr8S5FknEKEVbN8SUqPoDPIKhC/uqweDa3bWEit8npSSmTgi1X1U01pJl2RQYoSmLnNY6fIcA/F87fIZ/BCTnHSvLK4TYTcjioYfv157Lk3IhoGWnIG30+ccLuBh/vciLrqBZyTMFgb71VDPhTIVZKznNSya0liHAr+/WZNAshCinCxnv/QUN4niwBZFAwJmpDu28A3vaLdd/SbtUvvCSKx+H957hcIhM0M9L2tGEjY0NDuy/nDKvUChIicwoRGgZ33vPe//8+rX/faFvS6u2fR6bW3E1zxhubFL0KmpviTIRk0XgIHmK3NDUE0IICK3preyV68H91Lk+v9+84ebfkT6+bXNzCxWhMBlOJBrvMEZDikTvO5VcEnXbkOuMSw9czhHv/svTb7vtCbukqw9Y575cKUVZlNRNzWgyIdOaxlqMUownNXXTooxCa/33wD/v8DkOlVL/o8gLNjY3qMrq1CwM053YLISUaXP1uSwUVaLbaWs1C+Wf93v54bpt/lJnGhsemiZPpJN1pk4MxSUpcDEy6A868IePb7lQqJaEEF5r/diG0UNgNzqwTZ39+NJVj88bH6jrlizLWBr0CN6iBPR6PY5urCOioq0tq0vLJNeycc/dzbJI3/FYuDO/d/PNv5uS+m8qZQgyJnXCmCXGI49RBSlAPR5hFKysLNE0FhEFwuRUB6544lPuPPd6Z7GXvWJ9YwsXE8vLy1hraWxNnmlkiggi2kiU0SAltvXU1tPbv7/azPSP7pLi5a8XiOS9n+eUBr0e1nuKLOuYUTKDVl140Tr/w+fjPGOMb2nahsx0kjakeCZkv58GfNK5cmApJTKtCSGewAjC+eVGfL21Hikf5u5luuLfnmN0LqCUZDgakmc5wH+/gFSLdUrBwGOYD9GoXQFSOeEkvnBt7Wp72aVP2vSeLNZcum8vzbjh/uEWwmh0oVHKYExOM2lZ7g3YOHYUOzy60Wv9LR9au+vjO3nyT/nBH7yMVhhpQjIpl51mgz/tMA4pSYCotaABCojOCS1EDDGGmFLNYODCZCJcr//jE6u/ozfYo0ajEVHlDPIKNx7TrwqqfqJpJrS1RCqJlDlZYTh05BDLj7v827/wx1//C7K1KmkthPFJWaVCFoL1JtVA7pSQwaRMadHB6pyITommmBf2plwp4VOSWoiopIz33HuvHeSXtqpo07p0cWnPnrcn5HPGTQsEBnlOaBukEAgjaYOlbSzLvT6FzhlujQla0ObmK5+ytvbcD6yt/ep5FkZ8iZRSxBjn9V3j8Zh+VTGaTKa5HokPkdilxH71/MDE/e/nU/XhEEOX950KoAoEaVpYvX2QJyISvibC689V0sJ6v83JgpwKzJ5HbsR3CAEmy3DWdivkdGpS5DTjDJzuw9IJDgyKolsszBcN8N7dnEBKJ6WFhBCP9R0Yu8qBPfv226+6J9fXZEsrKBJpYhlubqKU4tJL9zFqa2rbovMM17SUeYVzXRhNjeVH//zgHTsC3PjGV796+Z8mk1fpKz/h+Vsh5E45hBAYY6JSSsYY8d6fklF8OyRbLxu2trZY2rOPra0N8qLA2hbtGsrBgBREystcNLWjX67QNBPyMseJGhsd3lmEkUitKPI+6+sbNI2iKJfY8s1nOCE+LPMsE0QHxsQMgjLYTIBQaCvRUdDJwkmiLPAZOClBRAqR4duGFAJCQEyS4vFXsukjWuZ415D3B/imxdkGoySFlNSuRuWGkW3p9fsImWgmNYgMkxeYKqcoMl27j9x5vhzC1J6QUvrUzJhO30kpnHOE4LHWkhtD6xxLS0sc3dhAK/UB58N5K2AVUrxJRPHCWV7O2fCASUxs500UEq31S1tnX3+uzqnIMmpriTHN0X3n2e5TSr23aezTlHx0+ZfZGLbWoqR+3Znses//7ottemlWPVZ3YAAxhLirHNi9xJ+c9Ip91jts61nWht6goh2POHbkMLJXdAS91lIUJYWuOHbfPShfH/OHjuyIIu2Xr6195t+Pt35N7ll9fLXShxg6ZFPHTSdn6MgYzSlribabDZ69gyWqpNizdz8TYcmXL+Xw/fdQDjRuqxWGhkRGaMf0S8P61mEGl/TYHB6lKCq8DxhjWB9u0ssqMpXhYmDvnj0cHh/LBJHCR6MQeAVOSPy0/ECVnixGTJDIJLFS4iVY1ZVulglMlYN3ncOtKjZGQ8qlAU3TUJRLTNqWRKTIc3IXYNKSSYlLHl1k1L7F+IRGduhEEqPhkDLXpH715Cff8sob/u7mu+44L7Frrb8thjCXZdCyIMbIUr/P1lSmZNDrcWx9o+M39OG81rE1bfs/JOKFSipa2562/uQ4FZUghPAJwJM4B7pVSkpaazFaEULYDc5rVtT8Rq3l01KMZyxTf3IOTEpJ6wP9foFUhrquf27XV/SelBbVSq9bb3kMKxG0u6bvv+K66557/4F9v9xesqpkXuGcY6A0sWlQKeKCRxYZUmc04wajMqRUyKZh9K//8sa/u+22l5zrE332K9YO/KsYvWvPp3zKp7DcU1uTGiH13FHNwg8zJdf4EAMohECvKBETT2NbUs8wDhOKKsPVE1aKAb72aN0nRqjrMeWgYJJGHT+iTyhpUGgykRPGLb2swrrA0E2QKyWQKFxARbBSdPRUMOUwtOiUyHzHWxCExOnuPQDKOTIpkFMF2mQMYaoSHZFIleNi7MIwzrGsDNG2BA01gWC6BWAlDLENuJjIqx6bTU2vzJChJhw+PM6PbX76n6y9+l/PRwRRCCGLLMe2DVJ0SEQ1I3MVomPcFwIbYkywAgzP82BJM72rLs/IvBYybXMsYlqw62JASX2jj/6RLBI+WQr+UU4h8nMdspmjmPIMNs6RZ4baupkfON8kskvAhprGBh8shPiAHazoygOklLgY8D5C17/i/JYI8HISr5NSdJIj267jxDd21G5IQUrimDb6b7UySCVz570AYlmWjIZDoiAWRUHTNMJ0UQihtaZt21QUhXiwa04peWNM0TRNI4VUUnUTntRaSyl7o9HoW51zHzzjXYyU14oUXzV7gGa755SO36g0fe5nC7Q8z/+1rpt7qn5Pt20TkxB5WZZhPB6TZVkCkrU2lWWJc05s31CkqVTOQ+QSk9aauq4py3KmxJ2EEB7opZTKtm2v1QBNv3hjtXePaki0riGlxJatKYsMZ33HPB8DeEtZFB3qTnrazSPNFbZ3/d/twEM02mPu2LP3k588MoboElIZ5LYCXk4QPHxojKdWitZZlAJVdrVGPV0QLRSqorUJIRU2dKFsU2S44NEiI4aIRHZZmSQIWIRRNKkl6g4VGHwnijl9C4LU8RV2ilJIVCc7oY5PS7PKWQChDDFFopSdYA8gkpgmvSUET4aEFBFK0cQIWhEFKKEQoXtIfIoIqVAKvGvpaYOMASEUITO9RshXAi/e4TnhCzVCCkSHllS620WLaU2hEIQYEaorgC3y/Dfrth5y/nMdN4YUb6+KkhC7mqaYOgkXOXVkc55TkQhAiP524GE7sBwIaVtx7An5loQgbdt1dSwgJstp7HlfGG/lWf42AV/f2haBYDpBd7WjM/IDIaZyTOKEgqMQOs2pJARaaVzw1+4CqpEkEYgotm214gNWCnEq5xNDIsEe14Yvcpx4P2w96Qq0YZ4nnB3nuxtrz3BBNcvAig6VOg2zFlqX7mFFQ6RwNiLn88wD+aLELISbutq88cR/QoRP2Boeryu3bXet3h3/9qF79KnAU33GtDB+IP/9wYNfKVYHl9XBUxU9CgyVztBCTxmkM4wxBOvIpGY8mXRKzMMNiuC/952vufac13w9d21tZSx4/kQmsn5/G34zPuoWZNcgzolxRZLbNM78vM3fE48r/s7LNqUnzN4nPCp5VJr+n5h+PhFB9/p8JSogie49guOt+7sT5T2OP1THP0uccB3H36uEnH9HJEwni0DwDW1b03hHvrxKWqr+02d+//fvKFt9kRfPF6LTMNuunDwLH0nZ8Tk63xVq1239ZnYHavKXAKzv8q6o7lzllG1dStmx0sSOSX11ZZmsW84+9WxTGAm64mpBx4yTZ/lcUfp8W2vbt3rvEQjKssQ624mSzgEus7CheDDVX3zwAL/MBcYKIadF7qdrEtCd6OUjOnZNoqVEComa9dmsRsuYRy3eeSYlEGIqAnuq46O5vjM5zk5Vm6XeJx0JgSQUyQbkyKKkpMhyQt0l1cs8p6xyxpOG/vIKMkUOf/jed/3pK2/66R1Z0sEnlqt9QlViG9uFb2SYssTD9qq7lB7rOnJnngmfhVWFFIipvEUKoVO4LUvqukVn5VKxUn0i8MEdOrPl4P13dioG25Bn4kQ6Hu9cx87Rrfz+aFdMzG37j8aYf3XOfUKcCRNsC/HNg9bT/PZwOJxyO/LNMfK+s0mWmgS44EkktNZYa7vJa3cQJLwjxJDyLBeTekKRFzRtg5KqW+FLSYrpQQlyQwhkJvuL1tl/ZtdyQJ452e8pSwceYesQm6eoP0M8qpCr2BZFOBNn/WDnKBGcFU6307Qp2ZyQR+vhJabXQ2cGEaGQBYUoKHVO8oKUBNZ7Nja2WFpaYXNzk7s//GHfk/qVOzYX99Qea20cDrdwtiFXsx2Q7HZL2/QwZq9d7G17f0jU/KiERklNGyNJaRofsMgd24FVVfX06cr6uPMinSBFEWMkz3NiihRF8abdxH+XUnrjDN6vlERrhdYaYwyZ1hilMFMErDGm086T8vsetrM8g8lmVgDe2pZEekjg0g5aI6X8CTENA4YQWF1ZxQffRXNieMhriymipHgzj1FLqSt8fzTHk8fMbBzt2LycTn88G9f3YNc9OwWdldXfHKkbag9Lg1WU7gg5R01L1u8xtmPaFChXlxg2IwqtUNq84a9efvUHduxmt83fmbJn+1VVqKxk4hwSvUsWm7tf7yD5TohxvnaSUI8sl+5ZJlR9/P1b/7JT5zSZTF5olGYu0zuP5Xfmp+AbESNKKpqm+bndJR3i32aMuWsW508nM5Kn7jWtFXXTdgrDSgA8E3j3w13hq207ru3fF6fyJXVTU5Ud/ZYLXuyaZ0/KtzZtMxeudBvr5FlO0zRopU8LsorzVbZg0ra/ulsuJ853Fo+Mrf5sapHE7UXTs3DsNJkgzrLUyTxFmR6ekGQ6h1ot2zfv2vn2T5f0gF7Vo3WeKCTWR4JItPUYkQtyk0PrkU2LGDVHmvsO38SOqibfcc/TX/fqu5V3T6xTlz5VMkMuQoYPOoC2hzFmD3pXoyTYv1yRJjWTo8fu/fvbb/+XHUSoPdvHQCbVqcMX21nfu3PebQWs/+yc++s8058uRIdKm4dqpwKjMzRsZjphTec8RZG9qGnsu8/WSUjRIW2VVEzqCREw2mC92y2O/n1yCuDo9XpsbWzQ2hajDc534eEHe3a11u+33t3HQkzrlH+bto2ZWehQTEE/modNjikuULGwJN977S0f7R8d/vigtTFrapSIJOHJCo3WkqYeI6PHb20RDh0e790aP+8f7rprxxFh2XB0TXv46AbeUhQahCeKuGgP0gJh/jMKhO6yrJpAOLbOkX/4v1urQu4YArEoiq/N83y26EjbwwEzWM3M6SYSWZb9wG6cfIqieJ2zAe8i1nfNh9hB50PExdiRD/tIigIpFEKob5o68IeFCdiuQn3C6l5KbPCElCjKCiXVnI5rt5jS6vXWWdY31gkkql6f1jvyonxAjigCYdti1Hr3o+zSvFecwsofTZvpyz2SBnLbOBEPYIF5uCwgcrafSh0KTKBO+C44u+d/Ji0JddrGtsWvBHjH9de/+Mjf/cMHjW3G0U/QhcC6EUp4BrnCbqyTt/Wkato3vuOm63/jfDw471y7/W1Z3f6sbupJs7GOPAlNmFI4K6jEx0pTSiBEOqFfYvR4bwlNjTh6bOsKZX7gj6+5/ld3cFV+R9u2lGVJSEk8IIa/zaEpqWK09ud34yTWNM2vzPINswS6kh0ScYYMVFLNWdVnDPZsUyF+OIvYWZ9sjzZ0UHRJSp3MUYiBtMti6s77tyqp0FPavNF4RJ7lc+HKUwEdps9AAn6DC0S5/hEi85lmrR7ZcfZMcMpnQz/S6xNnqMuzHR19qmM6C9d3uuO2az1BvF18/u03v75eKl/ke0WOTBglsVtDlqV08ujmf33fTa8576uip7/6tpe4weDHapPhTXbCzdte1CzEQk/u5IfS+07SPa/rjUuHo5vedXDtx3bwND5VCPF3s3sz58DbFscXzMUAOyRiDLv2Jhqlf1cI8WWzcNg22PfcmblpOK8sSuq2QUr5xyGEM4XUP0nA3yup5gl6rTRCCJx381Dc9pCiECL5GORu6iclZARETHF2jvPQV4gBJRUhhvm9jx3q9LcSPHsXea9rpVSvEkLgg5+f8wxdOev/2bM8+/lMPUd8BMfZmJmdy6xvQwzkRfGMpmn+8GFQkh2MMd3ivEMg0FoH551iG9fncaKB4/cvPorzf6ijmPKLaiHx0++dvS6k6hYAMb7gARPE016zdnnQ6mvKMr9kvLkRlvL8r4dbR37//Ws/umuQYM/8wbWVFKsv91o9USR1ZSJuiCQKoZAkGWddmwiC1HELbj/KruoqyWmJMSRSQiCnVXqnO4okZl2sdDblQ9BndBRJpSSC2P47MkKUbD/GIJJUSQSXUhJJiCTmx9l5pG3nMTsKpJx9jhAhxaQShA6J2N31mFJYDz7+tQzh9957zTXDHa6h+vKqqp65vr7eFEUhovciRnyHU4jTraNUQJISIZL6yza0b9+tDqxf9J8+aSZfobW0ILHeMqgqH2JUk6ZJEim01lhvvUSqSIxLS0uXbG1tvewMv+JAkWUvth1jb4xEtNTBR68zrZP1XmRaC621bxorp/1W+hhv3FUODPWcRPo0KTFZVjBpJmLbwIhaSpllBd5b673PsixTIaVfcc59aBeFjJ/lrX2Gj9FlOkvWWzG9B2Ra430XqC+yDO89WmfE6M80PZRO/fqDr0OyrKvT3RqNWFlaYmNrK2mp1crKUjkcj3+ybdt/OtPrG1SDr4kpfqFzbmS0DK31eZZlwXsLSLIsS5PJSGZZJiVE6z1S6kdRZHBmrn1WHpJlGZNmkiRyukmJQmqdSSl/+/8HMGHdoW1h+DYAAAAASUVORK5CYII="


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
            "context": itemgetter("input") | retriever | format_docs,
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


def to_standalone(question, history):
    """이전 대화를 반영해 검색용 독립 질문으로 바꾸기 (예: '그건 언제야?' → '훈련장려금은 언제 지급되나요?')"""
    if not history:
        return question
    prompt = ChatPromptTemplate.from_messages([
        ("system", "이전 대화를 참고해 마지막 질문을 그 자체로 이해되는 하나의 한국어 질문으로 바꾸세요. "
                   "질문만 출력하고, 이미 독립적인 질문이면 그대로 출력하세요."),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])
    chain = prompt | ChatOpenAI(model=MODEL, temperature=0) | StrOutputParser()
    return chain.invoke({"input": question, "history": history}).strip() or question


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
           _p(f"{prof['track']} · {prof['stage']} · {prof['period']} ({prof['weeks']}주) · 작성일 {dt.date.today():%Y-%m-%d}", "sub"),
           _p(f"학습 목표: {prof['goal']}", "box"), _p("계획 요약", "h1"), _p(plan["summary"])]
    for key, title in (("weekday_schedule", "평일 하루 계획"), ("weekend_schedule", "주말 하루 계획")):
        st_ += [_p(title, "h1"), _table(["시간", "할 일", "세부 내용"],
                                        [[b["time"], b["activity"], b["detail"]] for b in plan[key]],
                                        [28 * mm, 45 * mm, CW - 73 * mm])]
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
st.markdown(f"""
<style>
.hero {{background: linear-gradient(120deg,#1F3864 0%,#2E75B6 100%); color:#fff;
       padding: 22px 28px; border-radius: 16px; margin-bottom: 14px;}}
.hero h1 {{color:#fff; font-size: 1.9rem; margin: 0;}}
/* 사이드바 문서 업로드 칸 작게 */
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{padding: 0.35rem 0.6rem; min-height: 0;}}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] svg {{display: none;}}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] small {{display: none;}}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] span {{font-size: 0.8rem;}}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {{padding: 0.1rem 0.6rem; min-height: 0; font-size: 0.8rem;}}
/* 로고 버튼 (누르면 대화 시작) */
.st-key-logo_start button {{
    background: url("data:image/png;base64,{LOGO_B64}") center / contain no-repeat !important;
    border: none !important; box-shadow: none !important;
    width: 100%; height: 110px; color: transparent !important; cursor: pointer;
    transition: transform .15s ease;
}}
.st-key-logo_start button:hover {{transform: scale(1.04);}}
.st-key-logo_start button p {{color: transparent !important;}}
/* 추천 질문 버튼 */
div[class*="st-key-sugg_"] button {{border-radius: 20px; font-size: 0.85rem;}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────── 시작 화면: 로고를 누르면 대화 시작
if not st.session_state.get("started"):
    st.markdown("<div style='height:16vh'></div>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        if st.button("AIVLE", key="logo_start", help="눌러서 대화 시작"):
            st.session_state.started = True
            st.rerun()
        st.markdown("<p style='text-align:center;color:#595959;'>로고를 누르면 대화가 시작돼요</p>",
                    unsafe_allow_html=True)
    st.stop()

st.markdown('<div class="hero"><h1>AIVLE School 학습 도우미</h1></div>', unsafe_allow_html=True)

if not os.environ.get("OPENAI_API_KEY"):
    st.error("OpenAI API 키가 없습니다. Streamlit Cloud의 Settings → Secrets에 OPENAI_API_KEY를 등록하세요.")
    st.stop()

# 기본 RAG 체인 (백서)
default_chain = get_chain()
if get_vectorstore()._collection.count() == 0:
    st.warning("벡터 DB가 비어 있습니다. chroma_db 폴더 또는 백서 PDF 경로(PDF_PATH)를 확인하세요.")

if "history" not in st.session_state:
    st.session_state.history = []   # [{"q":..., "a":..., "sources":[...]}] 최신이 앞

# ─────────────────────────────────────────── 사이드바: 문서 업로드 · 대화 삭제
with st.sidebar:
    st.markdown("**📂 문서 업로드**")
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
        if st.button("📘 기본 백서로 돌아가기"):
            for k in ("upload_chain", "upload_retriever", "upload_id", "upload_name"):
                st.session_state.pop(k, None)
            st.rerun()
    else:
        st.caption("📘 기본 문서: AIVLE School 백서")

    st.divider()
    if st.button("🗑️ 대화 삭제"):
        st.session_state.history = []
        st.rerun()

    if st.session_state.history:
        st.markdown("**🕘 최근 질문**")
        for i, h in enumerate(st.session_state.history[:8]):
            st.caption(f"{i + 1}. {h['q']}")
    st.caption("※ 지원 일정·금액 등은 기수마다 바뀔 수 있으니 공식 공지를 함께 확인하세요.")

chain = st.session_state.get("upload_chain", default_chain)
retriever = st.session_state.get("upload_retriever", get_retriever())

tab_chat, tab_plan, tab_doc = st.tabs(["💬 학습 도우미", "📅 개인 학습 계획", "📝 AI 문서 작성"])

# ─────────────────────────────────────────── 탭 1. 학습 도우미 (최근 대화가 위)
SUGGESTED = [
    "비전공자도 에이블스쿨을 따라갈 수 있을까요?",
    "AI 개발자 트랙과 DX 컨설턴트 트랙은 어떻게 다른가요?",
    "훈련장려금은 얼마나, 언제 받을 수 있나요?",
    "선배들이 강조한 빅프로젝트 수상 비결은?",
    "하루 일과는 어떻게 되나요?",
    "포트폴리오는 어떻게 작성하면 좋을까요?",
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
                st.caption(s)


with tab_chat:
    # ① 질문 입력 (맨 위)
    with st.form("ask_form", clear_on_submit=True):
        c1, c2 = st.columns([8, 1])
        typed = c1.text_input("질문", placeholder="궁금한 점을 입력하세요. 예) 수료 기준이 어떻게 되나요?",
                              label_visibility="collapsed")
        sent = c2.form_submit_button("보내기", type="primary", width="stretch")

    # ② 추천 질문 (항상 표시, 여러 번 눌러도 동작)
    st.markdown("**💡 추천 질문**")
    clicked = None
    cols = st.columns(3)
    for i, q in enumerate(SUGGESTED):
        if cols[i % 3].button(q, key=f"sugg_{i}", width="stretch"):
            clicked = q

    question = clicked or (typed.strip() if sent and typed.strip() else None)

    # ③ 새 질문 → 답변을 맨 위에 바로 표시
    if question:
        with st.chat_message("user", avatar="🙋"):
            st.markdown(question)
        with st.chat_message("assistant", avatar="📘"):
            past = history_messages()
            standalone = to_standalone(question, past)
            answer = st.write_stream(chain.stream({"input": standalone, "history": past}))
            sources = [d.page_content[:300] + ("…" if len(d.page_content) > 300 else "")
                       for d in retriever.invoke(standalone)]
            show_sources(sources)
        st.session_state.history.insert(0, {"q": question, "a": answer, "sources": sources})
        shown = 1
    else:
        shown = 0

    # ④ 이전 대화 (최신이 위)
    if st.session_state.history[shown:]:
        st.divider()
    for h in st.session_state.history[shown:]:
        with st.chat_message("user", avatar="🙋"):
            st.markdown(h["q"])
        with st.chat_message("assistant", avatar="📘"):
            st.markdown(h["a"])
            show_sources(h["sources"])

# ─────────────────────────────────────────── 탭 2. 개인 학습 계획
with tab_plan:
    st.subheader("📅 개인 학습 계획 만들기")
    today = dt.date.today()
    with st.form("plan_form"):
        c1, c2 = st.columns(2)
        track = c1.radio("트랙", ["DX 컨설턴트", "AI 개발자"], horizontal=True)
        stage = c2.selectbox("현재 단계", ["교육 시작 전", "STEP1 (분석형·생성형 AI)", "STEP2 (Cloud·서비스 개발/제안)",
                                         "STEP3 (빅프로젝트)", "수료 전후 (취업 준비)"])
        period = date_range_input("계획 기간 (시작일 ~ 종료일)", today, today + dt.timedelta(days=27), "plan_period")
        goal = st.text_area("학습 목표 *", placeholder="예) AICE Associate 합격, SQL 기본기 다지기")
        c4, c5 = st.columns(2)
        wd_m = c4.number_input("평일 개인 학습 가능 시간 (교육 시간 외, 분/일)", min_value=0, max_value=360,
                               value=120, step=30, format="%d")
        we_m = c5.number_input("주말 학습 가능 시간 (분/일)", min_value=0, max_value=720,
                               value=240, step=30, format="%d")
        level = st.text_area("현재 수준·강점·보완점", placeholder="예) 비전공자, 파이썬 기초는 됨, SQL이 약함")
        ok = st.form_submit_button("📅 학습 계획 생성", type="primary")

    if ok:
        if not goal.strip():
            st.warning("학습 목표를 입력해 주세요.")
        elif not period:
            st.warning("계획 기간의 시작일과 종료일을 모두 선택해 주세요.")
        elif (period[1] - period[0]).days > 7 * 12:
            st.warning("계획 기간은 최대 12주까지 설정할 수 있어요.")
        else:
            start, end = period
            ranges = week_ranges(start, end)
            weeks = len(ranges)
            period_txt = f"{fmt_md(start)}~{fmt_md(end)}"
            week_txt = "\n".join(f"- {i}주차: {fmt_md(a)}~{fmt_md(b)}" for i, (a, b) in enumerate(ranges, 1))
            prof = {"track": track, "stage": stage, "goal": goal, "weeks": weeks,
                    "period": period_txt, "ranges": ranges}
            with st.spinner("백서를 참고해 계획을 만드는 중..."):
                context = search_context(["개인 학습 전략 하루 루틴", f"{stage} 단계별 전략", "일일 타임테이블",
                                          f"{track} 트랙", goal, "배경별 팁"])
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """당신은 AIVLE School 학습 코치입니다. [백서 근거]와 학습자 정보로 실행 가능한 학습 계획을 만드세요.
규칙:
1. 평일 09:00~18:00은 교육 시간입니다(체크인 08:50~09:10, 체크아웃 17:50~18:10). 평일 시간표에 '교육 참여' 블록으로 넣고, 개인 학습은 교육 시간 밖에만 배치하세요.
2. 개인 학습 블록의 합은 평일 하루 {wd}, 주말 하루 {we}을 넘지 마세요. 시간대는 30분 단위로 나누세요.
3. 휴식·식사 시간을 넣어 무리하지 않게 하세요.
4. 주간 계획은 아래 주차 구분대로 정확히 {weeks}개 주차로 만들고, 할 일에는 분량(문제 수·시간·회차)을 적으세요.
5. 백서에 없는 사실·날짜·수치는 만들지 마세요. 한국어로 작성하세요.

[주차 구분]
{week_txt}

[백서 근거]
{context}"""),
                    ("human", "트랙: {track}\n단계: {stage}\n목표: {goal}\n계획 기간: {period} ({weeks}주)\n현재 수준: {level}"),
                ])
                plan = (prompt | ChatOpenAI(model=MODEL, temperature=0.3).with_structured_output(StudyPlan)).invoke({
                    "context": context, "track": track, "stage": stage, "goal": goal, "weeks": weeks,
                    "week_txt": week_txt, "period": period_txt,
                    "wd": fmt_minutes(wd_m), "we": fmt_minutes(we_m), "level": level or "입력 없음",
                }).model_dump()
            st.session_state.plan, st.session_state.plan_prof = plan, prof
            st.session_state.plan_pdf = plan_pdf(plan, prof)

    if "plan" in st.session_state:
        plan, prof = st.session_state.plan, st.session_state.plan_prof
        cols = {"time": "시간", "activity": "할 일", "detail": "세부 내용"}
        st.divider()
        st.markdown(f"### 🎯 {prof['goal']}")
        st.caption(f"📆 {prof['period']} ({prof['weeks']}주)")
        st.info(plan["summary"])
        c1, c2 = st.columns(2)
        c1.markdown("#### 평일 하루 계획")
        c1.dataframe(pd.DataFrame(plan["weekday_schedule"]).rename(columns=cols), hide_index=True, width="stretch")
        c2.markdown("#### 주말 하루 계획")
        c2.dataframe(pd.DataFrame(plan["weekend_schedule"]).rename(columns=cols), hide_index=True, width="stretch")
        st.markdown("#### 주간 계획")
        for w in plan["weekly_plan"]:
            with st.expander(f"{w['week']}주차 ({week_label(prof, w['week'])}) · {w['theme']}", expanded=(w["week"] == 1)):
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

    today = dt.date.today()
    with st.form("project_form", clear_on_submit=True):
        c1, c2 = st.columns([2, 1])
        name = c1.text_input("프로젝트 이름 *")
        ptype = c2.selectbox("유형", ["빅프로젝트", "미니프로젝트", "공모전", "해커톤", "스터디", "개인 프로젝트", "기타"])
        proj_period = date_range_input("기간 (시작일 ~ 종료일)", today - dt.timedelta(days=55), today, "proj_period")
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
        elif not proj_period:
            st.warning("기간의 시작일과 종료일을 모두 선택해 주세요.")
        else:
            a, b = proj_period
            weeks_n = max(1, math.ceil(((b - a).days + 1) / 7))
            period = f"{a:%Y.%m.%d}~{b:%Y.%m.%d} ({weeks_n}주)"
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
5. projects는 입력 순서대로 입력 개수만큼 만들고, period는 입력한 기간을 그대로 쓰세요.

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

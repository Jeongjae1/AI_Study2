import os
import sys
import json
import time
import datetime
import requests
from google import genai
from google.genai import errors

HISTORY_FILE = "history.json"

# 사용할 모델 순서 (앞에서부터 시도, 실패하면 다음 모델로)
# - gemini-3.8-flash      : 현재 최신 Flash (메인)
# - gemini-3.6-flash      : 1차 대체
# - gemini-3.5-flash-lite : 2차 대체 (가볍고 혼잡에 강함)
MODELS = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite"]

# 재시도할 오류 코드: 429(호출 한도), 500(서버 오류), 503(서버 혼잡), 504(시간 초과)
RETRY_CODES = (429, 500, 503, 504)
# 다음 모델로 바로 넘어갈 오류 코드: 404(모델 없음/종료됨)
SKIP_CODES = (404,)
MAX_TRIES_PER_MODEL = 4  # 모델당 시도 횟수 (대기: 10 → 20 → 40초)

# 전체 커리큘럼 로드맵 (입문 → 실무)
CURRICULUM = """
[1단계 입문] AI/머신러닝 기본 개념, 학습 유형(지도/비지도/강화), 데이터와 특성, 과적합
[2단계 기초 ML] 선형/로지스틱 회귀, 결정트리, 앙상블, 평가지표, 교차검증
[3단계 딥러닝 기초] 퍼셉트론, 신경망, 역전파, 활성화함수, 손실함수, 옵티마이저
[4단계 딥러닝 구조] CNN(이미지), RNN/LSTM(시퀀스), 임베딩, 정규화/드롭아웃
[5단계 트랜스포머/LLM] 어텐션, 트랜스포머 구조, 사전학습/파인튜닝, 토큰화
[6단계 LLM 실무] 프롬프트 엔지니어링, RAG, 임베딩 검색, 에이전트, 파인튜닝
[7단계 MLOps/배포] 모델 서빙, API화, 모니터링, 파이프라인, 실서비스 적용
"""


def load_history():
    """학습 이력 로드. 없으면 초기 상태 반환."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"day": 0, "stage": "1단계 입문", "topics": []}


def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def send_telegram(text):
    """텔레그램 전송. 실패하면 예외 발생."""
    resp = requests.post(
        f"https://api.telegram.org/bot{os.environ['TELEGRAM_BOT_TOKEN']}/sendMessage",
        json={"chat_id": os.environ["TELEGRAM_CHAT_ID"], "text": text},
        timeout=30,
    )
    resp.raise_for_status()


def generate_with_retry(client, prompt):
    """
    모델 여러 개를 순서대로 시도하며, 일시적 오류는 기다렸다가 재시도.
    (식당이 만석이면 잠깐 기다렸다 다시 가보고, 계속 만석이면 옆 식당으로)
    """
    last_err = None
    for model in MODELS:
        for attempt in range(1, MAX_TRIES_PER_MODEL + 1):
            try:
                print(f"[{model}] 시도 {attempt}/{MAX_TRIES_PER_MODEL}")
                response = client.models.generate_content(model=model, contents=prompt)
                text = (response.text or "").strip()
                if not text:
                    raise ValueError("빈 응답")
                print(f"[{model}] 성공")
                return text
            except errors.APIError as e:
                last_err = e
                if e.code in SKIP_CODES:
                    print(f"[{model}] {e.code} 사용 불가 모델 → 다음 모델로")
                    break
                if e.code not in RETRY_CODES:
                    raise  # 키 오류(401/403) 등은 재시도해도 소용없으므로 바로 중단
                print(f"[{model}] {e.code} 오류: {e.message}")
            except ValueError as e:
                last_err = e
                print(f"[{model}] {e}")

            if attempt < MAX_TRIES_PER_MODEL:
                wait = 10 * 2 ** (attempt - 1)  # 10 → 20 → 40초
                print(f"  {wait}초 후 재시도")
                time.sleep(wait)
        print(f"[{model}] 계속 실패 → 다음 모델로 전환")
    raise RuntimeError(f"모든 모델 실패. 마지막 오류: {last_err}")


history = load_history()
today = datetime.date.today().isoformat()
day_no = history["day"] + 1

# 지금까지 다룬 주제 목록 (프롬프트에 주입)
past_topics = history["topics"]
past_topics_text = "\n".join(
    f"- Day {t['day']}: {t['topic']} ({t['stage']})" for t in past_topics
) or "(아직 없음 - 오늘이 첫날)"

# 직전 예고 (있으면 오늘 이어받기)
last_preview = past_topics[-1]["next_preview"] if past_topics else ""

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

prompt = f"""
너는 세계 최고의 AI 교육 전문가다.

목표는 학습자가 AI를 입문부터 실무 수준까지 '체계적인 커리큘럼'으로 매일 하나씩 공부하도록,
어제까지의 학습에 이어 오늘의 콘텐츠를 만드는 것이다.

[전체 커리큘럼 로드맵]
{CURRICULUM}

[지금까지 학습한 주제 - 절대 중복 금지]
{past_topics_text}

[현재 진행 위치]
- 오늘은 Day {day_no}
- 현재 단계: {history["stage"]}
- 어제 예고한 내용: {last_preview or "(없음)"}

[작성 규칙]
1. 위 '지금까지 학습한 주제'와 절대 겹치지 않는 새 주제를 골라라.
2. 커리큘럼 로드맵의 순서를 따라 자연스럽게 다음 단계로 심화시켜라.
   앞 단계 주제를 충분히 다뤘으면 다음 단계로 넘어가라.
3. 어제 예고한 내용이 있으면 그것을 오늘의 주제로 이어받아라.
4. 오늘 주제가 어제 내용과 어떻게 연결되는지 '📌 어제와의 연결' 항목에서 한 줄로 짚어라.

반드시 아래 형식을 지켜라.

🤖 AI Morning Brief (Day {day_no} / {history["stage"]})

📌 어제와의 연결
(어제 배운 것과 오늘 주제가 어떻게 이어지는지 한 줄. 첫날이면 "커리큘럼 시작"이라고 써라)

📚 오늘의 학습 주제
(한 가지 주제)

🎯 오늘의 목표
(오늘 무엇을 배우는지)

💡 핵심 개념
(3~5개 핵심 포인트)

📰 최신 AI 동향
(최근 AI 업계에서 알아두면 좋은 소식 1개)

🎥 추천 영상
(유튜브에서 검색할 수 있는 제목)

📖 추천 문서
(공식 문서나 블로그 이름)

💻 15분 실습
(직접 따라 해볼 과제)

❓ 오늘의 퀴즈
(객관식 또는 단답형 1문제)

🚀 내일 예고
(내일 이어질 주제를 한 줄로. 커리큘럼상 다음에 올 내용)

답변 맨 마지막 줄에 아래 형식으로 메타데이터를 정확히 출력해라. (사용자에게 보이지만 기록용이다)
[META] 주제=<오늘 주제> | 단계=<현재 단계명> | 다음예고=<내일 예고 한 줄>

답변은 800자 이내의 한국어로 작성한다.
"""

# 1) 콘텐츠 생성 (재시도 + 대체 모델)
try:
    message = generate_with_retry(client, prompt)
except Exception as e:
    print(f"콘텐츠 생성 실패: {e}")
    try:
        send_telegram(f"⚠️ AI Morning Brief 생성 실패 (Day {day_no})\nGemini 서버 혼잡으로 오늘 발송을 못 했어요.\n{e}")
    except Exception:
        pass
    sys.exit(1)

# 2) [META] 라인 파싱
topic = "미확인"
stage = history["stage"]
next_preview = ""
for line in message.splitlines():
    if line.strip().startswith("[META]"):
        meta = line.strip()[len("[META]"):].strip()
        for part in meta.split("|"):
            if "=" in part:
                key, _, val = part.partition("=")
                key, val = key.strip(), val.strip()
                if key == "주제":
                    topic = val
                elif key == "단계":
                    stage = val
                elif key == "다음예고":
                    next_preview = val
        break

# 사용자에게 보낼 메시지에서는 [META] 라인 제거
clean_message = "\n".join(
    l for l in message.splitlines() if not l.strip().startswith("[META]")
).strip()

# 3) 텔레그램 전송 (성공했을 때만 이력 저장 → 못 받은 주제가 '학습함'으로 기록되지 않게)
send_telegram(clean_message)

# 4) 이력 갱신
history["day"] = day_no
history["stage"] = stage
history["topics"].append({
    "day": day_no,
    "date": today,
    "topic": topic,
    "stage": stage,
    "next_preview": next_preview,
})
save_history(history)
print(f"완료: Day {day_no} - {topic}")

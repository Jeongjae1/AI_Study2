import os
import json
import datetime
import requests
from google import genai

HISTORY_FILE = "history.json"

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

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=prompt,
)

message = response.text or ""

# [META] 라인 파싱해서 이력 갱신
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

# 이력 갱신
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

# 텔레그램 전송
requests.post(
    f"https://api.telegram.org/bot{os.environ['TELEGRAM_BOT_TOKEN']}/sendMessage",
    json={
        "chat_id": os.environ["TELEGRAM_CHAT_ID"],
        "text": clean_message,
    },
)

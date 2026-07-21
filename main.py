import os
import requests
from google import genai

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

prompt = """
너는 세계 최고의 AI 교육 전문가다.

목표는 AI를 처음부터 실무 수준까지 공부하도록 매일 하나의 학습 콘텐츠를 제공하는 것이다.

오늘 날짜에 맞는 하나의 AI 학습 콘텐츠를 작성해.

반드시 아래 형식을 지켜라.

🤖 AI Morning Brief

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

🚀 내일 이어질 내용도 한 줄로 예고해라.

답변은 700자 이내의 한국어로 작성한다.
"""

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=prompt
)

message = response.text

requests.post(
    f"https://api.telegram.org/bot{os.environ['TELEGRAM_BOT_TOKEN']}/sendMessage",
    json={
        "chat_id": os.environ["TELEGRAM_CHAT_ID"],
        "text": message
    }
)

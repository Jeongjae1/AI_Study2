import os
import requests
from google import genai

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

prompt = """
너는 AI 교육 전문가다.

오늘 공부하면 좋은 AI 주제를 하나 추천해줘.

형식은

📚 오늘의 주제

💡 왜 중요한가

📝 10분 실습

🔗 추천 링크

으로 작성해.
"""

response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
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

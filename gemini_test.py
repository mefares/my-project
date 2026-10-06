import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise SystemExit("Error: GEMINI_API_KEY is missing. Add it to your .env file.")

response = requests.post(
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent",
    headers={"x-goog-api-key": api_key},
    json={"contents": [{"parts": [{"text": "Say hello in one sentence"}]}]},
)

if response.status_code != 200:
    raise SystemExit(f"Request failed with status {response.status_code}: {response.text}")

print(response.json()["candidates"][0]["content"]["parts"][0]["text"])

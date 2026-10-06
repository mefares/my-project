import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise SystemExit("Error: MISTRAL_API_KEY is missing. Add it to your .env file.")

response = requests.post(
    "https://api.mistral.ai/v1/chat/completions",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "model": "ministral-8b-latest",
        "messages": [{"role": "user", "content": "Say hello in one sentence"}],
    },
)

if response.status_code != 200:
    raise SystemExit(f"Request failed with status {response.status_code}: {response.text}")
    

print(response.json()["choices"][0]["message"]["content"])

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("API Key not found.")
else:
    try:
        client = genai.Client(api_key=api_key)
        print("Available models:")
        for m in client.models.list():
            if 'generateContent' in m.supported_actions:
                print(f" - {m.name}")
    except Exception as e:
        print(f"Error: {e}")

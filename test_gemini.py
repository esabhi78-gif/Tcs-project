import os
from dotenv import load_dotenv
import sys

print("--- DIAGNOSTIC SCRIPT ---")

# 1. Check dot env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    masked_key = api_key[:4] + "*" * 20 + api_key[-4:] if len(api_key) > 8 else "too_short"
    print(f"API Key found: {masked_key}")
else:
    print("API Key NOT FOUND in environment or .env file")

# 2. Check packages
print("\n--- Package check ---")
try:
    import google.generativeai as old_genai
    print("Old SDK (google-generativeai) is INSTALLED")
except ImportError:
    print("Old SDK (google-generativeai) is NOT installed")

try:
    from google import genai
    print("New SDK (google-genai) is INSTALLED")
    new_sdk = True
except ImportError:
    print("New SDK (google-genai) is NOT installed. You need to run 'pip install google-genai'")
    new_sdk = False

# 3. Test API Call if possible
if api_key and new_sdk:
    print("\n--- Testing API Call ---")
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents='Respond with a single word: Hello'
        )
        print(f"API CALL SUCCESSFUL! Response: {response.text}")
    except Exception as e:
        print(f"API CALL FAILED! Error: {e}")

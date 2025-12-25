# /Users/vj/Desktop/AI-ML/1st.py
from dotenv import load_dotenv
load_dotenv()

import os
from google import genai
from google.genai.errors import ClientError

api_key = os.getenv("GENAI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise SystemExit("GENAI_API_KEY or GOOGLE_API_KEY environment variable is required.")

client = genai.Client(api_key=api_key)

try:
    response = client.models.generate_content(
        model='gemini-1.5-flash',   
        contents='Explain how AI works in one sentence.'
    )
    print(response.text)
except ClientError as e:
    print("API error:", e)   
    raise


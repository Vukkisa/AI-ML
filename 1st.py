import os
from google import genai
from google.genai.errors import ClientError

# Read API key from environment for safety (do not hard-code keys)
api_key = os.getenv("GENAI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise SystemExit(
        "GENAI_API_KEY or GOOGLE_API_KEY environment variable is required.\n"
        "Set it with: export GENAI_API_KEY='YOUR_API_KEY'"
    )

# Initialize the client with your API key
client = genai.Client(api_key=api_key)

# Send a prompt to the model
try:
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents='Explain how AI works in one sentence.'
    )
    print(response.text)
except ClientError as e:
    print("API error:", e)
    raise


from transformers import pipeline
gen = pipeline("text-generation", model="gpt2")   # small demo model
gen_output = gen("Write a one-sentence summary of reinforcement learning:", max_length=60)
import os
from dotenv import load_dotenv
from google import genai
import json

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("GOOGLE_API_KEY not found!")

# Initialize the GenAI client
client = genai.Client(api_key=api_key)

def swing_feedback(prompt):
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        raw = response.text.strip()

        # Remove Markdown formatting (```json ... ```)
        if raw.startswith("```json"):
            raw = raw[7:]  # Remove the opening ```
        if raw.endswith("```"):
            raw = raw[:-3]  # Remove the closing ```

        return json.loads(raw.strip())
    
    except Exception as e:
        print("Gemini API error:", e)
        return "Error: Could not retrieve feedback."
import os
import json
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("GOOGLE_API_KEY not found!")

# Initialize the GenAI client
model = genai.GenerativeModel("gemini-1.5-flash")

def swing_feedback(prompt):
    try:
        response = model.generate_content(prompt)
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
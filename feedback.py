import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv("API_KEY")
api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-pro")

def swing_feedback(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print("Gemini API error:", e)
        return "Error: Could not retrieve feedback."
import os, google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "models/gemini-1.5-flash-latest"   # hoặc gemini-1.5-pro-latest
model = genai.GenerativeModel(MODEL_NAME)

def gemini_generate(prompt: str) -> str:
    response = model.generate_content(prompt)
    return response.text.strip()

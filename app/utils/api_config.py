import os
import dotenv
import google.generativeai as genai
import logging

dotenv.load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    logging.error("GOOGLE_API_KEY não encontrado no arquivo .env")
    raise ValueError("GOOGLE_API_KEY não encontrado no arquivo .env")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-pro")

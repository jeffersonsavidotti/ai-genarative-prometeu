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
model_config = genai.GenerationConfig(
            temperature=0.1,
            top_k=1000,
            max_output_tokens=1000,
            top_p=1
        )

model = genai.GenerativeModel(
            model_name="gemini-pro",
            generation_config=model_config
        )



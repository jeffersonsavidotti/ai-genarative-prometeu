import os
import dotenv
import google.generativeai as genai
import streamlit as st
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Função para extrair texto do PDF
def extract_text_from_pdf(file):
    # Read the file content into a bytes object
    pdf_data = file.read()

    # Open the PDF using the bytes data
    doc = fitz.open(stream=pdf_data, filetype="pdf")

    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Função para dividir o texto em segmentos
def split_text(text, max_len=500):
    words = text.split()
    segments = []
    current_segment = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > max_len:
            segments.append(' '.join(current_segment))
            current_segment = []
            current_length = 0
        current_segment.append(word)
        current_length += len(word) + 1

    if current_segment:
        segments.append(' '.join(current_segment))

    return segments

# Função para criar embeddings dos segmentos de texto
def create_embeddings(all_text_segments, client):
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(all_text_segments)
    return vectorizer, embeddings

# Função para buscar o segmento mais relevante com base nos embeddings
def search_with_embeddings(prompt, vectorizer, embeddings, all_text_segments):
    query_vec = vectorizer.transform([prompt])
    similarities = cosine_similarity(query_vec, embeddings).flatten()
    best_match_index = np.argmax(similarities)
    best_match_segment = all_text_segments[best_match_index]
    return best_match_index, best_match_segment

# Carregar variáveis de ambiente do arquivo .env
dotenv.load_dotenv()

# Configure a API do Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-pro")

pdf_paths = ["./Data/eventosPI.pdf"]
all_text_segments = []

for pdf_path in pdf_paths:
    with open(pdf_path, "rb") as file:
        text = extract_text_from_pdf(file)
        text_segments = split_text(text)
        all_text_segments.extend(text_segments)

vectorizer, embeddings = create_embeddings(all_text_segments, client=None)  # Ajustado para não usar `client`

st.title("🔥 IA Generativa Prometeu 💬")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "E aí, camarada! Zé Devinho na área, o seu assistente pessoal da Programmers Beyond IT, aqui pra te dar um suporte de primeira. 😎\n\nFala aí, qual trampo que você tá precisando dar um gás? Se liga que o Zé tá pronto pra te ajudar, mano! 🤘\n\n"}
    ]

avatar = 'https://tmssl.akamaized.net/images/foto/galerie/neymar-brazil-2022-1668947300-97010.jpg?lm=1668947335'
for msg in st.session_state.messages:
    st.chat_message(msg["role"], avatar=avatar if msg["role"] == "assistant" else None).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar='🦹‍♂️').write(prompt)

    if prompt:
        try:
            best_match_index, best_match_segment = search_with_embeddings(prompt, vectorizer, embeddings, all_text_segments)

            chat_session = model.start_chat(
            history=[
                    {
                        "role": "model",
                        "parts": [
                            "Você é um assistente pessoal muito ligeiro, fala em girias e seu nome é Zé Devinho. Sua função é servir a todos os colaboradores da Programmers Beyond IT.",
                            "Não responda perguntas que fujam do tema: Programmers Beyond IT. E não invente respostas",
                            "Você não deve desencorajar o usuário. Sempre forneça informações úteis e positivas",
                            "No final de cada frase você diz balinha",
                        ],
                    },
                    {
                        "role": "user",
                        "parts": [
                            f"Use o seguinte contexto para responder: {best_match_segment}",
                            prompt
                        ],
                    }
                ]
            )

            response = chat_session.send_message(prompt)
            answer = response.text

            st.session_state["messages"].append({"role": "assistant", "content": answer})
            st.chat_message("assistant", avatar="🕵️‍♂️").write(answer)
        except Exception as e:
            st.error(f"Gemini API error: {e}")
    else:
        st.write("Please enter a question.")

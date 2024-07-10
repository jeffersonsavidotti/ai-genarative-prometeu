import os
import dotenv
import google.generativeai as genai
import streamlit as st
import fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ==== Funções de Processamento de PDF ====
def extract_text_from_pdf(file):
    """Extrai texto de um arquivo PDF."""
    pdf_data = file.read()
    doc = fitz.open(stream=pdf_data, filetype="pdf")
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def split_text(text, max_len=500):
    """Divide o texto em segmentos menores."""
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

def create_embeddings(all_text_segments, client):
    """Cria embeddings dos segmentos de texto usando TF-IDF."""
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(all_text_segments)
    return vectorizer, embeddings

def search_with_embeddings(prompt, vectorizer, embeddings, all_text_segments):
    """Busca o segmento mais relevante com base nos embeddings."""
    query_vec = vectorizer.transform([prompt])
    similarities = cosine_similarity(query_vec, embeddings).flatten()
    best_match_index = np.argmax(similarities)
    best_match_segment = all_text_segments[best_match_index]
    return best_match_index, best_match_segment

def load_existing_pdfs():
    pdf_paths = [os.path.join("./Data", filename) 
                 for filename in os.listdir("./Data") 
                 if filename.lower().endswith(".pdf")] 
    all_text_segments = []
    for pdf_path in pdf_paths:
        with open(pdf_path, "rb") as file:
            text = extract_text_from_pdf(file)
            text_segments = split_text(text)
            all_text_segments.extend(text_segments)
    return all_text_segments


# ==== Configuração da API do Gemini Pro ====
dotenv.load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-pro")


# ==== Carregamento de PDFs Existentes (agora em uma função) ====
all_text_segments = load_existing_pdfs()
vectorizer, embeddings = create_embeddings(all_text_segments, client=None)

# ==== Barra Lateral Personalizada ====
with st.sidebar:
    st.image("https://tmssl.akamaized.net/images/foto/galerie/neymar-brazil-2022-1668947300-97010.jpg?lm=1668947335", width=150)
    st.title("🔥 Zé Devinho 💬")

    # ==== Botão de Upload de Arquivo ====
    uploaded_file = st.file_uploader("🚀 Upload do PDF:", type=["pdf"])
    if uploaded_file is not None:
        original_filename = uploaded_file.name
        name, ext = os.path.splitext(original_filename)
        ext = ext.lower()
        save_path = os.path.join("./Data", name + ext)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        st.success(f"Arquivo PDF salvo com sucesso em: {save_path}")

        all_text_segments = load_existing_pdfs()
        vectorizer, embeddings = create_embeddings(all_text_segments, client=None)
        st.success("Embeddings criados com sucesso!")


# ==== Chat e Processamento Principal ====
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
    st.chat_message("user").write(prompt)  # Removido o avatar do usuário

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
            st.chat_message("assistant", avatar=avatar).write(answer)
        except Exception as e:
            st.error(f"Gemini API error: {e}")

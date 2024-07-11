import os
import json
import dotenv
import google.generativeai as genai
import streamlit as st
import fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime
import re

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


# ==== Carregamento de PDFs Existentes ====
all_text_segments = load_existing_pdfs()
vectorizer, embeddings = create_embeddings(all_text_segments, client=None)

st.set_page_config(
    page_title="IA Generativa PI",
    page_icon="https://i.imgur.com/LlUM5am.jpg?lm=1668947335"
)

# ==== Funções para Salvar e Carregar Conversas ====
def save_conversations_to_json():
    with open("conversations.json", "w") as f:
        json.dump(st.session_state.conversations, f)

def load_conversations_from_json():
    if os.path.exists("conversations.json"):
        with open("conversations.json", "r") as f:
            return json.load(f)
    return []

# ==== Barra Lateral Personalizada ====
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center;">
            <img src="https://i.imgur.com/10PJpQH.png?lm=1668947335" width="150">
            <h1>Alpha-C, ao seu dispor! 💬</h1>
        </div>
        """, unsafe_allow_html=True)

    # ==== Botão de Upload de Arquivo ====
    uploaded_file = st.file_uploader("Upload 📎:", type=["pdf"])
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
st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://i.imgur.com/LlUM5am.jpg?lm=1668947335" width="150">
        <h1>IA Generativa PI</h1>
    </div>
    """, unsafe_allow_html=True)

# Inicializar o estado da sessão
if "conversations" not in st.session_state:
    st.session_state.conversations = load_conversations_from_json()

if "current_conversation" not in st.session_state:
    st.session_state.current_conversation = {
        "title": f"Conversa iniciada em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "messages": [{"role": "assistant", "content": "Como posso te ajudar a criar hoje?"}]
    }

if "delete_confirm" not in st.session_state:
    st.session_state.delete_confirm = None

# Funções para gerenciar conversas
def start_new_conversation():
    st.session_state.conversations.append(st.session_state.current_conversation)
    st.session_state.current_conversation = {
        "title": f"Conversa iniciada em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "messages": [{"role": "assistant", "content": "Como posso te ajudar a criar hoje?"}]
    }
    save_conversations_to_json()

def edit_conversation_title(index, new_title):
    st.session_state.conversations[index]["title"] = new_title
    save_conversations_to_json()

def delete_conversation(index):
    if st.session_state.delete_confirm == index:
        st.session_state.conversations.pop(index)
        st.session_state.delete_confirm = None
        save_conversations_to_json()
        if st.session_state.conversations:
            st.session_state.current_conversation = st.session_state.conversations[0]
        else:
            st.session_state.current_conversation = {
                "title": f"Conversa iniciada em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "messages": [{"role": "assistant", "content": "Como posso te ajudar a criar hoje?"}]
            }
    else:
        st.session_state.delete_confirm = index

# Botões na barra lateral
st.sidebar.button("Nova Conversa", on_click=start_new_conversation)

# Histórico de Conversas
st.sidebar.title("Histórico de Conversas")
for i, conversation in enumerate(st.session_state.conversations):
    if st.sidebar.button(conversation["title"], key=f"conversation_{i}"):
        st.session_state.current_conversation = conversation

    with st.sidebar.expander(conversation["title"], expanded=False):
        new_title = st.text_input("Editar título", value=conversation["title"], key=f"title_{i}")
        if st.button("Salvar título", key=f"save_title_{i}"):
            edit_conversation_title(i, new_title)
        if st.button("Apagar", key=f"delete_{i}"):
            delete_conversation(i)
        if st.session_state.delete_confirm == i:
            st.write("Tem certeza que deseja apagar esta conversa?")
            st.button("Sim, apagar", key=f"confirm_delete_{i}", on_click=delete_conversation, args=(i,))
            st.button("Não, cancelar", key=f"cancel_delete_{i}", on_click=lambda: st.session_state.update({"delete_confirm": None}))

# Exibir mensagens na interface do chat
avatar = 'https://i.imgur.com/10PJpQH.png?lm=1668947335'
for msg in st.session_state.current_conversation["messages"]:
    st.chat_message(msg["role"], avatar=avatar if msg["role"] == "assistant" else None).write(msg["content"])

# Entrada do usuário
if prompt := st.chat_input("Qual é a sua dúvida hoje?"):
    st.session_state.current_conversation["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    try:
        best_match_index, best_match_segment = search_with_embeddings(prompt, vectorizer, embeddings, all_text_segments)

        chat_session = model.start_chat(
            history=[
                {
                    "role": "model",
                    "parts": [
                        "Você é um assistente pessoal muito ligeiro, fala em girias e seu nome é Alpha-C. Sua função é servir a todos os colaboradores da Programmers Beyond IT.",
                        "Não responda perguntas que fujam do tema: Programmers Beyond IT. E não invente respostas",
                        "Você não deve desencorajar o usuário. Sempre forneça informações úteis e positivas",
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

        st.session_state.current_conversation["messages"].append({"role": "assistant", "content": answer})
        st.chat_message("assistant", avatar=avatar).write(answer)
    except Exception as e:
        st.session_state.current_conversation["messages"].append({"role": "assistant", "content": "Desculpe, algo deu errado ao processar a sua solicitação."})
        st.chat_message("assistant", avatar=avatar).write("Desculpe, algo deu errado ao processar a sua solicitação.")
        st.error(f"Erro: {str(e)}")

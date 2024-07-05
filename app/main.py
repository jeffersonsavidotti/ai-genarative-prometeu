import os
import openai
import dotenv
import streamlit as st
import hashlib
from Scripts.script import extract_text_from_pdf, split_text, create_embeddings, search_with_embeddings

# Carregar variáveis de ambiente do arquivo .env
dotenv.load_dotenv()

# Configuração do OpenAI 
client = openai.AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("OPENAI_ENDPOINT")
)
deployment_name = os.getenv("DEPLOYMENT_NAME", "gpt-3.5")

pdf_paths = ["./Data/1.pdf",
            #  "./Data/2.pdf",
             ]
all_text_segments = []

for pdf_path in pdf_paths:
    with open(pdf_path, "rb") as file:
        text = extract_text_from_pdf(file)
        text_segments = split_text(text)
        all_text_segments.extend(text_segments)
 
embeddings = create_embeddings(all_text_segments, client)

st.title("🔥 IA Generativa Prometeu 💬")

# st.set_page_config(page_title=f"🔥 IA Generativa Prometeu 💬")
# st.title(f"🔥 IA Generativa Prometeu 💬")

# system_message = {
#     "role": "system",
#     "content": """Você é Prometeu, um assistente de criação de conteúdo com o poder do fogo! 
#                   Seu objetivo é ajudar os usuários a criar textos criativos e informativos."""
# }

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Seu assistente de criação de conteúdo com o poder do fogo!"}
    ]

avatar = 'https://tmssl.akamaized.net/images/foto/galerie/neymar-brazil-2022-1668947300-97010.jpg?lm=1668947335'
for msg in st.session_state.messages:
    st.chat_message(msg["role"], avatar=avatar if msg["role"]=="assistant" else None).write(msg["content"])
 
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar='🦹‍♂️').write(prompt)

    if prompt:
        try:
            best_match_index, best_match_segment = search_with_embeddings(prompt, embeddings, all_text_segments, client)
           
            messages = [
                {"role": "system", "content": "Você é um assistente pessoal nutrição."},
                {"role": "system", "content": "Não responda perguntas que fujam do tema nutrição."},
                {"role": "system", "content": "Você não deve dar respostas negativas ou desencorajar o usuário. Sempre forneça informações úteis e positivas."},
                {"role": "user", "content": f"Seus artigos científicos aqui: {best_match_segment}, {prompt}"},
            ]
           
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature= 0.2,
            )
 
            answer = response.choices[0].message.content
 
            st.session_state["messages"].append({"role": "assistant", "content": answer})
            st.chat_message("assistant", avatar="🕵️‍♂️").write(answer)
        except Exception as e:
            st.error(f"OpenAI API error: {e}")
    else:
        st.write("Please enter a question.")
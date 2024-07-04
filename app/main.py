import os
import openai
import dotenv
import streamlit as st
import hashlib  # For generating unique keys

# Carregar variáveis de ambiente do arquivo .env
dotenv.load_dotenv()

# Configuração do OpenAI 
client = openai.AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("OPENAI_ENDPOINT")
)
deployment_name = os.getenv("DEPLOYMENT_NAME", "gpt-4")

# Configuração da IA (no prompt da API)
system_message = {
    "role": "system",
    "content": """Você é Prometeu, um assistente de criação de conteúdo com o poder do fogo! 
                  Seu objetivo é ajudar os usuários a criar textos criativos e informativos."""
}

# Inicializar o estado da sessão (com valores padrão)
if "current_conversation" not in st.session_state:
    st.session_state.current_conversation = None
if "conversations" not in st.session_state:
    st.session_state.conversations = {}

# Configurar o título da página
st.set_page_config(page_title=f"🔥 IA Generativa Prometeu 💬")

# Função para gerar um título para a conversa (aprimorada)
def generate_conversation_title(messages):
    if messages:
        prompt = f"Crie um título curto e descritivo para esta conversa:\n\n{messages[-1]['content']}"
        response = client.chat.completions.create(
            engine=deployment_name, messages=[system_message, {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    else:
        return "Nova Conversa"

# Inicialização do Streamlit
st.title(f"🔥 IA Generativa Prometeu 💬")
st.write("Seu assistente de criação de conteúdo com o poder do fogo!")

# Botão para iniciar nova conversa na barra lateral (corrigido)
if st.sidebar.button("Nova Conversa", key="new_conversation_button"):
    st.session_state.current_conversation = generate_conversation_title([])

# Histórico de Conversas (com unique keys e lógica de exibição aprimorada)
st.sidebar.title("Histórico de Conversas")
for title in st.session_state.conversations:
    if title != st.session_state.current_conversation:
        key = hashlib.md5(title.encode()).hexdigest()
        if st.sidebar.button(title, key=key):
            st.session_state.current_conversation = title

# Área principal de chat
if st.session_state.current_conversation is None:
    st.session_state.current_conversation = generate_conversation_title([])
if st.session_state.current_conversation not in st.session_state.conversations:
    st.session_state.conversations[st.session_state.current_conversation] = []

st.header(st.session_state.current_conversation)
for message in st.session_state.conversations.get(st.session_state.current_conversation, []):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Digite sua mensagem"):
    st.session_state.conversations[st.session_state.current_conversation].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Chamada à API do OpenAI (atualizado)
    messages = [system_message] + st.session_state.conversations[st.session_state.current_conversation]
    response = client.chat.completions.create(
        model=deployment_name, messages=messages
    )

    # Processar a resposta da API
    msg = response.choices[0].message.content
    st.session_state.conversations[st.session_state.current_conversation].append({"role": "assistant", "content": msg})
    with st.chat_message("assistant"):
        st.markdown(msg)

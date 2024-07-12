import os
import utils.api_config as api_config
import streamlit as st
import logging
from utils.pdf_processing import extract_text_from_pdf, split_text, load_existing_pdfs
from utils.embedding import create_embeddings, search_with_embeddings, try_fallback_search
from utils.conversation import (
    save_conversations_to_json, 
    load_conversations_from_json, 
    start_new_conversation, 
    edit_conversation_title, 
    delete_conversation, 
    save_and_load_conversation,
    save_current_conversation,
)

avatar = 'https://i.imgur.com/10PJpQH.png'

# Carregamento de PDFs Existentes
all_text_segments = load_existing_pdfs()
vectorizer, embeddings = create_embeddings(all_text_segments, client=None)

# Configurações da página
st.set_page_config(
    page_title="IA Generativa PI",
    page_icon="https://i.imgur.com/LlUM5am.jpg"
)

# Inicialização de st.session_state.conversations, se ainda não estiver inicializado
if "conversations" not in st.session_state:
    st.session_state.conversations = load_conversations_from_json()

if "current_conversation" not in st.session_state:
    start_new_conversation()

if "delete_confirm" not in st.session_state:
    st.session_state.delete_confirm = None

# Barra lateral personalizada
sidebar_expanded = True  # Define a barra lateral como expandida por padrão

with st.sidebar:
    st.markdown(
        """
        <style>
        .sidebar-content {
            text-align: center;
        }
        </style>
        <div class="sidebar-content">
            <img src='https://i.imgur.com/10PJpQH.png' width="150">
            <h1>Alpha-C, ao seu dispor! 💬</h1>
        </div>
        """, unsafe_allow_html=True)
    with st.container():
        col1, col2, col3 = st.columns([0.2, 0.6, 0.2])  # 20%, 60%, 20%
        with col2:
            st.button("Nova Conversa", on_click=start_new_conversation)

    # Abas no Sidebar
    tab1, tab2 = st.sidebar.tabs(["Histórico de Conversas", "Upload"])

    with tab1:

        # Verifica se st.session_state.conversations existe antes de iterar sobre ele
        if "conversations" in st.session_state:
            for i, conversation in enumerate(st.session_state.conversations):
                col1, col2 = st.columns([8, 1])

                with col1:
                    if st.button(conversation["title"], key=f"conversation_{i}"):
                        save_and_load_conversation(conversation)

                        # Exibir opções da conversa selecionada
                        with st.expander(f"Opções da Conversa '{conversation['title']}'"):
                            new_title = st.text_input("Editar título", value=conversation["title"], key=f"title_{i}")

                            if st.button("Salvar título", key=f"save_title_{i}"):
                                edit_conversation_title(i, new_title)

                            if st.button("Apagar", key=f"delete_{i}"):
                                st.session_state.delete_confirm = i

                            # Popup de confirmação para apagar
                            if st.session_state.delete_confirm == i:
                                with st.container():  # Use um container para o popup
                                    st.warning("Tem certeza que deseja apagar esta conversa?")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.button("Sim, apagar", key=f"confirm_delete_{i}", on_click=delete_conversation, args=(i,))
                                    with col2:
                                        st.button("Não, cancelar", key=f"cancel_delete_{i}", 
                                                  on_click=lambda: st.session_state.update({"delete_confirm": None}))

                with col2:
                    # Botão de menu com lógica de alternância
                    if st.button("⚙️", key=f"menu_{i}", on_click=lambda: st.session_state.update(
                        {"menu_open": i if st.session_state.get("menu_open") != i else None}
                    )):
                        pass  # Não faz nada adicional no clique, apenas alterna o estado do menu

                    # Exibir opções apenas se o menu estiver aberto para esta conversa
                    if st.session_state.get("menu_open") == i:
                        with st.sidebar.expander("Opções", expanded=True):
                            new_title = st.text_input("Editar título", value=conversation["title"], key=f"title_{i}")

                            if st.button("Salvar título", key=f"save_title_{i}"):
                                edit_conversation_title(i, new_title)

                            if st.button("Apagar", key=f"delete_{i}"):
                                st.session_state.delete_confirm = i

                        # Popup de confirmação para apagar
                        if st.session_state.delete_confirm == i:
                            with st.sidebar.container():  # Use um container para o popup
                                st.warning("Tem certeza que deseja apagar esta conversa?")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.button("Sim, apagar", key=f"confirm_delete_{i}", on_click=delete_conversation, args=(i,))
                                with col2:
                                    st.button("Não, cancelar", key=f"cancel_delete_{i}", 
                                              on_click=lambda: st.session_state.update({"delete_confirm": None, "menu_open": None}))

    with tab2:
        # Conteúdo da aba de Upload
        uploaded_file = st.file_uploader("Upload 📎:", type=["pdf"])
        if uploaded_file is not None:
            original_filename = uploaded_file.name
            name, ext = os.path.splitext(original_filename)
            ext = ext.lower()
            save_path = os.path.join("./Data", name + ext)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            st.success(f"Arquivo PDF salvo com sucesso em: {save_path}")

            # Recarrega os segmentos de texto após o upload
            all_text_segments = load_existing_pdfs()
            if all_text_segments:
                vectorizer, embeddings = create_embeddings(all_text_segments, client=None)
                st.success("Embeddings criados com sucesso!")
            else:
                vectorizer, embeddings = None, None
                st.error("Nenhum segmento de texto foi encontrado nos PDFs carregados.")
        

# Interface principal (chat e processamento)
st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://i.imgur.com/LlUM5am.jpg" width="150">
        <h1>IA Generativa PI</h1>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.current_conversation.get("messages", []):
    st.chat_message(msg["role"], avatar=avatar if msg["role"] == "assistant" else None).write(msg["content"])

# Entrada do usuário
if prompt := st.chat_input("Qual é a sua dúvida hoje?"):
    api_history = []
    for msg in st.session_state.current_conversation["messages"]:
        role = "model" if msg["role"] == "assistant" else "user"
        api_history.append({
            "role": role,
            "parts": [
                msg["content"]
            ],
        })
    st.session_state.current_conversation["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    try:
        best_match_index, best_match_segment = search_with_embeddings(prompt, vectorizer, embeddings, all_text_segments)
    except Exception as e:
        logging.error(f"Erro ao buscar a primeira vez: {e}")
        best_match_index, best_match_segment = try_fallback_search(prompt, vectorizer, embeddings, all_text_segments)

    if best_match_index is None:
        st.session_state.current_conversation["messages"].append({"role": "assistant", "content": best_match_segment})
        st.chat_message("assistant", avatar=avatar).write(best_match_segment)
    else:
        chat_session = api_config.model.start_chat(
            history=[
                {
                    "role": "model",
                    "parts": [
                        "Você é um assistente pessoal muito ligeiro, fala em gírias e seu nome é Alpha-C. Sua função é servir a todos os colaboradores da Programmers Beyond IT.",
                        "Não responda perguntas que fujam do tema: Programmers Beyond IT. E não invente respostas, use apenas informações dos arquivos que você tem acesso",
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
            ] + api_history
        )

        response = chat_session.send_message(prompt)
        answer = response.text

        st.session_state.current_conversation["messages"].append({"role": "assistant", "content": answer})
        st.chat_message("assistant", avatar=avatar).write(answer)

save_current_conversation()

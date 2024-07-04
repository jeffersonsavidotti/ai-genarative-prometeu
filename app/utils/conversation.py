import json
import os
from datetime import datetime
import streamlit as st

def save_conversations_to_json():
    if not os.path.exists("conversation_history"):
        os.makedirs("conversation_history")
    with open("conversation_history/conversations.json", "w") as f:
        json.dump(st.session_state.conversations, f)

def load_conversations_from_json():
    if os.path.exists("conversation_history/conversations.json"):
        with open("conversation_history/conversations.json", "r") as f:
            return json.load(f)
    return []

def start_new_conversation():
    st.session_state.current_conversation = {
        "title": f"Conversa iniciada em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "messages": [{"role": "assistant", "content": "Como posso te ajudar hoje?"}]
    }

def save_current_conversation():
    if any(msg["role"] == "user" for msg in st.session_state.current_conversation["messages"]):
        current_title = st.session_state.current_conversation["title"]
        conversation_titles = [conv["title"] for conv in st.session_state.conversations]

        if current_title in conversation_titles:
            index = conversation_titles.index(current_title)
            st.session_state.conversations[index] = st.session_state.current_conversation
        else:
            st.session_state.conversations.append(st.session_state.current_conversation)
        
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
            start_new_conversation()
    else:
        st.session_state.delete_confirm = index

def save_and_load_conversation(conversation):
    current_title = st.session_state.current_conversation.get("title")
    conversation_titles = [conv["title"] for conv in st.session_state.conversations]

    if current_title in conversation_titles:
        index = conversation_titles.index(current_title)
        st.session_state.conversations[index] = st.session_state.current_conversation
    else:
        if current_title:
            st.session_state.conversations.append(st.session_state.current_conversation)
    
    st.session_state.current_conversation = conversation

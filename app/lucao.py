import os
from dotenv import load_dotenv, dotenv_values
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.chat import ChatPromptTemplate
import streamlit as st
 
load_dotenv()
 
# temperature = os.getenv("TEMPERATURE")
model = os.getenv("DEPLOYMENT")
 
llm = ChatGoogleGenerativeAI(
        # temperature=temperature,
        google_api_key=os.getenv("GOOGLE_KEY"),
        model=model,
        max_tokens=100,
    )
 
def chatbot_interaction(question):
    prompt = ChatPromptTemplate.from_messages([
        ("system", ""),
        ("user", f"{question}\ ai: ")
    ])
    prompt_text = prompt.format_prompt(question=question)
    response = llm.invoke(prompt_text)
    return response.content
 
st.title("Mayuri AI")
 
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Tuturu~ Como a Mayuri pode te ajudar hoje?"}]
 
for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])
 
if prompt := st.chat_input():
    st.session_state["messages"].append({"role": "user", "content": prompt})
    response = chatbot_interaction(prompt)
    st.session_state["messages"].append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
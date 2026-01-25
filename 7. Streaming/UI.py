import streamlit as st
from langchain_core.messages import HumanMessage
from main import graph, checkpointer

chat_bot = graph.compile(checkpointer=checkpointer)

st.set_page_config(page_title="LangGraph Chatbot", layout="wide")
st.title("🤖 LangGraph Streaming Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

thread_id = "chat_thread_1"
config = {'configurable': {'thread_id': thread_id}}

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        streaming = chat_bot.stream(
            {'messages': [HumanMessage(content=user_input)]},
            config=config,
            stream_mode='messages'
        )
        
        for message_chunk, metadata in streaming:
            full_response += message_chunk.content
            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from chatbot import graph, checkpointer

# Page config
st.set_page_config(page_title="LangGraph Chatbot", page_icon="💬", layout="centered")

st.title("💬 LangGraph Chatbot")
st.caption("A chatbot with memory powered by LangGraph")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit_chat_thread"

# Compile the chatbot graph
chat_bot = graph.compile(checkpointer=checkpointer)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get bot response
    config = {'configurable': {'thread_id': st.session_state.thread_id}}
    response = chat_bot.invoke({'messages': [HumanMessage(content=prompt)]}, config=config)
    bot_message = response['messages'][-1].content
    
    # Display bot response
    with st.chat_message("assistant"):
        st.markdown(bot_message)
    
    # Add bot message to session state
    st.session_state.messages.append({"role": "assistant", "content": bot_message})

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.thread_id = f"streamlit_chat_thread_{len(st.session_state.messages)}"
        st.rerun()
    
    st.divider()
    st.caption(f"Thread ID: {st.session_state.thread_id}")
    st.caption(f"Messages: {len(st.session_state.messages)}")  


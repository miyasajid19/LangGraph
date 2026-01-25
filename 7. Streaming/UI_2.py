import streamlit as st
from langchain_core.messages import HumanMessage
from main import graph, checkpointer
import uuid
from datetime import datetime

chat_bot = graph.compile(checkpointer=checkpointer)

st.set_page_config(page_title="LangGraph Chatbot", layout="wide")
st.title("🤖 LangGraph Streaming Chatbot")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"thread_{uuid.uuid4().hex[:8]}"

if "all_threads" not in st.session_state:
    st.session_state.all_threads = [st.session_state.thread_id]

if "thread_messages" not in st.session_state:
    st.session_state.thread_messages = {st.session_state.thread_id: []}

# Sidebar
with st.sidebar:
    st.title("💬 Chat Manager")
    
    # New Chat Button
    if st.button("🆕 New Chat", use_container_width=True):
        # Generate new thread ID
        new_thread_id = f"thread_{uuid.uuid4().hex[:8]}"
        
        # Save current messages before switching
        st.session_state.thread_messages[st.session_state.thread_id] = st.session_state.messages
        
        # Switch to new thread
        st.session_state.thread_id = new_thread_id
        st.session_state.all_threads.append(new_thread_id)
        st.session_state.messages = []
        st.session_state.thread_messages[new_thread_id] = []
        st.rerun()
    
    st.divider()
    
    # Display current thread ID
    st.caption(f"**Current Thread:** `{st.session_state.thread_id}`")
    
    st.divider()
    
    # My Conversations section
    st.subheader("📂 My Conversations")
    
    # Display all threads as clickable buttons
    for thread in reversed(st.session_state.all_threads):
        # Show thread name with message count
        msg_count = len(st.session_state.thread_messages.get(thread, []))
        button_label = f"💬 {thread} ({msg_count} msgs)"
        
        # Highlight current thread
        if thread == st.session_state.thread_id:
            st.success(f"✓ {button_label}")
        else:
            if st.button(button_label, key=thread, use_container_width=True):
                # Save current messages before switching
                st.session_state.thread_messages[st.session_state.thread_id] = st.session_state.messages
                
                # Switch to selected thread
                st.session_state.thread_id = thread
                st.session_state.messages = st.session_state.thread_messages.get(thread, [])
                st.rerun()

thread_id = st.session_state.thread_id
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
        st.session_state.thread_messages[st.session_state.thread_id] = st.session_state.messages
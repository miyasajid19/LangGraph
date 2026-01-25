import streamlit as st
from langchain_core.messages import HumanMessage
from main import graph, checkpointer
import uuid
from datetime import datetime

chat_bot = graph.compile(checkpointer=checkpointer)


def _convert_lc_messages_to_streamlit(messages: list) -> list[dict]:
    """Map LangChain messages to Streamlit-friendly dicts."""
    mapped = []
    for msg in messages or []:
        role = getattr(msg, "type", "assistant")
        role = "user" if role == "human" else "assistant" if role == "ai" else role
        mapped.append({"role": role, "content": getattr(msg, "content", "")})
    return mapped


def _load_messages_for_thread(thread_id: str) -> list[dict]:
    """Fetch persisted messages for a thread from the checkpointer."""
    try:
        state = chat_bot.get_state({"configurable": {"thread_id": thread_id}})
        messages = state.values.get("messages", []) if state and state.values else []
        return _convert_lc_messages_to_streamlit(messages)
    except Exception:
        return []


def _load_threads_from_db(limit: int = 200) -> list[str]:
    """Return thread ids present in the checkpoint store, newest first."""
    seen = set()
    ordered: list[str] = []
    try:
        for cp in checkpointer.list(config=None, limit=limit):
            cfg = cp.config or {}
            tid = cfg.get("configurable", {}).get("thread_id")
            if tid and tid not in seen:
                seen.add(tid)
                ordered.append(tid)
    except Exception:
        return []
    return ordered

st.set_page_config(page_title="LangGraph Chatbot", layout="wide")
st.title("🤖 LangGraph Streaming Chatbot")

# Initialize session state (load persisted threads first time)
if "initialized" not in st.session_state:
    persisted_threads = _load_threads_from_db()
    if persisted_threads:
        st.session_state.thread_id = persisted_threads[0]
        st.session_state.all_threads = persisted_threads
        st.session_state.thread_messages = {
            tid: _load_messages_for_thread(tid) for tid in persisted_threads
        }
        st.session_state.messages = st.session_state.thread_messages.get(
            st.session_state.thread_id, []
        )
    else:
        st.session_state.thread_id = f"thread_{uuid.uuid4().hex[:8]}"
        st.session_state.all_threads = [st.session_state.thread_id]
        st.session_state.thread_messages = {st.session_state.thread_id: []}
        st.session_state.messages = []
    st.session_state.initialized = True

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
                if thread not in st.session_state.thread_messages:
                    st.session_state.thread_messages[thread] = _load_messages_for_thread(thread)
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
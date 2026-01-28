from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import RemoveMessage
load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V3.1-Terminus',
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)


model= ChatHuggingFace(llm=llm)



class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
    
def chat(state: ChatState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def delete_old_messages(state: ChatState):
    msgs = state["messages"]

    # if more than 10 messages, delete the earliest 6
    if len(msgs) > 10:
        to_remove = msgs[:6]
        print(f"Deleting {len(to_remove)} old messages to manage short-term memory.")
        return {"messages": [RemoveMessage(id=m.id) for m in to_remove]}

    return {}

builder=StateGraph(ChatState)
builder.add_node("chat", chat)
builder.add_node("delete_old_messages", delete_old_messages)
builder.add_edge(START, "chat")
builder.add_edge("chat", "delete_old_messages")
builder.add_edge("delete_old_messages", "chat")
builder.add_edge("delete_old_messages", END)


checkpointer=InMemorySaver()


graph=builder.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    while True:
        user_input = input("User: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        print("User:", user_input)
        current_state = {"messages": [HumanMessage(content=user_input)]}
        print(1)
        result = graph.invoke(current_state, config={"configurable": {"thread_id": "thread_1"}})
        print(2)
        bot_message = result["messages"][-1]
        print(f"Bot: {bot_message.content}")
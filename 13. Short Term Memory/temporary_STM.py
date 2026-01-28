from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id="zai-org/GLM-4.7-Flash",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)
model= ChatHuggingFace(llm=llm)



# defining functions
def chat(state: dict):
    """A simple chat function to interact with the model."""
    messages = state["messages"]
    response = model.invoke(messages)
    # Add bot response to message history
    return {"messages": add_messages(messages, response)}

# defining state
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    
# defining graph
graph = StateGraph(ChatState)


# adding nodes
graph.add_node('chat', chat)


# adding edges
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

# defining checkpointer
checkpointer=InMemorySaver()

workflow=graph.compile(checkpointer=checkpointer)

config={
    "configurable":{
        "thread_id": "thread_1"
    }
}

if __name__ == "__main__":
    while True:
        user_input = input("User: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        # Don't reset state - let it load from memory with thread_id
        result = workflow.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
        # Get the last message (bot's response)
        last_message = result["messages"][-1]
        print(f"Bot: {last_message.content}")
        print("=" * 20)
    
        # memory
        print("Chat History:")
        snap = workflow.get_state(config)
        vals = snap.values
        for m in vals.get("messages", []):
                print("-", type(m).__name__, ":", m.content)
                
        print("==" * 20)
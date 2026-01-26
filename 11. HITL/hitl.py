from typing import Annotated
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import AnyMessage, AIMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
import os

load_dotenv()


llm= HuggingFaceEndpoint(
    # repo_id="deepcogito/cogito-671b-v2.1-FP8",
    # repo_id="zai-org/GLM-4.7-Flash",
    repo_id="moonshotai/Kimi-K2-Instruct-0905",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)



model= ChatHuggingFace(llm=llm)


# defining state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages] 
    
    
def chat_node(state: ChatState):

    decision = interrupt({
        "type": "approval",
        "reason": "Model is about to answer a user question.",
        "question": state["messages"][-1].content,
        "instruction": "Approve this question? yes/no"
    })
    
    if decision.get("approved") == 'no':
        return {"messages": [AIMessage(content="Not approved.")]}

    else:
        response = model.invoke(state["messages"])
        return {"messages": [response]}
    
    

# 3. Build the graph: START -> chat -> END
builder = StateGraph(ChatState)

builder.add_node("chat", chat_node)

builder.add_edge(START, "chat")
builder.add_edge("chat", END)

# Checkpointer is required for interrupts
checkpointer = MemorySaver()

# Compile the app
app = builder.compile(checkpointer=checkpointer)



# Create a new thread id for this conversation
config = {"configurable": {"thread_id": '1234'}}


if __name__ == "__main__":
    while True:
        user_input = input("User: ")
        if user_input.lower() in ['exit', 'quit']:
            break

        from langchain_core.messages import HumanMessage

        user_message = HumanMessage(content=user_input)

        # Initial state with the user message
        initial_state = {"messages": [user_message]}

        # Run the app until interrupt
        for event in app.stream(initial_state, config=config, stream_mode="values"):
            if "__interrupt__" in event:
                # Get the interrupt data
                interrupt_data = event["__interrupt__"][0]
                print(f"\n{interrupt_data.value['reason']}")
                print(f"Question: {interrupt_data.value['question']}")
                
                # Get human approval
                approval = input(f"{interrupt_data.value['instruction']}: ").strip().lower()
                
                # Resume with the decision
                for event in app.stream(Command(resume={"approved": approval}), config=config, stream_mode="values"):
                    pass

        # Get the final state
        final_state = app.get_state(config)
        ai_response = final_state.values["messages"][-1]
        print(f"AI: {ai_response.content}")
        
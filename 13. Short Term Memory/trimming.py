from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages.utils import trim_messages,count_tokens_approximately

load_dotenv()

llm= HuggingFaceEndpoint(
repo_id='deepseek-ai/DeepSeek-V3.1-Terminus',
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)


model= ChatHuggingFace(llm=llm)

# defining states
class MessagesState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages] 

    
    
MAX_TOKENS=200

# defining functions
def chat_model(state:dict):
    """Trim messages to fit within the token limit."""
    messages=state['messages']
    trimmed=trim_messages(messages,strategy='last',max_tokens=MAX_TOKENS,token_counter=count_tokens_approximately)
    print(f"Trimmed {len(messages)-len(trimmed)} messages to fit within {MAX_TOKENS} tokens.")
    
    response=model.invoke(trimmed)
    return {'messages':[response]}



# defining graph
graph=StateGraph(MessagesState)

# adding nodes
graph.add_node('chat_model',chat_model)
# adding edges
graph.add_edge(START,"chat_model")
graph.add_edge("chat_model",END)

# defining checkpointer
checkpointer=InMemorySaver()

workflow=graph.compile(checkpointer=checkpointer)

config={
    "configurable":{
        "thread_id": "thread_1"
    }
}


if __name__=="__main__":
    while True:
        user_input = input("User: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        # invoke the workflow
        output=workflow.invoke(
            {
                "messages":[
                    HumanMessage(content=user_input)
                ]
            },
            config=config
        )
        bot_message=output['messages'][-1]
        print("Bot:",bot_message.content)
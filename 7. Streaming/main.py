from langgraph.graph import StateGraph,START,END
from typing import TypedDict, Annotated,Literal
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os

# llm= HuggingFaceEndpoint(
#     repo_id="deepcogito/cogito-671b-v2.1-FP8",
#     task="text-generation",
#     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
# )
load_dotenv()

llm = HuggingFaceEndpoint(
    # repo_id="zai-org/GLM-4.7-Flash",
    # repo_id="deepseek-ai/DeepSeek-V3.2",
    repo_id="moonshotai/Kimi-K2-Instruct-0905",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

model_hf= ChatHuggingFace(llm=llm)

    
# defining state
class Chatbot_State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages] = Field(..., description="List of messages exchanged in the chat")
    
    
# for memory
checkpointer=MemorySaver()

graph=StateGraph(Chatbot_State)


def chat(state:Chatbot_State)->Chatbot_State:
    messages=state['messages']
    response = model_hf.invoke(messages)
    return {'messages': [response]}


# defining nodes
graph.add_node('chat',chat,description="Chat with the user using LLM")

# defining edges
graph.add_edge(START,'chat')
graph.add_edge('chat',END)


if __name__=="__main__":
    
    chat_bot=graph.compile(checkpointer=checkpointer)
    initial_state={'messages':[]}
    thread_id="chat_thread_1"
    while True:
        user_input=input("User: ")
        if user_input.lower() in ['exit','quit']:
            break
        
        user_message=HumanMessage(content=user_input)
        config={'configurable':{'thread_id':thread_id}}
        # streaming response
        streaming=chat_bot.stream(
            {'messages':[HumanMessage(content=user_input )]},
            config=config,
            stream_mode='messages'
        )
        
        for message_chunk,metadata in streaming:
            print(message_chunk.content,end='',flush=True)
            
        print()  # for new line after completion
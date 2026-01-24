from langgraph.graph import StateGraph,START,END
from typing import TypedDict
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os
load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="zai-org/GLM-4.7-Flash",
    # repo_id="deepseek-ai/DeepSeek-V3.2",
    # repo_id="moonshotai/Kimi-K2-Instruct-0905",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

chat = ChatHuggingFace(llm=llm)


# defining state
class LLM_State(TypedDict):
    user_input: str
    llm_response: str
    


# define the graph
graph=StateGraph(LLM_State)

def get_llm_response(state:LLM_State)->LLM_State:
    user_input=state['user_input']
    response = chat.invoke([HumanMessage(content=user_input)])
    state['llm_response']=response.content
    return state

# adding nodes
graph.add_node('get_llm_response',get_llm_response,description="Get response from LLM based on user input")


# adding edges
graph.add_edge(START,'get_llm_response')
graph.add_edge('get_llm_response',END)


if __name__=="__main__":
    workflow=graph.compile()
    input_state={'user_input':"what is islam in one line"}
    output_state=workflow.invoke(input_state)
    print("Final Output State:",output_state)
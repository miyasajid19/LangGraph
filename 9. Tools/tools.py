from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import os
import requests


load_dotenv()


llm= HuggingFaceEndpoint(
    repo_id="zai-org/GLM-4.7-Flash",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

model= ChatHuggingFace(llm=llm)


search_tool = DuckDuckGoSearchRun()

@tool
def calculator_tool(query: str) -> str:
    """A calculator tool to perform mathematical calculations."""
    try:
        result = eval(query)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"
    

@tool
def get_currency_conversion_rate(source_currency: str, target_currency: str) -> float:
    """
    Fetches the currency conversion rate from source_currency to target_currency.
    Returns only the numeric conversion factor.
    """
    api_key = os.getenv("EXCHANGE_RATE_API_KEY")
    url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{source_currency}/{target_currency}"

    response = requests.get(url)
    data = response.json()

    # Return ONLY the float rate
    return data["conversion_rate"]


tools=[get_currency_conversion_rate,calculator_tool,search_tool]



# binding llms with tools
model_with_tools = model.bind_tools(tools=tools)




# defining state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    

# graph nodes
def chat_node(state: ChatState):
    messages = state['messages']
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


tool_node=ToolNode(tools=tools)


# defining graph structure
graph= StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tool_node", tool_node, condition=tools_condition)


# defining edges
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tool_node", "chat_node")
graph.add_edge("chat_node", END)



workflow=graph.compile()



print(workflow.invoke({
    "messages": [HumanMessage(content="What is the conversion rate between USD and EUR?")]
}))
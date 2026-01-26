from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os
load_dotenv()
llm= HuggingFaceEndpoint(
    repo_id="zai-org/GLM-4.7-Flash",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)
model= ChatHuggingFace(llm=llm)

pdf_path=input("Enter path to PDF document: ")

loader = PyPDFLoader(pdf_path)

docs=loader.load()

splitter= RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

split_docs=splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )

vector_store= FAISS.from_documents(split_docs,embedding=embeddings)

retriever=vector_store.as_retriever(search_type="similarity",search_kwargs={"k":3})

@tool
def rag_tool(query:str)->str:
    """A tool to perform retrieval augmented generation from a PDF document."""
    relevant_docs=retriever.invoke(query)
    context="\n".join([doc.page_content for doc in relevant_docs])
    metadata="\n".join([str(doc.metadata) for doc in relevant_docs])
    return {
        "query":query,
        "context":context,
        "metadata":metadata
    }
    
tools=[rag_tool]


# binding llms with tools
model_with_tools = model.bind_tools(tools=tools)

# defining state
class RAG_State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages] 
    
def rag_chat(state:RAG_State)->RAG_State:
    messages=state['messages']
    response = model_with_tools.invoke(messages)
    return {'messages': [response]}


tool_node=ToolNode(tools)

# defining graph
graph=StateGraph(RAG_State)


# defining nodes
graph.add_node('rag_chat',rag_chat,description="Chat with the user using RAG")
graph.add_node('tools',tool_node,description="Tool Node for RAG")
# defining edges
graph.add_edge(START,'rag_chat')
graph.add_conditional_edges('rag_chat',tools_condition)
graph.add_edge('tools','rag_chat')
graph.add_edge('rag_chat',END)

# Compile and display the graph
workflow=graph.compile()

if __name__=="__main__":
    while True:
        user_input=input("User: ")
        if user_input.lower() in ['exit','quit']:
            break
        
        initial_state={'messages':[HumanMessage(content=user_input)]}
        result=workflow.invoke(initial_state)
        for message in result['messages']:
            print(f"Bot: {message.content}")
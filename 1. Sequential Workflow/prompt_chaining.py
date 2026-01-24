from langgraph.graph import StateGraph,START,END
from typing import TypedDict
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os


load_dotenv()

# llm
llm = HuggingFaceEndpoint(
    # repo_id="zai-org/GLM-4.7-Flash",
    # repo_id="inclusionAI/Ling-1T",
    repo_id="deepseek-ai/DeepSeek-V3.2",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)


# model
chat = ChatHuggingFace(llm=llm)


# defining state
class BlogState(TypedDict):
    topic: str
    outline: str
    blog_post: str
    

# define the graph
graph=StateGraph(BlogState)


def generate_outline(state:BlogState)->BlogState:
    topic=state['topic']
    prompt=f"Generate a detailed outline for a blog post about {topic}."
    response = chat.invoke([HumanMessage(content=prompt)])
    state['outline']=response.content
    print("Generated Outline:",response.content)
    print("==="*20)
    return state


def generate_blog_post(state:BlogState)->BlogState:
    outline=state['outline']
    prompt=f"Write a comprehensive blog post based on the following outline:\n{outline}"
    response = chat.invoke([HumanMessage(content=prompt)])
    state['blog_post']=response.content
    return state


# adding nodes
graph.add_node('generate_outline',generate_outline,description="Generate blog outline from topic")
graph.add_node('generate_blog_post',generate_blog_post,description="Generate blog post from outline")


# adding edges
graph.add_edge(START,'generate_outline')
graph.add_edge('generate_outline','generate_blog_post')
graph.add_edge('generate_blog_post',END)


if __name__=="__main__":
    workflow=graph.compile()
    input_state={'topic':"The Future of Artificial Intelligence"}
    output_state=workflow.invoke(input_state)
    print("Final Output State:",output_state)
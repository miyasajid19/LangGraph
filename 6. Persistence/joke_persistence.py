from langgraph.graph import StateGraph,START,END
from typing import TypedDict, Annotated,Literal
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os


load_dotenv()


# llm setup
llm = HuggingFaceEndpoint(
    # repo_id="moonshotai/Kimi-K2-Instruct-0905",
    # repo_id="deepcogito/cogito-671b-v2.1-FP8",
    repo_id="zai-org/GLM-4.7-Flash",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)


model_hf= ChatHuggingFace(llm=llm)


# defining state
class JokeState(TypedDict):

    topic: str
    joke: str
    explanation: str
    
    
def generate_joke(state:JokeState):
    print("Generating joke...")
    topic=state['topic']
    prompt=f"Generate a funny joke about {topic}."
    response = model_hf.invoke([HumanMessage(content=prompt)])
    state['joke']=response.content
    return {'joke':response.content}


def generate_explanation(state:JokeState):
    print("Generating explanation...")
    joke=state['joke']
    prompt=f"Explain the following joke in simple terms:\n{joke}"
    response = model_hf.invoke([HumanMessage(content=prompt)])
    state['explanation']=response.content
    return {'explanation':response.content}


# define the graph
graph=StateGraph(JokeState)

# adding nodes
graph.add_node('generate_joke',generate_joke,description="Generate a joke based on the given topic")
graph.add_node('generate_explanation',generate_explanation,description="Generate an explanation for the given joke")

# adding edges
graph.add_edge(START,'generate_joke')
graph.add_edge('generate_joke','generate_explanation')
graph.add_edge('generate_explanation',END)


# for memory persistence
checkpointer=InMemorySaver()

workflow=graph.compile(checkpointer=checkpointer)


config_1={'configurable':{'thread_id':'joke_thread_1'}}
print(workflow.invoke({'topic':'programming'},config=config_1))

# getting state
print("Persisted State:",workflow.get_state(config_1))

print('\n'*5)

# listing all states
print("All Persisted States:")
for state in list(workflow.get_state_history(config_1)):
    print(state)
    print("===="*20)
    
    
config_2={'configurable':{'thread_id':'joke_thread_2'}}
print(workflow.invoke({'topic':'artificial intelligence'},config=config_2))

# getting state
print("Persisted State:",workflow.get_state(config_2))
print('\n'*5)
# listing all states
print("All Persisted States:")
for state in list(workflow.get_state_history(config_2)):
    print(state)
    print("===="*20)
    
    
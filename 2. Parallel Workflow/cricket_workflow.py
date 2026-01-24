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
    # repo_id="deepcogito/cogito-671b-v2.1",
    repo_id="deepcogito/cogito-671b-v2.1-FP8",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)


# model
model= ChatHuggingFace(llm=llm)


# defining state
class Cricketer_State(TypedDict):
    
    name: str
    age: int
    batting_style: str
    bowling_style: str
    role: str
    total_innings: int
    profile_summary: str
    
    
    strike_rate: float
    average: float
    total_runs: int
    total_wickets: int
    
# define the graph
graph=StateGraph(Cricketer_State)


# Note for parallel workflow, each function should return only the part of the state it modifies.
def calculate_strike_rate(state:Cricketer_State):
    print("Calculating strike rate...")
    total_runs=state['total_runs']
    # assuming total balls faced is 1000 for calculation
    total_balls_faced=1000
    strike_rate=(total_runs/total_balls_faced)*100
    state['strike_rate']=strike_rate
    print("Strike rate calculated.")
    return {"strike_rate": strike_rate}

def calculate_average(state:Cricketer_State):
    print("Calculating average...")
    total_runs=state['total_runs']
    # assuming total innings played is 25 for calculation
    total_innings=state['total_innings']
    average=total_runs/total_innings
    state['average']=average
    print("Average calculated.")
    return {"average": average}

def generate_profile_summary(state:Cricketer_State):
    print("Generating profile summary...")
    name=state['name']
    age=state['age']
    batting_style=state['batting_style']
    bowling_style=state['bowling_style']
    role=state['role']
    total_innings=state['total_innings']
    strike_rate=state['strike_rate']
    average=state['average']
    total_runs=state['total_runs']
    total_wickets=state['total_wickets']
    
    prompt=f"Create a detailed cricket player profile for {name}, a {age}-year-old {role} who bats {batting_style} and bowls {bowling_style}. They have played {total_innings} innings, scored {total_runs} runs with a strike rate of {strike_rate:.2f} and an average of {average:.2f}, and taken {total_wickets} wickets."
    
    response = model.invoke([HumanMessage(content=prompt)])
    state['profile_summary']=response.content
    print("Profile summary generated.")
    return {"profile_summary": response.content}


# adding nodes
graph.add_node('calculate_strike_rate',calculate_strike_rate,description="Calculate strike rate from total runs")
graph.add_node('calculate_average',calculate_average,description="Calculate average from total runs and innings")
graph.add_node('generate_profile_summary',generate_profile_summary,description="Generate cricketer profile summary")
# adding edges
graph.add_edge(START,'calculate_strike_rate')
graph.add_edge(START,'calculate_average')
graph.add_edge('calculate_strike_rate','generate_profile_summary')
graph.add_edge('calculate_average','generate_profile_summary')
graph.add_edge('generate_profile_summary',END)


if __name__=="__main__":
    workflow=graph.compile()
    input_state={
        'name':"MS Dhoni",
        'age':42,
        'batting_style':"Right-hand bat",
        'bowling_style':"Right-arm medium",
        'role':"Wicketkeeper-batsman",
        'total_innings':350,
        'total_runs':10773,
        'total_wickets':1
    }
    output_state=workflow.invoke(input_state)
    print("Final Output State:",output_state)
from langgraph.graph import StateGraph,START,END
from typing import TypedDict

# defining state
class BMI_State(TypedDict):
    weight: float  # in kilograms
    height: float  # in meters
    BMI: float  # Body Mass Index
    category: str  # BMI Category
    
# define the graph
graph=StateGraph(BMI_State)


def calculate_BMI(state:BMI_State)->BMI_State:
    weight=state['weight']
    height=state['height']
    BMI=weight/(height**2)
    state['BMI']=BMI
    return state

def determine_category(state:BMI_State)->BMI_State:
    BMI=state['BMI']
    if BMI<18.5:
        category='Underweight'
    elif 18.5<=BMI<24.9:
        category='Normal weight'
    elif 25<=BMI<29.9:
        category='Overweight'
    else:
        category='Obesity'
    state['category']=category
    return state

# adding nodes
graph.add_node('calculate_BMI',calculate_BMI,description="Calculate BMI from weight and height")
graph.add_node('determine_category',determine_category,description="Determine BMI category from BMI")



# adding edges
graph.add_edge(START,'calculate_BMI')
graph.add_edge('calculate_BMI','determine_category')
graph.add_edge('determine_category',END)

if __name__=="__main__":
    workflow=graph.compile()
    input_state={'weight':77,'height':1.695}
    output_state=workflow.invoke(input_state)
    print("Final Output State:",output_state)
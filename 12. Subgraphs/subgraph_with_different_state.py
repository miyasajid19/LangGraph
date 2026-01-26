from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()




class SubState(TypedDict):

    input_text: str
    translated_text: str
    
llm_subgraph=  HuggingFaceEndpoint(
    repo_id="zai-org/GLM-4.7-Flash",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

model_subgraph= ChatHuggingFace(llm=llm_subgraph)

def translate_text(state: SubState):

    prompt = f"""
Translate the following text to urdu.
Keep it natural and clear. Do not add extra content.

Text:
{state["input_text"]}
""".strip()
    
    translated_text = model_subgraph.invoke(prompt).content

    return {'translated_text': translated_text}


subgraph= StateGraph(SubState)

subgraph.add_node("translate_text", translate_text)

subgraph.add_edge(START, "translate_text")
subgraph.add_edge("translate_text", END)

subgraph_workflow= subgraph.compile()


class ParentState(TypedDict):

    question: str
    answer_eng: str
    answer_urdu: str
    
parent_llm=  HuggingFaceEndpoint(
    repo_id="zai-org/GLM-4.7-Flash",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

parent_model= ChatHuggingFace(llm=parent_llm)

def generate_answer(state: ParentState):

    answer=parent_model.invoke(
        f"Answer the question in English:\n{state['question']}"
    ).content
    
    return {'answer_eng': answer}



def translate_answer(state: ParentState):

    subgraph_input= {
        "input_text": state['answer_eng']
    }
    
    result= subgraph_workflow.invoke(subgraph_input)
    
    return {'answer_urdu': result['translated_text']}



parent_builder= StateGraph(ParentState)

parent_builder.add_node("generate_answer", generate_answer)
parent_builder.add_node("translate_answer", translate_answer)
parent_builder.add_edge(START, "generate_answer")
parent_builder.add_edge("generate_answer", "translate_answer")
parent_builder.add_edge("translate_answer", END)

parent_workflow= parent_builder.compile()

if __name__=="__main__":
    while True:
        user_input= input("Enter your question (or 'exit' to quit): ")
        if user_input.lower() in ['exit', 'quit']:
            break

        initial_state= {
            "question": user_input,
            "answer_eng": "",
            "answer_urdu": ""
        }

        final_state= parent_workflow.invoke(initial_state)

        print("\nAnswer in English:")
        print(final_state['answer_eng'])
        print("\nAnswer in urdu:")
        print(final_state['answer_urdu'])
        print("\n" + "="*50 + "\n")
        
        
        
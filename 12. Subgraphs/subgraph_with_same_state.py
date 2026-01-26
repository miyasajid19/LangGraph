from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
load_dotenv()
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

subgraph_llm=  HuggingFaceEndpoint(
    repo_id="deepcogito/cogito-671b-v2.1-FP8",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)
subgraph_model= ChatHuggingFace(llm=subgraph_llm)

def translate_text(state: ParentState):

    prompt = f"""
Translate the following text to Urdu.
Keep it natural and clear. Do not add extra content.

Text:
{state["answer_eng"]}
""".strip()
    
    translated_text = subgraph_model.invoke(prompt).content

    return {'answer_urdu': translated_text}

subgraph_builder = StateGraph(ParentState)

subgraph_builder.add_node('translate_text', translate_text)

subgraph_builder.add_edge(START, 'translate_text')
subgraph_builder.add_edge('translate_text', END)

subgraph = subgraph_builder.compile()

def generate_answer(state: ParentState):

    answer = parent_model.invoke(f"You are a helpful assistant. Answer clearly.\n\nQuestion: {state['question']}").content
    return {'answer_eng': answer}

parent_builder = StateGraph(ParentState)

parent_builder.add_node("answer", generate_answer)
parent_builder.add_node("translate", subgraph)

parent_builder.add_edge(START, 'answer')
parent_builder.add_edge('answer', 'translate')
parent_builder.add_edge('translate', END)


if __name__ == "__main__":
    graph = parent_builder.compile()
    while True:
        user_input = input("User: ")

        initial_state = {
            "question": user_input,
            "answer_eng": "",
            "answer_urdu": ""
        }

        final_state = graph.invoke(initial_state)

        print("\nAnswer in English:")
        print(final_state['answer_eng'])
        print("\nAnswer in urdu:")
        print(final_state['answer_urdu'])
        
        print("\n" + "="*50 + "\n")
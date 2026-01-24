from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
import operator
from dotenv import load_dotenv
import os


load_dotenv()

model= ChatGoogleGenerativeAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        model="gemini-2.5-flash"
    )

llm= HuggingFaceEndpoint(
    repo_id="deepcogito/cogito-671b-v2.1-FP8",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

model_hf= ChatHuggingFace(llm=llm)

class Evaluation_schema(BaseModel):
    
    feedback: Annotated[str, Field(description="Detailed feedback on the essay")]
    score: Annotated[int, Field(description="Score out of 10", ge=0, le=10)]
    
    
structured_model=model.with_structured_output(Evaluation_schema)

prompt_1= PromptTemplate(
    input_variables=["essay"],
    template="Provide a detailed feedback and score out of 10 for the following essay:\n{essay}"
)

prompt_2= PromptTemplate(
    input_variables=["language_feedback","analytical_feedback","clarity_feedback"],
    template=(
        "Based on the following individual feedbacks, provide an overall feedback and final score out of 30:\n"
        "Language Feedback: {language_feedback}\n"
        "Analytical Feedback: {analytical_feedback}\n"
        "Clarity Feedback: {clarity_feedback}\n"
    )
    
)

# defining state
class Essay_State(TypedDict):
    essay: str
    language_feedback: str
    analytical_feedback: str
    clarity_feedback: str
    individual_score:Annotated[int, Field(description="Individual score out of 10"),operator.add]
    over_all_feedback: str
    final_score: int
    
    
def evaluate_language(state:Essay_State)->Essay_State:
    print("Evaluating language...")
    essay=state['essay']
    prompt_filled=prompt_1.format(essay=essay)
    response=structured_model.invoke([HumanMessage(content=prompt_filled)])
    feedback=response.feedback
    score=response.score
    state['language_feedback']=feedback
    state['individual_score']=score
    print("Language evaluation completed.")
    return {"language_feedback": feedback, "individual_score": score}

def evaluate_analytical(state:Essay_State)->Essay_State:
    print("Evaluating analytical skills...")
    essay=state['essay']
    prompt_filled=prompt_1.format(essay=essay)
    response=structured_model.invoke([HumanMessage(content=prompt_filled)])
    feedback=response.feedback
    score=response.score
    state['analytical_feedback']=feedback
    state['individual_score']=score
    print("Analytical evaluation completed.")
    return {"analytical_feedback": feedback, "individual_score": score}


def evaluate_clarity(state:Essay_State)->Essay_State:
    print("Evaluating clarity...")
    essay=state['essay']
    prompt_filled=prompt_1.format(essay=essay)
    response=structured_model.invoke([HumanMessage(content=prompt_filled)])
    feedback=response.feedback
    score=response.score
    state['clarity_feedback']=feedback
    state['individual_score']=score
    print("Clarity evaluation completed.")
    return {"clarity_feedback": feedback, "individual_score": score}


def aggregate_feedback(state:Essay_State)->Essay_State:
    prompt= prompt_2.format(
        language_feedback=state['language_feedback'],
        analytical_feedback=state['analytical_feedback'],
        clarity_feedback=state['clarity_feedback']
    )
    
    response= model_hf.invoke([HumanMessage(content=prompt)])
    final_score=(sum([
        state['individual_score']
    ]))/3
    return {"over_all_feedback": response.content, "final_score": float(final_score)}




# define the graph
graph=StateGraph(Essay_State)


# adding nodes
graph.add_node('evaluate_language',evaluate_language,description="Evaluate essay language")
graph.add_node('evaluate_analytical',evaluate_analytical,description="Evaluate essay analytical skills")
graph.add_node('evaluate_clarity',evaluate_clarity,description="Evaluate essay clarity")
graph.add_node('aggregate_feedback',aggregate_feedback,description="Aggregate individual feedbacks into overall feedback")

# adding edges
graph.add_edge(START,'evaluate_language')
graph.add_edge(START,'evaluate_analytical')
graph.add_edge(START,'evaluate_clarity')
graph.add_edge('evaluate_language','aggregate_feedback')
graph.add_edge('evaluate_analytical','aggregate_feedback')
graph.add_edge('evaluate_clarity','aggregate_feedback')
graph.add_edge('aggregate_feedback',END)



if __name__=="__main__":
    workflow=graph.compile()
    essay="""
    Artificial Intelligence (AI) has rapidly shifted from the realm of speculative fiction into a transformative force shaping contemporary society. From healthcare diagnostics to financial markets and creative industries, AI systems increasingly influence how decisions are made, knowledge is produced, and power is distributed. While AI offers unprecedented opportunities for efficiency, innovation, and problem-solving, it also raises profound ethical, social, and political challenges. This essay argues that AI should be understood not merely as a technological advancement, but as a socio-technical system whose benefits and risks depend on human values, governance, and accountability.

    At its core, AI refers to computational systems designed to perform tasks that typically require human intelligence, such as learning, reasoning, pattern recognition, and language processing. Recent advances in machine learning—particularly deep learning—have enabled AI systems to analyse vast datasets with remarkable accuracy. In healthcare, for example, AI tools can detect diseases such as cancer earlier than human clinicians in some cases, improving patient outcomes and reducing costs. Similarly, in environmental science, AI models are used to predict climate patterns and optimise energy consumption. These developments demonstrate AI’s potential to address complex global challenges more effectively than traditional methods.

    However, the power of AI systems is inseparable from the data on which they are trained. Because AI learns from historical data, it can replicate and even amplify existing social biases. Numerous studies have shown that algorithmic decision-making systems used in hiring, policing, and credit scoring can discriminate against marginalised groups. This raises serious concerns about fairness and justice, particularly when such systems are deployed at scale with limited transparency. Unlike human decision-makers, AI systems often operate as “black boxes,” making it difficult to understand or challenge their outputs. As a result, accountability becomes blurred, undermining trust in institutions that rely on these technologies.

    Another significant concern relates to employment and economic inequality. Automation driven by AI threatens to displace workers in sectors such as manufacturing, transportation, and administrative services. While proponents argue that AI will create new jobs and increase productivity, the transition may be uneven, disproportionately affecting low-skilled workers and widening socioeconomic divides. Without proactive policies—such as reskilling programmes and social safety nets—the benefits of AI risk being concentrated among a small group of technology firms and highly skilled professionals.

    Ethical considerations also extend to the autonomy and control of AI systems. As AI becomes more capable, questions arise about how much decision-making authority should be delegated to machines. In areas such as military technology and autonomous weapons, the stakes are particularly high, as errors or misuse could result in catastrophic harm. This highlights the need for robust ethical frameworks and international regulation to ensure that AI development aligns with human values and respects fundamental rights.

    In conclusion, Artificial Intelligence represents one of the most significant technological developments of the modern era. Its potential to improve lives and solve pressing global problems is undeniable, yet its risks are equally substantial. A first-class understanding of AI requires moving beyond technological optimism or fear, and instead adopting a critical, interdisciplinary approach that considers ethical, social, and political dimensions. Ultimately, the future of AI will not be determined by algorithms alone, but by the choices societies make about how these systems are designed, governed, and used.
    """
    input_state={'essay':essay}
    output_state=workflow.invoke(input_state)
    print("Final Output State:",output_state)
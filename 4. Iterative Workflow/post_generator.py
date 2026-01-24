from langgraph.graph import StateGraph,START,END
from typing import TypedDict, Annotated,Literal
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
import operator
from dotenv import load_dotenv
import os


load_dotenv()

google_model= ChatGoogleGenerativeAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        model="gemini-2.5-flash"
    )


llm= HuggingFaceEndpoint(
    # repo_id="deepseek-ai/DeepSeek-Prover-V2-671B",
    # repo_id="deepseek-ai/DeepSeek-V3-0324",
    repo_id="moonshotai/Kimi-K2-Instruct",
    task="text-generation", 
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

model_hf= ChatHuggingFace(llm=llm)



class PostEvaluation(BaseModel):
    evaluation:Literal['approved','rejected']= Field(description="Evaluation result for the generated post")    
    feedback: str= Field(description="Feedback for the generated post")
    
    
structured_evaluation_model=google_model.with_structured_output(PostEvaluation)

# post state
# state
class PostState(TypedDict):

    topic: str
    post_caption: str
    evaluation: Literal["approved", "rejected"]
    feedback: str
    iteration: int
    max_iteration: int

    post_caption_history: Annotated[list[str], operator.add]
    feedback_history: Annotated[list[str], operator.add]
    
    
    
def generate_post_caption(state:PostState)->PostState:
    print("Generating post caption...")
    topic=state['topic']
    prompt=f"Generate a catchy social media post caption about {topic}."
    response = model_hf.invoke([HumanMessage(content=prompt)])
    state['post_caption']=response.content
    print("Post caption generated.")
    return state
def evaluate_post_caption(state: PostState):
    print("Evaluating post caption...")
    messages = [
        SystemMessage(content="You are a ruthless, no-laugh-given Twitter critic. You evaluate tweets based on humor, originality, virality, and tweet format."),
        HumanMessage(content=f"""
Evaluate the following tweet:
Tweet: "{state['post_caption']}"
Use the criteria below to evaluate the tweet:
1. Originality – Is this fresh, or have you seen it a hundred times before?  
2. Humor – Did it genuinely make you smile, laugh, or chuckle?  
3. Punchiness – Is it short, sharp, and scroll-stopping?  
4. Virality Potential – Would people retweet or share it?  
5. Format – Is it a well-formed tweet (not a setup-punchline joke, not a Q&A joke, and under 280 characters)?
Auto-reject if:
- It's written in question-answer format (e.g., "Why did..." or "What happens when...")
- It exceeds 280 characters
- It reads like a traditional setup-punchline joke
- Don't end with generic, throwaway, or deflating lines that weaken the humor (e.g., “Masterpieces of the auntie-uncle universe” or vague summaries)
### Respond ONLY in structured format:
- evaluation: "approved" or "needs_improvement"  
- feedback: One paragraph explaining the strengths and weaknesses 
""")
    ]
    response = structured_evaluation_model.invoke(messages)
    return {
        'evaluation': response.evaluation,
        'feedback': response.feedback,
        'feedback_history': [response.feedback]
    }

def optimize_post_caption(state: PostState):
    print("Optimizing post caption...")
    messages = [
        SystemMessage(content="You punch up tweets for virality and humor based on given feedback."),
        HumanMessage(content=f"""
Improve the tweet based on this feedback:
"{state['feedback']}"
Topic: "{state['topic']}"
Original Tweet:
{state['post_caption']}
Re-write it as a short, viral-worthy tweet. Avoid Q&A style and stay under 280 characters.
""")
    ]
    response = model_hf.invoke(messages).content
    iteration = state['iteration'] + 1
    return {'post_caption': response, 'iteration': iteration, 'post_caption_history': [response]}



def route_evaluation(state: PostState):

    if state['evaluation'] == 'approved' or state['iteration'] >= state['max_iteration']:
        return 'approved'
    else:
        return 'rejected'
    
    
graph=StateGraph(PostState)

# adding nodes
graph.add_node('generate_post_caption',generate_post_caption,description="Generate social media post caption from topic")
graph.add_node('evaluate_post_caption',evaluate_post_caption,description="Evaluate generated post caption")
graph.add_node('optimize_post_caption',optimize_post_caption,description="Optimize post caption based on feedback")
graph.add_node('route_evaluation',route_evaluation,description="Route based on evaluation result")

# adding edges
graph.add_edge(START,'generate_post_caption')
graph.add_edge('generate_post_caption','evaluate_post_caption')
graph.add_conditional_edges('evaluate_post_caption',route_evaluation,{'approved':END,'rejected':'optimize_post_caption'})
graph.add_edge('optimize_post_caption','evaluate_post_caption')



if __name__=="__main__":
    workflow=graph.compile()
    input_state={
        'topic':"The impact of AI on modern society",
        'iteration':0,
        'max_iteration':3,
        'post_caption_history':[],
        'feedback_history':[]
    }
    output_state=workflow.invoke(input_state)
    print("\n\n\n")
    print("Topic :: ",input_state['topic'])
    
    
    print("\n\n\n")
    print("Final Post Caption :: \n")
    print(output_state['post_caption'])
    
    print("\n\n\n")
    print("Post Caption History :: \n")
    for idx,caption in enumerate(output_state['post_caption_history']):
        print(f"Iteration {idx+1}: {caption}")
        
    print("\n\n\n")
    print("Feedback History :: \n")
    for idx,feedback in enumerate(output_state['feedback_history']):
        print(f"Iteration {idx+1}: {feedback}")
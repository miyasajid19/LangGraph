from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
import operator
from dotenv import load_dotenv
import os


load_dotenv()

# google model
google_model= ChatGoogleGenerativeAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        model="gemini-2.5-flash"
    )

# huggingface model
hf_llm= HuggingFaceEndpoint(
    repo_id="deepcogito/cogito-671b-v2.1-FP8",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)


hf_model= ChatHuggingFace(llm=hf_llm)


# structured output schema 
class SentimentSchema(BaseModel):
    sentiment: Literal["positive", "negative"]

class DiagnosisSchema(BaseModel):
    issue_type:Literal['UX','Performance','Security','Support','Other']= Field(description="Type of issue identified in the review")
    tone:Literal['frustrated','calm','angry']= Field(description="Tone of the review")
    urgency:Literal['high','medium','low']= Field(description="Urgency level for addressing the review")
    


structured_google_model_for_sentiment=google_model.with_structured_output(SentimentSchema)
structured_google_model_for_diagnosis=google_model.with_structured_output(DiagnosisSchema)


# defining state
class Review_State(TypedDict):
    review: str
    sentiment: Literal["positive", "negative"]
    diagnosis:dict
    reply: str
    


# functions
def analyze_sentiment(state:Review_State)->Review_State:
    print("Analyzing sentiment...")
    review=state['review']
    prompt=f"Analyze the sentiment of the following customer review and classify it as 'positive' or 'negative':\n{review}"
    response=structured_google_model_for_sentiment.invoke([HumanMessage(content=prompt)])
    state['sentiment']=response.sentiment
    print(f"Sentiment analyzed: {response.sentiment}")
    return state

def positive_reply(state:Review_State)->Review_State:
    print("Generating reply for positive review...")
    review=state['review']
    prompt=f"Generate a warm and appreciative reply to the following positive customer review:\n{review}"
    response=hf_model.invoke([HumanMessage(content=prompt)])
    state['reply']=response.content
    print("Reply for positive review generated.")
    return state

def diagnose_issue(state:Review_State)->Review_State:
    print("Diagnosing issue from negative review...")
    review=state['review']
    prompt=f"Diagnose the issues mentioned in the following negative customer review. Identify the issue type (UX, Performance, Security, Support, Other), tone (frustrated, calm, angry), and urgency (high, medium, low):\n{review}"
    response=structured_google_model_for_diagnosis.invoke([HumanMessage(content=prompt)])
    state['diagnosis']=response.model_dump()
    print("Issue diagnosed from negative review.")
    return state
def negative_reply(state:Review_State)->Review_State:
    print("Generating reply for negative review...")
    review=state['review']
    diagnosis=state['diagnosis']
    prompt=f"Generate a professional and empathetic reply to the following negative customer review, addressing the diagnosed issues: {diagnosis}\nReview: {review}"
    response=hf_model.invoke([HumanMessage(content=prompt)])
    state['reply']=response.content
    print("Reply for negative review generated.")
    return state

def check_sentiment_and_route(state:Review_State)->str:
    sentiment=state['sentiment']
    if sentiment=='positive':
        return 'positive_reply'
    else:
        return 'diagnose_issue'
    

# adding nodes
graph=StateGraph(Review_State)
graph.add_node('analyze_sentiment',analyze_sentiment,description="Analyze sentiment of the review")
graph.add_node('positive_reply',positive_reply,description="Generate reply for positive review" )
graph.add_node('diagnose_issue',diagnose_issue,description="Diagnose issues from negative review")
graph.add_node('negative_reply',negative_reply,description="Generate reply for negative review")


# adding edges
graph.add_edge(START,'analyze_sentiment')
graph.add_conditional_edges('analyze_sentiment',check_sentiment_and_route)
graph.add_edge('diagnose_issue','negative_reply')
graph.add_edge('positive_reply',END)
graph.add_edge('negative_reply',END)


if __name__=="__main__":
    workflow=graph.compile()
    review="""I recently purchased your software and have been extremely disappointed with its performance. The user interface is confusing and not intuitive at all, making it difficult to navigate through the features. Additionally, I've encountered several bugs that have caused the application to crash multiple times, leading to a loss of important data. Customer support has been unresponsive, which only adds to my frustration. I expected a much higher quality product for the price I paid. Please address these issues promptly."""
    input_state={'review':review}
    output_state=workflow.invoke(input_state)
    print("Final Output State:",output_state)
    print("\n\n\n")
    
    review="""I absolutely love this product! It has completely transformed the way I manage my daily tasks. The user interface is sleek and easy to navigate, making it a joy to use. I've noticed a significant increase in my productivity since I started using it. The features are well thought out and cater to all my needs. Customer support has also been fantastic, always quick to respond and very helpful. I highly recommend this software to anyone looking for an efficient solution to organize their work and life."""
    input_state={'review':review}
    output_state=workflow.invoke(input_state)
    print("Final Output State:",output_state)
    print("\n\n\n")
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from typing import TypedDict, Annotated
from langchain_core.prompts import PromptTemplate
from langgraph.constants import Send
import operator
from functools import reduce

# defining n individual llms in the council
N_MODELS = 6
i=0.5
models = [
    ChatOllama(model="kimi-k2-thinking:cloud"),
    ChatOllama(model="ministral-3:14b-cloud"),
    ChatOllama(model="qwen3-vl:235b-cloud"),
    ChatOllama(model="qwen3-coder:480b-cloud"),
    ChatOllama(model="deepseek-r1:1.5b"),
    ChatOllama(model="gpt-oss:120b-cloud"),
]

# master model for final synthesis
master_model = ChatOllama(model="deepseek-v3.1:671b-cloud", temperature=0.3)


# defining individual model state
class IndividualState(TypedDict):
    query: str
    model_id: int
    response: str
    scores: dict[int, list[float]]  # scores[model_id] = list of scores from all models
    
    
    
def merge_dicts(left: dict, right: dict) -> dict:
    """Merge two dictionaries"""
    if left is None:
        return right
    if right is None:
        return left
    return {**left, **right}


class CouncilState(TypedDict):
    query: str
    responses: Annotated[dict[int, str], merge_dicts]  # {model_id: response} - uses reducer for parallel writes
    all_scores: Annotated[dict[int, list[float]], merge_dicts]  # {model_id: scores} - uses reducer for parallel writes
    aggregated_scores: dict[int, float]  # {model_id: average_score}
    threshold: float  # minimum score to generate final response
    meets_threshold: bool
    final_response: str
    iteration: int  # current iteration number
    max_iterations: int  # maximum iterations allowed
    
    
    
def generate_response(state: CouncilState, model_id: int) -> dict:
    """Generate response from a specific model (Fan-out step)"""
    response = models[model_id].invoke(state['query'])
    return {"responses": {model_id: response.content if hasattr(response, 'content') else str(response)}}


def rank_responses(state: CouncilState, model_id: int) -> dict:
    """Model ranks all other responses (each model ranks n-1 responses)"""
    other_model_ids = [i for i in range(N_MODELS) if i != model_id]
    other_responses = {i: state['responses'][i] for i in other_model_ids}
    
    ranking_prompt = f"""You are evaluating responses to this query: "{state['query']}"
    
    Other models' responses:
    {chr(10).join([f"Model {i}: {other_responses[i]}" for i in other_model_ids])}
    
    Rate each response on a scale of 0-10 based on:
    1. Relevance to the query
    2. Clarity and coherence
    3. Comprehensiveness
    4. Accuracy
    
    Respond ONLY with comma-separated scores (no text), e.g., "8,7,9"
    """
    
    response = models[model_id].invoke(ranking_prompt)
    response_text = response.content if hasattr(response, 'content') else str(response)
    
    try:
        scores = [float(s.strip()) for s in response_text.strip().split(',')]
        scores = scores[:len(other_model_ids)]  # Ensure correct number of scores
    except:
        scores = [5.0] * len(other_model_ids)  # Default scores if parsing fails
    
    return {"all_scores": {model_id: scores}}


def fan_out_generate_responses(state: CouncilState):
    """Fan-out: Create parallel tasks for each model to generate responses"""
    return [Send("generate_response_node", {"query": state["query"], "model_id": i}) 
            for i in range(N_MODELS)]


def generate_response_node(state):
    """Generate response from a specific model (Fan-out step)"""
    model_id = state['model_id']
    response = models[model_id].invoke(state['query'])
    print(f"Model {model_id} responded.\n\nresponse: {response}")
    response_text = response.content if hasattr(response, 'content') else str(response)
    return {"responses": {model_id: response_text}}


def aggregate_responses(state: CouncilState) -> CouncilState:
    """Fan-in: Aggregate all responses and prepare for ranking"""
    return {
        "query": state["query"],
        "responses": state.get("responses", {}),
        "all_scores": {},
        "aggregated_scores": {},
        "threshold": state.get("threshold", 9.0),
        "meets_threshold": False,
        "final_response": "",
        "iteration": state.get("iteration", 1),
        "max_iterations": state.get("max_iterations", 3)
    }
    
    
    

def check_threshold_old(state: CouncilState):
    """Conditional router: Check if threshold is met"""
    print("checking threshold")
    max_score = max(state.get("aggregated_scores", {}).values()) if state.get("aggregated_scores") else 0
    meets_threshold = max_score >= state.get("threshold", 9.0)
    return "generate_master_response" if meets_threshold else "end"


def generate_master_response(state: CouncilState) -> CouncilState:
    """Master model generates final response based on aggregated scores"""
    aggregated_scores = state.get("aggregated_scores", {})
    responses = state.get("responses", {})
    print("master model invoked")
    # Get the best response(s)
    if aggregated_scores:
        best_model_id = max(aggregated_scores, key=aggregated_scores.get)
        best_score = aggregated_scores[best_model_id]
        best_response = responses.get(best_model_id, "")
        
        synthesis_prompt = f"""Query: {state["query"]}

Best response from council (score: {best_score:.1f}/10):
{best_response}

Based on this response and the query, provide an enhanced final answer. 
Consider improvements, additional insights, or refinements while maintaining the core of the best response."""
        
        final = master_model.invoke(synthesis_prompt)
        final_text = final.content if hasattr(final, 'content') else str(final)
    else:
        final_text = "Unable to generate response: threshold not met."
    
    return {
        "query": state["query"],
        "responses": state["responses"],
        "all_scores": state.get("all_scores", {}),
        "aggregated_scores": state.get("aggregated_scores", {}),
        "threshold": state.get("threshold", 9.0),
        "meets_threshold": True,
        "final_response": final_text,
        "iteration": state.get("iteration", 1),
        "max_iterations": state.get("max_iterations", 3)
    }
    
    
    
def fan_out_rank_responses(state: CouncilState):
    """Fan-out: Each model ranks other models' responses"""
    return [
        Send(
            "rank_response_node",
            {
                "query": state["query"],
                "responses": state["responses"],
                "model_id": i
            }
        )
        for i in range(N_MODELS)
    ]



def rank_response_node(state):
    """Each model ranks other models"""
    model_id = state["model_id"]
    other_model_ids = [i for i in range(N_MODELS) if i != model_id]

    ranking_prompt = f"""You are evaluating responses to this query:
"{state['query']}"

Other models' responses:
{chr(10).join([f"Model {i}: {state['responses'][i]}" for i in other_model_ids])}

Rate each response from 0-10.
Respond ONLY with comma-separated numbers.
Example: 8,7
"""

    response = models[model_id].invoke(ranking_prompt)
    response_text = response.content if hasattr(response, "content") else str(response)

    try:
        scores = [float(s.strip()) for s in response_text.split(",")]
        scores = scores[:len(other_model_ids)]
    except:
        scores = [5.0] * len(other_model_ids)

    return {"all_scores": {model_id: scores}}



def aggregate_scores(state: CouncilState):
    """Fan-in: Combine all rankings into aggregated scores"""

    responses = state.get("responses", {})
    all_scores = state.get("all_scores", {})

    # Initialize score tracking
    score_tracker = {i: [] for i in range(N_MODELS)}

    # Each model scored others
    for scorer_id, scores in all_scores.items():
        other_model_ids = [i for i in range(N_MODELS) if i != scorer_id]

        for idx, target_model_id in enumerate(other_model_ids):
            if idx < len(scores):
                score_tracker[target_model_id].append(scores[idx])

    # Compute averages
    aggregated_scores = {
        model_id: (
            sum(scores) / len(scores) if scores else 0.0
        )
        for model_id, scores in score_tracker.items()
    }

    max_score = max(aggregated_scores.values()) if aggregated_scores else 0
    meets_threshold = max_score >= state.get("threshold", 9.0)

    return {
        "query": state["query"],
        "responses": responses,
        "all_scores": all_scores,
        "aggregated_scores": aggregated_scores,
        "threshold": state.get("threshold", 9.0),
        "meets_threshold": meets_threshold,
        "final_response": "",
        "iteration": state.get("iteration", 1),
        "max_iterations": state.get("max_iterations", 10)
    }
    
    



def check_threshold(state: CouncilState):
    """Check threshold and decide: generate master response, reiterate, or end"""
    if state.get("meets_threshold"):
        return "generate_master_response"
    
    # Check if we can reiterate
    current_iteration = state.get("iteration", 1)
    max_iterations = state.get("max_iterations", 3)
    
    if current_iteration < max_iterations:
        return "reiterate"
    else:
        return "end"


def prepare_reiteration(state: CouncilState):
    """Prepare state for next iteration"""
    print(f"\n🔄 Threshold not met. Starting iteration {state.get('iteration', 1) + 1}/{state.get('max_iterations', 3)}...\n")
    
    return {
        "query": state["query"],
        "responses": {},  # Clear previous responses
        "all_scores": {},  # Clear previous scores
        "aggregated_scores": {},
        "threshold": state.get("threshold", 9.0),
        "meets_threshold": False,
        "final_response": "",
        "iteration": state.get("iteration", 1) + 1,
        "max_iterations": state.get("max_iterations", 3)
    }
    
    




def build_council_graph():
    """Build the AI Council workflow with fan-out/fan-in pattern and reiteration"""
    graph = StateGraph(CouncilState)
    
    # Add nodes (NOT the fan-out functions - those are conditional edges)
    graph.add_node("generate_response_node", generate_response_node)
    graph.add_node("aggregate_responses", aggregate_responses)
    graph.add_node("rank_response_node", rank_response_node)
    graph.add_node("aggregate_scores", aggregate_scores)
    graph.add_node("prepare_reiteration", prepare_reiteration)
    graph.add_node("generate_master_response", generate_master_response)
    
    # Start -> Fan-out to generate responses (using conditional edges with Send)
    graph.add_conditional_edges(
        START,
        fan_out_generate_responses,
        ["generate_response_node"]
    )
    
    # After all parallel generate_response_node tasks complete, aggregate
    graph.add_edge("generate_response_node", "aggregate_responses")
    
    # aggregate_responses -> Fan-out to rank responses (using conditional edges with Send)
    graph.add_conditional_edges(
        "aggregate_responses",
        fan_out_rank_responses,
        ["rank_response_node"]
    )
    
    # After all parallel rank_response_node tasks complete, aggregate scores
    graph.add_edge("rank_response_node", "aggregate_scores")
    
    # Check threshold and route conditionally
    graph.add_conditional_edges(
        "aggregate_scores",
        check_threshold,
        {
            "generate_master_response": "generate_master_response",
            "reiterate": "prepare_reiteration",
            "end": END
        }
    )
    
    # Reiteration loop: prepare_reiteration -> back to fan_out_generate
    graph.add_conditional_edges(
        "prepare_reiteration",
        fan_out_generate_responses,
        ["generate_response_node"]
    )
    
    # generate_master_response -> END
    graph.add_edge("generate_master_response", END)
    
    return graph.compile()


# Create the compiled graph
council_graph = build_council_graph()



# User interaction and query execution
def run_council(query: str, threshold: float = 9.0, max_iterations: int = 3):
    """Run the AI Council with user query"""
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print(f"THRESHOLD: {threshold}/10")
    print(f"MAX ITERATIONS: {max_iterations}")
    print(f"{'='*60}\n")
    
    initial_state = {
        "query": query,
        "responses": {},
        "all_scores": {},
        "aggregated_scores": {},
        "threshold": threshold,
        "meets_threshold": False,
        "final_response": "",
        "iteration": 1,
        "max_iterations": max_iterations
    }
    
    print("🔄 Starting AI Council workflow...\n")
    print(f"📡 Iteration 1/{max_iterations}: Generating responses from council members (Fan-out)...\n")
    
    result = council_graph.invoke(initial_state)
    
    print("\n" + "="*60)
    print("COUNCIL RESULTS")
    print("="*60)
    
    print(f"\n📊 Total Iterations: {result.get('iteration', 1)}/{max_iterations}")
    
    print("\n📝 Individual Responses:")
    for model_id in range(N_MODELS):
        response = result.get("responses", {}).get(model_id, "No response")
        print(f"\n  Model {model_id}:")
        print(f"  {response}")
    
    print("\n\n⭐ Aggregated Scores:")
    for model_id in range(N_MODELS):
        score = result.get("aggregated_scores", {}).get(model_id, 0)
        print(f"  Model {model_id}: {score:.2f}/10")
    
    print(f"\n🔍 Threshold Requirement: Score >= {result.get('threshold', 9.0)}")
    print(f"✅ Threshold Met: {result.get('meets_threshold', False)}")
    
    if result.get('meets_threshold'):
        print("\n" + "="*60)
        print("🏆 MASTER RESPONSE (Threshold Met)")
        print("="*60)
        print(f"\n{result.get('final_response', 'No response generated')}")
    else:
        best_score = max(result.get("aggregated_scores", {}).values()) if result.get("aggregated_scores") else 0
        print(f"\n❌ No master response generated.")
        print(f"   Best score ({best_score:.2f}/10) below threshold ({result.get('threshold', 9.0)}/10)")
        print(f"   Max iterations ({max_iterations}) reached.")
    
    print("\n" + "="*60 + "\n")
    return result


# Example usage - get user input
print("AI Council System Ready!")
print("="*60)
user_query = input("Enter your query: ")
user_threshold = float(input("Enter threshold score (0-10, default 9.0): ") or "9.0")
user_max_iterations = int(input("Enter max iterations (default 3): ") or "3")

result = run_council(user_query, user_threshold, user_max_iterations)



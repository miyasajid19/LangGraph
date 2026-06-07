import operator
import sys
from typing import TypedDict, List, Dict, Annotated, Any
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import PydanticOutputParser  
from langchain_core.utils.uuid import uuid7

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from psycopg_pool import ConnectionPool

console = Console()

def message_chunk_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text") or item.get("content") or "")
        return "".join(parts)

    return str(content) if content else ""

def load_history(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Get latest state snapshot
        snapshot = app.get_state(config)

        if not snapshot:
            console.print("[yellow]No history found for this thread.[/yellow]")
            return

        messages = snapshot.values.get("messages", [])

        console.print("\n[bold cyan]=== Conversation History ===[/bold cyan]\n")

        for msg in messages:
            if isinstance(msg, HumanMessage):
                console.print(f"[bold green]You:[/bold green] {msg.content}")
            elif isinstance(msg, AIMessage):
                console.print(f"[bold blue]AI:[/bold blue] {msg.content}")

        console.print("\n[bold cyan]===========================[/bold cyan]\n")

    except Exception as e:
        console.print(f"[red]Failed to load history:[/red] {e}")

# Connection String Config
URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"

# ==========================================
# 1. State Structures & Models Setup
# ==========================================
class State(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    loaded_memories: List[Dict[str, str]] 
    extracted_memories: List[Dict[str, str]] 
    response: str

class MemoryDetail(BaseModel):
    memory: str = Field(..., description="The specific value or detail of the memory.")

class MemoryExtraction(BaseModel):
    memories: List[Dict[str, MemoryDetail]] = Field(
        ...,
        description=(
            "A list of extracted user attributes. Each item is a dictionary where the key is the topic "
            "(e.g., 'age' or 'class') and the value is an object containing the 'memory' string. "
            "Example: [{'age': {'memory': '22 years old'}}, {'class': {'memory': 'B.Tech'}}]"
        )
    )
    
class MemoryAgentState(TypedDict):
    query: str
    past_memories: List[Dict[str, str]]
    extracted_memories: List[Dict[str, str]]

# Model initializations
model = ChatOllama(model="minimax-m3:cloud", reasoning=True)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize DB infrastructure with connection string management
pool = ConnectionPool(conninfo=URI, max_size=10, kwargs={"autocommit": True})

store = PostgresStore(
    conn=pool, 
    index={
        'embed': embeddings, 
        'dims': 384,
        'fields': ['text']
    }
)
saver = PostgresSaver(conn=pool)
NAMESPACE = ("memories", "sajid")

# ==========================================
# 2. Sub-Graph: Memory Extraction Node Workflows
# ==========================================
def extract_memories_node(state: MemoryAgentState) -> Dict[str, Any]:
    parser = PydanticOutputParser(pydantic_object=MemoryExtraction)
    
    prompt = PromptTemplate(
        template="""Extract new lifelong profile facts from this query.
Only extract information that is not already in the long-term memory store and that would be useful for future conversations.
If there is a memory that needs to be updated, then extract the new information. Return the result in the strict given format.

Past Memories :: {past_memories}
Current Query :: {query}

Format Instructions :: {format_instructions}
""",
        input_variables=["past_memories", "query"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    query = state["query"]
    past_memories = state["past_memories"]
    chain = prompt | model | parser
    result = chain.invoke({"query": query, "past_memories": past_memories})
    
    processed_memories = []
    if result and result.memories:
        for item in result.memories:
            
            for topic, detail_obj in item.items():
                clean_topic = topic.strip().lower()
                processed_memories.append({
                    "topic": clean_topic,
                    "text": detail_obj.memory.strip() 
                })
                
    return {"extracted_memories": processed_memories}






sub_builder = StateGraph(MemoryAgentState)
sub_builder.add_node("extract_facts", extract_memories_node)
sub_builder.add_edge(START, "extract_facts")
sub_builder.add_edge("extract_facts", END)
memory_extraction_graph = sub_builder.compile()


# ==========================================
# 3. Main Graph: Node Workflows
# ==========================================
def load_memories(state: State) -> Dict[str, Any]:
    if not state.get("messages"):
        return {"loaded_memories": []}
    last_user_message = state["messages"][-1].content
    search_results = store.search(NAMESPACE, query=str(last_user_message), limit=3)
    memories = [{"text": item.value.get("text", "")} for item in search_results]
    return {"loaded_memories": memories}

def conversation_agent(state: State) -> Dict[str, Any]:
    memory_context = "\n".join([f"- {m['text']}" for m in state.get("loaded_memories", [])])
    system_prompt = f"""You are a helpful assistant. 
Here are things you remember about the user:
{memory_context}

Respond naturally to the user's latest input using these memories where relevant."""
    
    messages_payload = [AIMessage(content=system_prompt)] + state["messages"]
    ai_response = model.invoke(messages_payload)
    
    return {
        "messages": [ai_response],
        "response": ai_response.content
    }

def call_memory_extraction_subgraph(state: State) -> Dict[str, Any]:
    """Adapter node to pass correct inputs into the nested Sub-Graph."""
    human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    last_query = human_messages[-1].content if human_messages else ""
    
    # Invoke the compiled sub-graph directly with its expected schema
    sub_graph_output = memory_extraction_graph.invoke({
        "query": last_query,
        "past_memories": state.get("loaded_memories", [])
    })
    return {"extracted_memories": sub_graph_output.get("extracted_memories", [])}

def update_memories(state: State):
    extracted = state.get("extracted_memories", [])
    for memory in extracted:
        # 1. Extract the raw key topic name (e.g., 'age', 'class')
        topic_key = memory.get("topic", "").strip().lower()
        
        # Fallback to a clean string if topic extraction missed something
        if not topic_key:
            topic_key = "general_profile"
            
        # 2. Use the meaningful topic name directly as your database Key item ID
        # This overwrites/updates the existing topic instead of creating random UUIDs
        store.put(NAMESPACE, topic_key, {"text": memory["text"]})
    return state


# ==========================================
# 4. Main Graph Composition & Compilation
# ==========================================
workflow = StateGraph(State)
workflow.add_node("load_memories", load_memories)
workflow.add_node("conversation_agent", conversation_agent)
workflow.add_node("memory_extractor", call_memory_extraction_subgraph)
workflow.add_node("update_memories", update_memories)

workflow.add_edge(START, "load_memories")
workflow.add_edge("load_memories", "conversation_agent")
workflow.add_edge("conversation_agent", "memory_extractor")
workflow.add_edge("memory_extractor", "update_memories")
workflow.add_edge("update_memories", END)

app = workflow.compile(checkpointer=saver)

# ==========================================
# 5. Chatbot Interface Loop
# ==========================================
def run_chatbot():
    store.setup()
    saver.setup()
    
    if input("Have thread ID to continue? (y/n): ").lower() != 'y':
        thread_id = str(uuid7())
    else:
        thread_id = input("Enter thread ID: ").strip()
        load_history(thread_id)
        
    config = {"configurable": {"thread_id": thread_id}}
    print(f"Active Session Config: {config}")
    
    console.print(Panel.fit(
        "[bold cyan]LangGraph Memory Chatbot Active![/bold cyan]\nType '[bold red]exit[/bold red]' or '[bold red]quit[/bold red]' to end the session.",
        border_style="cyan"
    ))

    while True:
        try:
            user_input = console.input("\n[bold green]You:[/bold green] ").strip()
            
            if user_input.lower() in ["exit", "quit"]:
                console.print("[bold yellow]Goodbye![/bold yellow]")
                break
                
            if not user_input:
                continue

            inputs = {"messages": [HumanMessage(content=user_input)]}
            
            console.print("[bold blue]AI:[/bold blue] ", end="")
            for message_chunk, metadata in app.stream(
                inputs,
                config=config,
                stream_mode="messages",
            ):
                if metadata.get("langgraph_node") != "conversation_agent":
                    continue

                chunk_text = message_chunk_to_text(message_chunk.content)
                if chunk_text:
                    console.print(chunk_text, end="", markup=False, highlight=False)

            console.print()

            snapshot = app.get_state(config)
            updated_state = snapshot.values if snapshot else {}

            if updated_state.get("extracted_memories"):
                console.print(f"[dim italic yellow]💡 System logged new memory: {updated_state['extracted_memories']}[/dim italic yellow]")
                
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Session interrupted. Goodbye![/bold yellow]")
            sys.exit(0)

if __name__ == "__main__":
    run_chatbot()

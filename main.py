import os
from typing import List
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph
from zep_cloud import Zep

load_dotenv()

# State schema - extensible with placeholders
class AgentState(TypedDict):
    human_message: str
    validation_status: str  # "valid", "invalid" - placeholder for future validation node
    facts: List[str]
    test_conditions: List[str]  # Or dicts for structure
    final_output: str
    errors: List[str]  # Accumulate errors

# Initialize Zep client
zep = Zep(api_key=os.getenv("ZEP_API_KEY"))

# LLM for generation
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2025-01-01-preview",
    temperature=0,
)

# Tester Node: Validates input and searches Zep for facts
def tester_node(state: AgentState) -> AgentState:
    human_message = state["human_message"]
    if not human_message:
        raise ValueError("human_message is required")

    # Basic validation
    validation_status = "valid"
    errors = state.get("errors", [])
    if len(human_message) < 10:
        errors.append("Input too short (minimum 10 characters)")
        validation_status = "invalid"
    if not any(keyword in human_message.lower() for keyword in ["as a", "i want", "story"]):
        errors.append("Input does not appear to be an agile story (missing keywords like 'as a', 'I want')")
        validation_status = "warning"  # Allow but warn

    # Zep search
    facts = []
    try:
        response = zep.graph.search(
            query=human_message,
            graph_id="pet-store-knowledge",
            scope="edges",
            limit=50,  # Increased to max for better retrieval
            min_fact_rating=0.3,  # Filter low-quality facts
            reranker="rrf",  # Use RRF for better ranking
        )
        edges = response.edges or []
        # Post-filter by score to ensure relevance
        edges = [e for e in edges if getattr(e, 'score', 0) > 0.5]
        facts = [edge.fact for edge in edges if hasattr(edge, 'fact')]
    except Exception as e:
        errors.append(f"Zep search failed: {str(e)}")

    return {
        "human_message": state.get("human_message", ""),
        "validation_status": validation_status,
        "facts": facts,
        "test_conditions": state.get("test_conditions", []),
        "final_output": state.get("final_output", ""),
        "errors": errors,
    }

# LLM Node: Generates test conditions based on story and facts
def llm_node(state: AgentState) -> AgentState:
    human_message = state["human_message"]
    facts = state.get("facts", [])
    errors = state.get("errors", [])

    # Prompt for test generation
    prompt = f"""
    Generate comprehensive test conditions/scenarios for the following agile user story, carefully analysing provided related facts from product knowledge.

    **User Story:**
    {human_message}

    **Related Facts:**
    {chr(10).join(f"- {fact}" for fact in facts) if facts else "None found"}

    Provide 3-5 detailed test scenarios in the format:
    - Test 1: [Description of test condition]
    - Test 2: [Description...]
    etc.

    Focus on functional, edge cases, and negative tests.
    """

    test_conditions = []
    final_output = ""
    try:
        response = llm.invoke(prompt)
        # Parse response into list (simple split by lines)
        lines = response.content.strip().split('\n')
        print(f"LLM response lines: {lines}")
        test_conditions = []
        for line in lines:
            line = line.strip()
            if any(line.startswith(f"{i}. ") for i in range(1, 8)) and "**Test " in line:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    desc = parts[1].strip().strip("**").strip()
                    test_conditions.append(desc)
        print(f"Parsed test_conditions: {test_conditions}")

        # Format final output
        final_output = f"""
Validation Status: {state.get('validation_status', 'unknown')}

User Story:
{human_message}

Related Facts:
{chr(10).join(facts) if facts else 'No facts retrieved'}

Generated Test Conditions:
{response.content.strip()}

Errors:
{chr(10).join(errors) if errors else 'None'}
        """.strip()

    except Exception as e:
        errors.append(f"LLM generation failed: {str(e)}")
        final_output = f"Error: Failed to generate tests. {chr(10).join(errors)}"

    return {
        "human_message": state["human_message"],
        "validation_status": state.get("validation_status", "unknown"),
        "facts": state.get("facts", []),
        "test_conditions": test_conditions,
        "final_output": final_output,
        "errors": errors,
    }
# Build the LangGraph
graph = StateGraph(AgentState)
graph.add_node("tester", tester_node)
graph.add_node("llm", llm_node)
graph.add_edge("tester", "llm")
graph.set_entry_point("tester")

# Compile the graph
compiled_graph = graph.compile()

# Export for LangGraph Studio
__all__ = ["compiled_graph"]
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
from em_research_agentic.utils.state import AgentState
from em_research_agentic.utils.nodes import plan_node, generation_node, reflection_node, research_plan_node, research_critique_node, should_continue


class GraphConfig(TypedDict):
    model_name: Literal["anthropic", "openai"]
    max_results_tavily: int


builder = StateGraph(AgentState, config_schema=GraphConfig)
builder.add_node("planner", plan_node)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_node("research_plan", research_plan_node)
builder.add_node("research_critique", research_critique_node)
builder.set_entry_point("planner")
builder.add_conditional_edges(
    "generate",
    should_continue,
    {END: END, "reflect": "reflect"}
)
builder.add_edge("planner", "research_plan")
builder.add_edge("research_plan", "generate")

builder.add_edge("reflect", "research_critique")
builder.add_edge("research_critique", "generate")

graph = builder.compile()

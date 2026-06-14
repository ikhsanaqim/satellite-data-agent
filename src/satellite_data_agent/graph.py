"""LangGraph workflow definition with conditional routing."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from satellite_data_agent.agents import (
    analyst_node,
    build_report_generator_node,
    build_supervisor_node,
    data_fetcher_node,
    rag_retriever_node,
)
from satellite_data_agent.llm import build_llm
from satellite_data_agent.state import AgentState


def route_from_supervisor(state: AgentState) -> str:
    """Read the Supervisor's routing decision from state."""
    return state.get("next_action", "FINISH")


def build_graph():
    """Build and compile the LangGraph state graph.

    Graph topology:
        START → supervisor ←→ [data_fetcher, rag_retriever, analyst, report_generator]
        supervisor → FINISH → END

    The supervisor is the central hub. After each worker node completes,
    control returns to the supervisor which decides the next action.
    """
    llm = build_llm()

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("supervisor", build_supervisor_node(llm))
    graph.add_node("data_fetcher", data_fetcher_node)
    graph.add_node("rag_retriever", rag_retriever_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("report_generator", build_report_generator_node(llm))

    # Entry edge
    graph.add_edge(START, "supervisor")

    # Supervisor conditional routing
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "data_fetcher": "data_fetcher",
            "rag_retriever": "rag_retriever",
            "analyst": "analyst",
            "report_generator": "report_generator",
            "FINISH": END,
        },
    )

    # All workers return to supervisor
    graph.add_edge("data_fetcher", "supervisor")
    graph.add_edge("rag_retriever", "supervisor")
    graph.add_edge("analyst", "supervisor")
    graph.add_edge("report_generator", "supervisor")

    return graph.compile()

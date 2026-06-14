"""RAG Retriever agent — fetches relevant context from ChromaDB knowledge base."""

from __future__ import annotations

from satellite_data_agent.rag.retriever import retrieve_context
from satellite_data_agent.state import AgentState


def rag_retriever_node(state: AgentState) -> dict:
    """Retrieve domain knowledge relevant to the user's query.

    Uses ChromaDB semantic search against indexed satellite operations documents.
    """
    query = state.get("query", "")

    # Also include analysis context if available, for more targeted retrieval
    analysis = state.get("analysis_result", {})
    alerts = analysis.get("alerts", [])
    if alerts:
        query = f"{query}. Relevant anomalies: {'; '.join(alerts[:3])}"

    context = retrieve_context(query)

    if context:
        return {
            "rag_context": context,
            "agent_history": [f"RAG Retriever: found relevant context ({len(context)} chars)."],
        }
    else:
        return {
            "rag_context": "",
            "agent_history": ["RAG Retriever: no relevant context found in knowledge base."],
        }

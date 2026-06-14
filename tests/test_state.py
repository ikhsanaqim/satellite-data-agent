"""Tests for AgentState schema."""

from satellite_data_agent.state import AgentState


def test_agent_state_minimal():
    """AgentState can be created with minimal fields (total=False)."""
    state: AgentState = {"query": "test"}
    assert state["query"] == "test"


def test_agent_state_all_fields():
    """AgentState accepts all defined fields."""
    state: AgentState = {
        "query": "test",
        "agent_history": ["step 1"],
        "fetched_data": {"records": []},
        "rag_context": "some context",
        "analysis_result": {"summary": "ok"},
        "final_report": "# Report",
        "next_action": "FINISH",
        "iteration_count": 1,
    }
    assert state["iteration_count"] == 1

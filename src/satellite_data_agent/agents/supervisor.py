"""Supervisor agent — LLM-based conditional routing with structured output."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from satellite_data_agent.llm import SimpleLLM, MockLLM
from satellite_data_agent.state import AgentState

MAX_ITERATIONS = 10

NextAction = Literal["data_fetcher", "rag_retriever", "analyst", "report_generator", "FINISH"]


class SupervisorDecision(BaseModel):
    """Structured output schema for supervisor routing."""
    next_action: NextAction = Field(
        description="The next agent to invoke, or FINISH if all work is done."
    )
    reasoning: str = Field(
        description="Brief explanation of why this agent was chosen."
    )


SUPERVISOR_SYSTEM_PROMPT = """\
You are the Supervisor of a satellite data analysis multi-agent system.

Your job is to decide which agent should run next based on the current state.

Available agents:
- data_fetcher: Fetches satellite/IoT telemetry data from APIs or local files. Call this FIRST if no data has been fetched yet.
- rag_retriever: Retrieves relevant technical documentation from a knowledge base. Call this when the analyst needs domain context (e.g., unfamiliar anomaly patterns, satellite-specific thresholds).
- analyst: Analyzes fetched data for anomalies and patterns. Call this AFTER data is fetched.
- report_generator: Generates a structured Markdown report. Call this AFTER analysis is complete.
- FINISH: All work is done, no more agents needed.

Rules:
1. Always fetch data before analyzing.
2. RAG retrieval is optional — only use it if the query mentions specific standards, documentation, or unfamiliar patterns.
3. Always generate a report before finishing.
4. Never call the same agent twice unless its previous run produced an error.

Current state will be provided as context. Decide the next action.
"""


def build_supervisor_node(llm: SimpleLLM):
    """Factory: returns a supervisor_node function with the LLM injected."""

    def supervisor_node(state: AgentState) -> dict:
        iteration = state.get("iteration_count", 0) + 1

        # Safety: prevent infinite loops
        if iteration >= MAX_ITERATIONS:
            return {
                "next_action": "FINISH",
                "iteration_count": iteration,
                "agent_history": [f"Supervisor: iteration limit ({MAX_ITERATIONS}) reached. Forcing FINISH."],
            }

        # If MockLLM, use deterministic fallback
        if isinstance(llm, MockLLM):
            return _deterministic_routing(state, iteration)

        # Build context message for LLM
        history = state.get("agent_history", [])
        context_parts = [
            f"User query: {state.get('query', 'No query provided')}",
            f"Iteration: {iteration}",
            f"Agent history so far: {history}",
            f"Has fetched_data: {state.get('fetched_data') is not None}",
            f"Has rag_context: {state.get('rag_context') is not None}",
            f"Has analysis_result: {state.get('analysis_result') is not None}",
            f"Has final_report: {state.get('final_report') is not None}",
        ]
        context_message = "\n".join(context_parts)

        try:
            # Use structured output if LLM supports it
            structured_llm = llm.with_structured_output(SupervisorDecision)
            decision: SupervisorDecision = structured_llm.invoke(
                [
                    {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
                    {"role": "user", "content": context_message},
                ]
            )
            return {
                "next_action": decision.next_action,
                "iteration_count": iteration,
                "agent_history": [f"Supervisor: chose '{decision.next_action}' — {decision.reasoning}"],
            }
        except Exception:
            # Fallback if structured output fails
            return _deterministic_routing(state, iteration)

    return supervisor_node


def _deterministic_routing(state: AgentState, iteration: int) -> dict:
    """Fallback routing when LLM is not available or fails."""
    history_text = " ".join(state.get("agent_history", []))

    if state.get("fetched_data") is None:
        action = "data_fetcher"
        reason = "No data fetched yet."
    elif state.get("analysis_result") is None:
        action = "analyst"
        reason = "Data fetched, analysis not done yet."
    elif state.get("final_report") is None:
        action = "report_generator"
        reason = "Analysis done, report not generated yet."
    else:
        action = "FINISH"
        reason = "All steps completed."

    return {
        "next_action": action,
        "iteration_count": iteration,
        "agent_history": [f"Supervisor (deterministic): chose '{action}' — {reason}"],
    }

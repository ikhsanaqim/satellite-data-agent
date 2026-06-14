"""Integration test for the full LangGraph pipeline."""

from satellite_data_agent.graph import build_graph


def test_full_pipeline_with_mock_llm(tmp_path, monkeypatch):
    """Full pipeline runs to completion with MockLLM and local data."""
    output_path = tmp_path / "report.md"
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("REPORT_OUTPUT_PATH", str(output_path))

    app = build_graph()
    final_state = app.invoke(
        {
            "query": "Analyze satellite and IoT telemetry health.",
            "iteration_count": 0,
        }
    )

    assert final_state["next_action"] == "FINISH"
    assert final_state.get("fetched_data") is not None
    assert final_state.get("analysis_result") is not None
    assert final_state.get("final_report") is not None
    assert output_path.exists()
    assert "Satellite" in output_path.read_text(encoding="utf-8")

    # Agent history should have accumulated entries
    history = final_state.get("agent_history", [])
    assert len(history) >= 4  # at least: supervisor decisions + 3 workers

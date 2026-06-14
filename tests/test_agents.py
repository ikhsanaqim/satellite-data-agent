"""Tests for individual agent nodes."""

import json
from pathlib import Path

from satellite_data_agent.agents.analyst import analyst_node
from satellite_data_agent.agents.data_fetcher import data_fetcher_node
from satellite_data_agent.agents.supervisor import _deterministic_routing


def test_deterministic_routing_starts_with_data_fetcher():
    state = {"query": "test", "agent_history": []}
    result = _deterministic_routing(state, iteration=1)
    assert result["next_action"] == "data_fetcher"


def test_deterministic_routing_analyst_after_data():
    state = {"query": "test", "agent_history": [], "fetched_data": {"records": []}}
    result = _deterministic_routing(state, iteration=2)
    assert result["next_action"] == "analyst"


def test_deterministic_routing_report_after_analysis():
    state = {
        "query": "test",
        "agent_history": [],
        "fetched_data": {"records": []},
        "analysis_result": {"summary": "ok"},
    }
    result = _deterministic_routing(state, iteration=3)
    assert result["next_action"] == "report_generator"


def test_deterministic_routing_finish():
    state = {
        "query": "test",
        "agent_history": [],
        "fetched_data": {"records": []},
        "analysis_result": {"summary": "ok"},
        "final_report": "# Report",
    }
    result = _deterministic_routing(state, iteration=4)
    assert result["next_action"] == "FINISH"


def test_analyst_empty_records():
    state = {"fetched_data": {"records": []}}
    result = analyst_node(state)
    assert result["analysis_result"]["summary"] == "No records available for analysis."


def test_analyst_with_records():
    records = [
        {
            "site_id": "SITE-1", "device_id": "DEV-1",
            "latency_ms": 900, "packet_loss_pct": 5.0,
            "snr_db": 6.0, "rssi_dbm": -92, "status": "degraded",
        },
        {
            "site_id": "SITE-2", "device_id": "DEV-2",
            "latency_ms": 500, "packet_loss_pct": 0.5,
            "snr_db": 14.0, "rssi_dbm": -70, "status": "ok",
        },
    ]
    state = {"fetched_data": {"records": records}}
    result = analyst_node(state)
    analysis = result["analysis_result"]
    assert analysis["metrics"]["high_latency_count"] == 1
    assert analysis["metrics"]["non_ok_status_count"] == 1
    assert len(result["agent_history"]) == 1

"""Data Fetcher agent — reads telemetry from local files or NASA CMR API."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from satellite_data_agent.state import AgentState
from satellite_data_agent.tools.nasa_api import fetch_cmr_granules


def data_fetcher_node(state: AgentState) -> dict:
    """Fetch satellite/IoT telemetry data.

    Routing logic:
    - If query mentions NASA/CMR/granule/coverage → call NASA CMR API.
    - Otherwise → read local telemetry file.
    """
    query = state.get("query", "").lower()
    use_cmr = any(kw in query for kw in ["cmr", "nasa", "granule", "coverage", "satellite coverage"])

    if use_cmr:
        try:
            data = fetch_cmr_granules()
            source = "NASA CMR API"
        except Exception as exc:
            return {
                "agent_history": [f"Data Fetcher: NASA CMR API error — {exc}. Falling back to local data."],
                "fetched_data": _load_local_telemetry(),
            }
    else:
        data = _load_local_telemetry()
        source = "local telemetry file"

    record_count = data.get("record_count", len(data.get("records", [])))
    return {
        "fetched_data": data,
        "agent_history": [f"Data Fetcher: loaded {record_count} records from {source}."],
    }


def _load_local_telemetry() -> dict:
    """Load telemetry from local JSON or CSV file."""
    input_path = Path(os.getenv("TELEMETRY_INPUT_PATH", "data/telemetry/sample.json"))

    if not input_path.exists():
        # Fallback ke CSV lama
        input_path = Path("data/sample_telemetry.csv")

    if not input_path.exists():
        return {"records": [], "record_count": 0, "error": f"File not found: {input_path}"}

    if input_path.suffix.lower() == ".json":
        with open(input_path, encoding="utf-8") as f:
            records = json.load(f)
    else:
        frame = pd.read_csv(input_path)
        records = frame.to_dict(orient="records")

    # Build summary
    sites = sorted(set(r.get("site_id", "") for r in records))
    devices = sorted(set(r.get("device_id", "") for r in records))
    timestamps = [r.get("timestamp", "") for r in records]

    return {
        "records": records,
        "record_count": len(records),
        "sites": sites,
        "devices": devices,
        "time_range": {"start": min(timestamps), "end": max(timestamps)} if timestamps else {},
    }

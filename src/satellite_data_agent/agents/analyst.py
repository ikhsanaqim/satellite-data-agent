"""Analyst agent — anomaly detection and insight extraction."""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Optional

from pydantic import BaseModel, Field

from satellite_data_agent.state import AgentState


class AnalysisMetrics(BaseModel):
    avg_latency_ms: float
    avg_packet_loss_pct: float
    high_latency_count: int
    packet_loss_alert_count: int
    low_signal_count: int
    non_ok_status_count: int


class AnalysisResult(BaseModel):
    summary: str
    metrics: AnalysisMetrics
    top_affected_sites: list[tuple[str, int]] = Field(default_factory=list)
    alerts: list[str] = Field(default_factory=list)
    confidence: str = "medium"  # low, medium, high


def analyst_node(state: AgentState) -> dict:
    """Analyze telemetry records for anomalies and patterns."""
    fetched = state.get("fetched_data", {})
    records = fetched.get("records", [])

    if not records:
        result = AnalysisResult(
            summary="No records available for analysis.",
            metrics=AnalysisMetrics(
                avg_latency_ms=0, avg_packet_loss_pct=0,
                high_latency_count=0, packet_loss_alert_count=0,
                low_signal_count=0, non_ok_status_count=0,
            ),
        )
        return {
            "analysis_result": result.model_dump(),
            "agent_history": ["Analyst: no records to analyze."],
        }

    # Thresholds
    LATENCY_THRESHOLD_MS = 800
    PACKET_LOSS_THRESHOLD_PCT = 3.0
    SNR_LOW_THRESHOLD_DB = 8.0
    RSSI_LOW_THRESHOLD_DBM = -88

    high_latency = [r for r in records if float(r.get("latency_ms", 0)) >= LATENCY_THRESHOLD_MS]
    packet_loss = [r for r in records if float(r.get("packet_loss_pct", 0)) >= PACKET_LOSS_THRESHOLD_PCT]
    low_signal = [
        r for r in records
        if float(r.get("snr_db", 99)) < SNR_LOW_THRESHOLD_DB
        or float(r.get("rssi_dbm", 0)) < RSSI_LOW_THRESHOLD_DBM
    ]
    down_devices = [r for r in records if str(r.get("status", "ok")).lower() != "ok"]

    site_alerts: Counter = Counter()
    for row in high_latency + packet_loss + low_signal + down_devices:
        site_alerts[row.get("site_id", "unknown")] += 1

    latency_values = [float(r.get("latency_ms", 0)) for r in records]
    loss_values = [float(r.get("packet_loss_pct", 0)) for r in records]

    # Determine confidence
    total_anomalies = len(high_latency) + len(packet_loss) + len(low_signal) + len(down_devices)
    if total_anomalies == 0:
        confidence = "high"
    elif total_anomalies <= len(records) * 0.3:
        confidence = "medium"
    else:
        confidence = "low"

    metrics = AnalysisMetrics(
        avg_latency_ms=round(mean(latency_values), 2),
        avg_packet_loss_pct=round(mean(loss_values), 2),
        high_latency_count=len(high_latency),
        packet_loss_alert_count=len(packet_loss),
        low_signal_count=len(low_signal),
        non_ok_status_count=len(down_devices),
    )

    result = AnalysisResult(
        summary=(
            f"Average latency is {metrics.avg_latency_ms} ms and average packet "
            f"loss is {metrics.avg_packet_loss_pct}% across {len(records)} records."
        ),
        metrics=metrics,
        top_affected_sites=site_alerts.most_common(3),
        alerts=[
            "Latency >= 800 ms indicates possible congestion or backhaul degradation.",
            "Packet loss >= 3% can disrupt telemetry delivery and remote operations.",
            "Low SNR/RSSI suggests weather impact, antenna alignment, or terminal issue.",
        ],
        confidence=confidence,
    )

    return {
        "analysis_result": result.model_dump(),
        "agent_history": [
            f"Analyst: analyzed {len(records)} records. "
            f"Found {total_anomalies} anomaly signals. Confidence: {confidence}."
        ],
    }

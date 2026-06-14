"""Report Generator agent — produces structured Markdown incident report."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from satellite_data_agent.llm import SimpleLLM
from satellite_data_agent.state import AgentState


class IncidentReport(BaseModel):
    title: str = "Satellite / IoT Telemetry Report"
    executive_summary: str
    scene_description: str = ""
    anomaly_list: list[str] = Field(default_factory=list)
    confidence_score: str = "medium"
    recommended_actions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


def build_report_generator_node(llm: SimpleLLM):
    """Factory: returns a report_generator_node function with the LLM injected."""

    def report_generator_node(state: AgentState) -> dict:
        fetched = state.get("fetched_data", {})
        analysis = state.get("analysis_result", {})
        rag_context = state.get("rag_context")

        # Build prompt
        parts = [
            "Write a concise executive summary for satellite/IoT telemetry analysis.",
            f"Data summary: {_data_summary_text(fetched)}.",
            f"Analysis: {analysis}.",
        ]
        if rag_context:
            parts.append(f"Relevant technical context from knowledge base: {rag_context}")

        prompt = " ".join(parts)
        llm_response = llm.invoke(prompt)
        if hasattr(llm_response, "content"):
            llm_response = llm_response.content

        # Build structured report
        report_obj = IncidentReport(
            executive_summary=str(llm_response),
            scene_description=_data_summary_text(fetched),
            anomaly_list=analysis.get("alerts", []),
            confidence_score=analysis.get("confidence", "medium"),
            recommended_actions=[
                "Check terminals with repeated low SNR/RSSI before peak traffic windows.",
                "Compare gateway load against high-latency periods.",
                "Prioritize field investigation when packet loss, weak signal, and non-ok status appear together.",
                "Cross-validate findings with additional data sources if confidence is below 'high'.",
            ],
            limitations=[
                "Analysis based on synthetic telemetry data — not validated against production systems.",
                "LLM-generated summary may hallucinate details not present in source data.",
            ],
        )

        report_md = _render_markdown(report_obj, fetched, analysis)

        # Write to file
        output_path = Path(os.getenv("REPORT_OUTPUT_PATH", "reports/output.md"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_md, encoding="utf-8")

        return {
            "final_report": report_md,
            "agent_history": [f"Report Generator: wrote report to {output_path}."],
        }

    return report_generator_node


def _data_summary_text(fetched: dict) -> str:
    """Format fetched data into a human-readable summary string."""
    n = fetched.get("record_count", 0)
    sites = ", ".join(fetched.get("sites", []))
    tr = fetched.get("time_range", {})
    return f"{n} records from sites [{sites}], time range {tr.get('start', '?')} to {tr.get('end', '?')}"


def _render_markdown(report: IncidentReport, fetched: dict, analysis: dict) -> str:
    """Render the IncidentReport Pydantic model as a Markdown string."""
    metrics = analysis.get("metrics", {})
    top_sites = analysis.get("top_affected_sites", [])
    top_site_lines = "\n".join(
        f"- `{site}`: {count} alert signals" for site, count in top_sites
    ) or "- No alert-heavy site detected."

    anomaly_lines = "\n".join(f"- {a}" for a in report.anomaly_list) or "- None detected."
    action_lines = "\n".join(f"- {a}" for a in report.recommended_actions)
    limitation_lines = "\n".join(f"- {l}" for l in report.limitations)

    return f"""# {report.title}

## Executive Summary

{report.executive_summary}

## Scene Description

{report.scene_description}

## Dataset

- Records: {fetched.get("record_count", 0)}
- Sites: {", ".join(fetched.get("sites", []))}
- Time range: {fetched.get("time_range", {}).get("start")} to {fetched.get("time_range", {}).get("end")}

## Key Metrics

- Average latency: {metrics.get("avg_latency_ms", 0)} ms
- Average packet loss: {metrics.get("avg_packet_loss_pct", 0)}%
- High latency records: {metrics.get("high_latency_count", 0)}
- Packet loss alerts: {metrics.get("packet_loss_alert_count", 0)}
- Low signal records: {metrics.get("low_signal_count", 0)}
- Non-OK status records: {metrics.get("non_ok_status_count", 0)}

## Anomalies Detected

{anomaly_lines}

**Confidence:** {report.confidence_score}

## Priority Sites

{top_site_lines}

## Recommended Actions

{action_lines}

## Limitations

{limitation_lines}
"""

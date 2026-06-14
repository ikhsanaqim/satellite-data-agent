# Implementation Plan: satellite-data-agent Alignment

**Objective:** Menyelaraskan codebase aktual dengan spesifikasi di `plan-satellite-data-agent.md`.

**Target audience:** AI coding assistant yang akan mengeksekusi plan ini secara mandiri.

**Repository:** `d:\GITHUB\satellite-data-agent`

**Python version:** >=3.10 (sudah di `pyproject.toml`)

**Packaging:** setuptools, sumber package di `src/`, editable install via `pip install -e .`

**Existing test runner:** pytest (di `requirements-dev.txt`)

---

## Current State Summary

Codebase saat ini memiliki LangGraph cyclic graph yang berjalan dengan 4 node: `supervisor`, `data_fetcher`, `analyst`, `report_generator`. Supervisor routing **deterministik** (if/else). Data dibaca dari file CSV lokal. LLM support ada untuk Groq, OpenRouter, dan MockLLM fallback. Semua agent ada di satu file `nodes.py`. Tidak ada RAG pipeline, tidak ada NASA CMR API, tidak ada Streamlit UI, tidak ada Pydantic validation.

### Files yang ada saat ini:
```
satellite-data-agent/
├── src/satellite_data_agent/
│   ├── __init__.py          (package init, version 0.1.0)
│   ├── cli.py               (argparse CLI, invoke graph)
│   ├── graph.py              (StateGraph build + conditional routing)
│   ├── llm.py                (SimpleLLM protocol, MockLLM, build_llm() factory)
│   ├── nodes.py              (supervisor_node, data_fetcher_node, analyst_node, build_report_generator_node)
│   └── state.py              (TelemetryState TypedDict)
├── data/sample_telemetry.csv  (13 rows synthetic PSN telemetry)
├── tests/test_graph_smoke.py  (1 smoke test)
├── scripts/render_graph.py    (LangGraph diagram renderer)
├── docs/langgraph_state_graph.mmd
├── reports/sample-report.md
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .gitignore
├── README.md
└── plan-satellite-data-agent.md
```

---

## Phase 1: Restructure Project Layout + State Schema

### Tujuan
Ubah struktur folder agar sesuai plan. Update state schema agar menggunakan `Annotated` reducer dan field yang benar.

### 1.1 Buat folder structure baru

Buat folder-folder berikut (buat `__init__.py` kosong di setiap folder Python):

```
src/satellite_data_agent/
├── __init__.py              (KEEP, tidak diubah)
├── state.py                 (REWRITE — lihat 1.2)
├── graph.py                 (REWRITE — lihat Phase 5)
├── llm.py                   (KEEP, tidak diubah)
├── cli.py                   (MODIFY — lihat Phase 6)
├── agents/                  (NEW folder)
│   ├── __init__.py          (NEW — export semua agent functions)
│   ├── supervisor.py        (NEW — pindah + rewrite dari nodes.py)
│   ├── data_fetcher.py      (NEW — pindah dari nodes.py)
│   ├── rag_retriever.py     (NEW — lihat Phase 3)
│   ├── analyst.py           (NEW — pindah dari nodes.py)
│   └── report_gen.py        (NEW — pindah dari nodes.py)
├── tools/                   (NEW folder)
│   ├── __init__.py          (NEW — empty)
│   └── nasa_api.py          (NEW — lihat Phase 4)
└── rag/                     (NEW folder)
    ├── __init__.py          (NEW — empty)
    ├── indexer.py           (NEW — lihat Phase 3)
    └── retriever.py         (NEW — lihat Phase 3)
```

Juga buat:
```
data/
├── telemetry/               (NEW folder)
│   └── sample.json          (NEW — konversi dari sample_telemetry.csv ke JSON)
└── knowledge_base/          (NEW folder)
    └── satellite_ops_guide.md  (NEW — lihat Phase 3)
```

**PENTING:** Setelah memindahkan semua kode, HAPUS file `src/satellite_data_agent/nodes.py`. File ini tidak boleh ada di hasil akhir.

### 1.2 Rewrite `state.py`

Ganti seluruh isi `state.py` dengan:

```python
"""Shared state schema for the satellite data agent graph."""

from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict


class AgentState(TypedDict, total=False):
    """State yang dibagi antar semua node di LangGraph.

    Fields:
        query: Pertanyaan atau instruksi analisis dari user.
        agent_history: Log akumulatif dari setiap agent. Menggunakan
            operator.add sehingga setiap node APPEND ke list, bukan replace.
        fetched_data: Data yang diambil oleh Data Fetcher (NASA CMR atau file lokal).
        rag_context: Konteks dokumen yang di-retrieve oleh RAG Retriever.
        analysis_result: Hasil analisis dari Analyst agent.
        final_report: Laporan Markdown final dari Report Generator.
        next_action: Keputusan routing Supervisor — salah satu dari:
            "data_fetcher", "rag_retriever", "analyst", "report_generator", "FINISH".
        iteration_count: Counter untuk safety limit agar graph tidak infinite loop.
            Increment setiap kali Supervisor dipanggil. Max 10 iterasi.
    """

    query: str
    agent_history: Annotated[list[str], operator.add]
    fetched_data: Optional[dict]
    rag_context: Optional[str]
    analysis_result: Optional[dict]
    final_report: Optional[str]
    next_action: str
    iteration_count: int
```

**Catatan teknis tentang `Annotated[list[str], operator.add]`:**
- Ketika sebuah node return `{"agent_history": ["Data Fetcher: loaded 13 records"]}`, LangGraph TIDAK akan menimpa list lama. Ia akan memanggil `operator.add(existing_list, new_list)` sehingga entry baru di-append.
- Semua field LAIN (seperti `fetched_data`, `next_action`) akan di-REPLACE saat node menulis ke field itu — ini perilaku default LangGraph.
- `total=False` tetap dipakai karena di awal invoke, tidak semua field harus diisi.

### 1.3 Buat `data/telemetry/sample.json`

Konversi `data/sample_telemetry.csv` ke JSON array. Gunakan script Python one-liner atau tulis manual. Format output:

```json
[
  {
    "timestamp": "2026-06-12T08:00:00+07:00",
    "site_id": "PSN-JKT-GW01",
    "device_id": "GW-JKT-01",
    "terminal_type": "gateway",
    "latitude": -6.2088,
    "longitude": 106.8456,
    "latency_ms": 612,
    "packet_loss_pct": 0.8,
    "snr_db": 12.4,
    "rssi_dbm": -74,
    "throughput_mbps": 82.5,
    "status": "ok"
  }
  // ... (semua 13 rows dari CSV)
]
```

**JANGAN hapus** `data/sample_telemetry.csv` — tetap simpan untuk backward compatibility. Tapi default input di CLI harus diubah ke `data/telemetry/sample.json`.

---

## Phase 2: Pindahkan + Refactor Agents ke Folder `agents/`

### 2.1 `agents/data_fetcher.py`

Pindahkan logika dari `nodes.py` `data_fetcher_node`. Modifikasi:

1. Ubah signature agar menerima `AgentState` (bukan `TelemetryState`).
2. Field `input_path` TIDAK ADA di `AgentState` baru. Data Fetcher harus memutuskan sumber data berdasarkan `query`:
   - Jika `query` mengandung kata "CMR", "NASA", "satellite coverage", atau "granule" → panggil NASA CMR API (Phase 4).
   - Jika TIDAK → coba baca dari file default `data/telemetry/sample.json`.
   - Untuk menentukan file path, tambahkan optional field atau hardcode default. Keputusan: **hardcode default path** `data/telemetry/sample.json` dan biarkan bisa di-override via environment variable `TELEMETRY_INPUT_PATH`.
3. Return harus meng-update `agent_history` (append, bukan replace) dan `fetched_data`.

```python
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
```

### 2.2 `agents/analyst.py`

Pindahkan logika `analyst_node` dari `nodes.py`. Modifikasi:

1. Gunakan `AgentState`, baca dari `state["fetched_data"]["records"]`.
2. Tambahkan **Pydantic output validation**. Definisikan model:

```python
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
```

### 2.3 `agents/report_gen.py`

Pindahkan `build_report_generator_node` + `_format_report` dari `nodes.py`. Modifikasi:

1. Gunakan `AgentState`. Baca dari `state["analysis_result"]`, `state["fetched_data"]`, `state.get("rag_context")`.
2. Tambahkan Pydantic model untuk report structure.
3. Masukkan RAG context ke dalam prompt jika tersedia.
4. Tulis report ke file output (path dari env var atau default `reports/output.md`).

```python
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
                "Prioritize field investigation when packet loss, weak signal, and non-OK status appear together.",
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
```

### 2.4 `agents/supervisor.py` — LLM-based Routing

**Ini perubahan paling signifikan.** Supervisor harus menggunakan LLM dengan structured output untuk memutuskan agent berikutnya, bukan if/else.

**Desain keputusan:**
- Gunakan LangChain `with_structured_output` untuk memaksa LLM mengembalikan JSON dengan field `next_action` dan `reasoning`.
- Jika LLM tidak tersedia (MockLLM), fallback ke logika deterministik (sama seperti sekarang) agar demo offline tetap jalan.
- `iteration_count` di-increment setiap kali supervisor dipanggil. Jika `iteration_count >= 10`, paksa `FINISH`.

```python
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
```

**Catatan:** `llm.with_structured_output(SupervisorDecision)` adalah fitur LangChain yang sudah ada di `ChatOpenAI` dan `ChatGroq`. Untuk `MockLLM`, kita skip dan pakai deterministic. Method `with_structured_output` tidak perlu ditambahkan ke `MockLLM`.

### 2.5 `agents/rag_retriever.py`

Placeholder agent yang akan memanggil RAG retriever chain dari `rag/retriever.py`. Lihat Phase 3.

### 2.6 `agents/__init__.py`

```python
"""Agent node functions for the satellite data LangGraph."""

from satellite_data_agent.agents.analyst import analyst_node
from satellite_data_agent.agents.data_fetcher import data_fetcher_node
from satellite_data_agent.agents.rag_retriever import rag_retriever_node
from satellite_data_agent.agents.report_gen import build_report_generator_node
from satellite_data_agent.agents.supervisor import build_supervisor_node

__all__ = [
    "analyst_node",
    "data_fetcher_node",
    "rag_retriever_node",
    "build_report_generator_node",
    "build_supervisor_node",
]
```

---

## Phase 3: RAG Pipeline (ChromaDB)

### Tujuan
Implementasi RAG pipeline sesuai plan: ChromaDB vector store + `all-MiniLM-L6-v2` embeddings + RAG Retriever node di graph.

### 3.1 Tambah dependencies

Tambahkan ke `requirements.txt`:
```
chromadb>=0.5.0
sentence-transformers>=3.0.0
langchain-community>=0.3.0
```

Tambahkan juga ke `pyproject.toml` di `dependencies`:
```
"chromadb>=0.5.0",
"sentence-transformers>=3.0.0",
"langchain-community>=0.3.0",
```

**Catatan:** `pydantic` sudah terinstall sebagai dependency transitif dari `langchain`. Tapi tambahkan eksplisit `pydantic>=2.0` di kedua file untuk kejelasan.

### 3.2 Buat `data/knowledge_base/satellite_ops_guide.md`

Buat file Markdown yang berisi pengetahuan domain satelit. Ini akan menjadi dokumen yang diindex oleh ChromaDB. Isi minimal:

```markdown
# Satellite Operations Reference Guide

## Signal Quality Thresholds

### SNR (Signal-to-Noise Ratio)
- Excellent: > 15 dB
- Good: 10–15 dB
- Marginal: 8–10 dB
- Poor: < 8 dB (action required: check antenna alignment, weather conditions)

### RSSI (Received Signal Strength Indicator)
- Strong: > -75 dBm
- Moderate: -75 to -85 dBm
- Weak: -85 to -90 dBm
- Critical: < -90 dBm (terminal may lose connectivity)

### Latency
- Normal (LEO/MEO): 20–150 ms
- Normal (GEO): 500–700 ms
- Elevated: 700–900 ms (possible congestion or rain fade)
- Critical: > 900 ms (investigate backhaul, routing, or hardware)

### Packet Loss
- Acceptable: < 1%
- Degraded: 1–3%
- Critical: > 3% (impacts real-time telemetry and remote operations)

## Common Anomaly Patterns

### Rain Fade
Simultaneous degradation of SNR and RSSI during precipitation events.
Typically affects Ka-band more than Ku-band. Duration: minutes to hours.
Check local weather data to correlate.

### Antenna Misalignment
Gradual SNR degradation over days/weeks without weather correlation.
Often affects a single terminal while neighbors remain stable.
Recommended action: schedule field technician for antenna re-pointing.

### Backhaul Congestion
High latency with normal SNR/RSSI. Often correlates with peak traffic hours.
Check gateway utilization and bandwidth allocation.

### Terminal Hardware Failure
Sudden drop to non-OK status with very low or zero throughput.
May show RSSI = 0 or SNR = 0. Requires terminal replacement or reboot.

## Satellite Systems Reference

### MODIS (Moderate Resolution Imaging Spectroradiometer)
- Platforms: Terra (EOS AM-1), Aqua (EOS PM-1)
- Key data products: MOD09GA (surface reflectance), MOD11A1 (land surface temperature)
- Spatial resolution: 250m (bands 1-2), 500m (bands 3-7), 1km (bands 8-36)

### Sentinel-2
- Operator: ESA (European Space Agency)
- Revisit time: 5 days at equator
- Spatial resolution: 10m (4 bands), 20m (6 bands), 60m (3 bands)
- Useful for: vegetation monitoring, land use change, water quality

### Landsat 8/9
- Operator: USGS/NASA
- Revisit time: 16 days per satellite (8 days combined)
- Spatial resolution: 30m multispectral, 15m panchromatic
```

**Tambahan opsional:** Jika mau lebih kaya, buat 2-3 file lagi di `data/knowledge_base/` (misalnya `nasa_cmr_guide.md`, `iot_telemetry_best_practices.md`). Tapi satu file sudah cukup untuk demo.

### 3.3 `rag/indexer.py`

```python
"""Document ingestion and ChromaDB indexing for the RAG pipeline.

Usage:
    python -m satellite_data_agent.rag.indexer

This reads all .md and .txt files from data/knowledge_base/,
splits them into chunks, embeds them with all-MiniLM-L6-v2,
and stores them in a persistent ChromaDB collection.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# Config
KNOWLEDGE_BASE_DIR = Path("data/knowledge_base")
CHROMA_PERSIST_DIR = Path("data/chroma_db")
COLLECTION_NAME = "satellite_ops_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def build_index() -> None:
    """Read documents, chunk, embed, and store in ChromaDB."""
    if not KNOWLEDGE_BASE_DIR.exists():
        print(f"Knowledge base directory not found: {KNOWLEDGE_BASE_DIR}")
        return

    # Load documents
    loader = DirectoryLoader(
        str(KNOWLEDGE_BASE_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()

    if not docs:
        print("No documents found in knowledge base.")
        return

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split {len(docs)} documents into {len(chunks)} chunks.")

    # Create ChromaDB client with persistent storage
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    # Delete collection if exists (re-index from scratch)
    try:
        client.delete_collection(COLLECTION_NAME)
    except ValueError:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    # Add chunks to collection
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=[chunk.page_content for chunk in chunks],
        metadatas=[{"source": chunk.metadata.get("source", "unknown")} for chunk in chunks],
    )

    print(f"Indexed {len(chunks)} chunks into ChromaDB collection '{COLLECTION_NAME}'.")
    print(f"Persisted to: {CHROMA_PERSIST_DIR}")


if __name__ == "__main__":
    build_index()
```

### 3.4 `rag/retriever.py`

```python
"""RAG retrieval chain using ChromaDB and LangChain LCEL."""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


CHROMA_PERSIST_DIR = Path("data/chroma_db")
COLLECTION_NAME = "satellite_ops_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3


def retrieve_context(query: str) -> str:
    """Retrieve relevant document chunks from ChromaDB for a given query.

    Returns concatenated text of top-K most similar chunks.
    If ChromaDB is not initialized or empty, returns empty string.
    """
    if not CHROMA_PERSIST_DIR.exists():
        return ""

    try:
        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
        )
    except Exception:
        return ""

    if collection.count() == 0:
        return ""

    results = collection.query(
        query_texts=[query],
        n_results=TOP_K,
    )

    documents = results.get("documents", [[]])[0]
    return "\n\n---\n\n".join(documents)
```

### 3.5 `agents/rag_retriever.py`

```python
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
```

### 3.6 Tambahkan `data/chroma_db/` ke `.gitignore`

Append baris berikut ke `.gitignore`:

```
data/chroma_db/
```

---

## Phase 4: NASA CMR API Integration

### Tujuan
Buat `tools/nasa_api.py` yang memanggil NASA Common Metadata Repository (CMR) API.

### 4.1 `tools/nasa_api.py`

```python
"""NASA Common Metadata Repository (CMR) API wrapper.

Endpoint: https://cmr.earthdata.nasa.gov/search
No authentication required — this is a public API.

Default region: Indonesian archipelago (bounding_box: 95,-11,141,6)
Default collection: MODIS Terra Surface Reflectance (MOD09GA v6.1)
"""

from __future__ import annotations

import requests
from datetime import datetime, timedelta


CMR_SEARCH_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"

# Indonesian archipelago bounding box: west, south, east, north
DEFAULT_BOUNDING_BOX = "95,-11,141,6"

# MODIS Terra Surface Reflectance Daily
DEFAULT_COLLECTION_CONCEPT_ID = "C2565788712-LPCLOUD"  # MOD09GA v6.1

DEFAULT_PAGE_SIZE = 10
DEFAULT_DAYS_BACK = 7


def fetch_cmr_granules(
    bounding_box: str = DEFAULT_BOUNDING_BOX,
    collection_concept_id: str = DEFAULT_COLLECTION_CONCEPT_ID,
    days_back: int = DEFAULT_DAYS_BACK,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> dict:
    """Fetch recent granule metadata from NASA CMR.

    Args:
        bounding_box: "west,south,east,north" in degrees.
        collection_concept_id: CMR collection concept ID.
        days_back: Number of days to look back from today.
        page_size: Max granules to return.

    Returns:
        dict with keys: records, record_count, source, time_range, sites, devices.
        Each record has: granule_id, dataset, time_start, time_end, bounding_box, links.
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)

    params = {
        "collection_concept_id": collection_concept_id,
        "bounding_box": bounding_box,
        "temporal": f"{start_date.strftime('%Y-%m-%dT00:00:00Z')},{end_date.strftime('%Y-%m-%dT23:59:59Z')}",
        "page_size": page_size,
        "sort_key": "-start_date",
    }

    response = requests.get(CMR_SEARCH_URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    entries = data.get("feed", {}).get("entry", [])

    records = []
    for entry in entries:
        record = {
            "granule_id": entry.get("id", ""),
            "title": entry.get("title", ""),
            "dataset": entry.get("dataset_id", ""),
            "time_start": entry.get("time_start", ""),
            "time_end": entry.get("time_end", ""),
            "updated": entry.get("updated", ""),
            "data_center": entry.get("data_center", ""),
            "original_format": entry.get("original_format", ""),
            "links": [
                link.get("href", "")
                for link in entry.get("links", [])
                if link.get("rel", "").endswith("/data#")
            ],
        }

        # Extract bounding box if available
        boxes = entry.get("boxes", [])
        if boxes:
            record["bounding_box"] = boxes[0]

        records.append(record)

    timestamps = [r["time_start"] for r in records if r.get("time_start")]

    return {
        "records": records,
        "record_count": len(records),
        "source": "NASA CMR API",
        "collection_concept_id": collection_concept_id,
        "bounding_box": bounding_box,
        "time_range": {
            "start": min(timestamps) if timestamps else "",
            "end": max(timestamps) if timestamps else "",
        },
        "sites": [],    # CMR doesn't have "sites" — field exists for interface compatibility
        "devices": [],  # same as above
    }
```

### 4.2 Tambahkan `requests` ke dependencies

Tambahkan ke `requirements.txt`:
```
requests>=2.31.0
```

Dan ke `pyproject.toml`:
```
"requests>=2.31.0",
```

**Catatan:** `requests` mungkin sudah terinstall sebagai dependency transitif dari package lain, tapi harus eksplisit.

### 4.3 `tools/__init__.py`

```python
"""External API tools."""
```

---

## Phase 5: Rewrite `graph.py`

Sekarang semua agent dan tools sudah ada, rewrite `graph.py` untuk:
1. Import dari `agents/` package.
2. Tambahkan `rag_retriever` node.
3. Update conditional edges untuk semua 5 possible next actions.
4. Pass LLM ke supervisor dan report_generator.

```python
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
```

---

## Phase 6: Update CLI + Entry Points

### 6.1 Update `cli.py`

```python
"""CLI entry point for the satellite data agent."""

from __future__ import annotations

import argparse

from satellite_data_agent.graph import build_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the satellite data LangGraph agent.")
    parser.add_argument(
        "query",
        nargs="?",
        default="Analyze satellite and IoT telemetry health for PSN operations.",
        help="User question or analysis objective",
    )
    parser.add_argument("--output", default="reports/output.md", help="Markdown report output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Set output path via env for report_gen to pick up
    import os
    os.environ.setdefault("REPORT_OUTPUT_PATH", args.output)

    app = build_graph()
    final_state = app.invoke(
        {
            "query": args.query,
            "iteration_count": 0,
        }
    )

    print("\n--- Agent History ---")
    for entry in final_state.get("agent_history", []):
        print(f"  {entry}")

    print(f"\nReport written to: {args.output}")

    if final_state.get("final_report"):
        print("\n--- Report Preview (first 500 chars) ---")
        print(final_state["final_report"][:500])


if __name__ == "__main__":
    main()
```

**Perubahan kunci:**
- `query` sekarang positional argument (sesuai plan: `python -m src.graph "Analyze recent..."`)
- Tidak ada `--input` — Data Fetcher memutuskan sendiri.
- `REPORT_OUTPUT_PATH` diset via env var, bukan state field.

### 6.2 Update `__init__.py` — TIDAK ADA PERUBAHAN DIPERLUKAN

File ini sudah benar. Tetap apa adanya.

### 6.3 Tambahkan `__main__.py` untuk `python -m satellite_data_agent`

Buat file `src/satellite_data_agent/__main__.py`:

```python
"""Allow running the package with python -m satellite_data_agent."""

from satellite_data_agent.cli import main

main()
```

---

## Phase 7: Streamlit UI

### 7.1 Tambah dependency

Di `requirements.txt`:
```
streamlit>=1.35.0
```

Di `pyproject.toml`:
```
"streamlit>=1.35.0",
```

### 7.2 Buat `app.py` di root project

```python
"""Streamlit UI for the satellite data agent."""

import streamlit as st
from satellite_data_agent.graph import build_graph


st.set_page_config(
    page_title="Satellite Data Agent",
    page_icon="🛰️",
    layout="wide",
)

st.title("🛰️ Satellite Data Agent")
st.markdown(
    "Multi-agent GenAI pipeline for automated satellite data analysis. "
    "Built with LangGraph cyclic graph · RAG + ChromaDB · LangSmith observability."
)

# Query input
query = st.text_area(
    "Analysis Query",
    value="Analyze satellite and IoT telemetry health for PSN operations.",
    height=80,
)

col1, col2 = st.columns([1, 4])
with col1:
    run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

if run_button:
    with st.spinner("Running agent pipeline..."):
        app = build_graph()
        final_state = app.invoke(
            {
                "query": query,
                "iteration_count": 0,
            }
        )

    # Agent History
    st.subheader("📋 Agent History")
    for entry in final_state.get("agent_history", []):
        st.markdown(f"- {entry}")

    # Analysis Metrics
    analysis = final_state.get("analysis_result", {})
    if analysis:
        st.subheader("📊 Analysis Metrics")
        metrics = analysis.get("metrics", {})
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Avg Latency", f"{metrics.get('avg_latency_ms', 0)} ms")
        with metric_cols[1]:
            st.metric("Avg Packet Loss", f"{metrics.get('avg_packet_loss_pct', 0)}%")
        with metric_cols[2]:
            st.metric("High Latency Records", metrics.get("high_latency_count", 0))
        with metric_cols[3]:
            st.metric("Low Signal Records", metrics.get("low_signal_count", 0))

    # RAG Context
    rag_context = final_state.get("rag_context")
    if rag_context:
        with st.expander("🔍 RAG Context Retrieved"):
            st.markdown(rag_context)

    # Final Report
    report = final_state.get("final_report", "")
    if report:
        st.subheader("📄 Generated Report")
        st.markdown(report)

        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name="satellite_analysis_report.md",
            mime="text/markdown",
        )
```

---

## Phase 8: Tests

### 8.1 Buat test files sesuai plan

**`tests/test_state.py`:**

```python
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
```

**`tests/test_agents.py`:**

```python
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
```

**`tests/test_graph.py`:**

```python
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
```

### 8.2 HAPUS `tests/test_graph_smoke.py`

File ini mereferensi old state (`TelemetryState`, `completed_steps`, `input_path`). Fungsinya sudah digantikan oleh `tests/test_graph.py`.

---

## Phase 9: Update Configuration Files

### 9.1 Update `requirements.txt`

Ganti seluruh file dengan:

```
langchain>=0.3.0
langchain-core>=0.3.0
langchain-community>=0.3.0
langchain-groq>=0.2.0
langchain-openai>=0.2.0
langgraph>=0.2.0
langsmith>=0.1.0
pandas>=2.2.0,<3.0.0
python-dotenv>=1.0.1
pydantic>=2.0
chromadb>=0.5.0
sentence-transformers>=3.0.0
requests>=2.31.0
streamlit>=1.35.0
```

### 9.2 Update `pyproject.toml`

Update hanya bagian `dependencies`:

```toml
dependencies = [
  "langchain>=0.3.0",
  "langchain-core>=0.3.0",
  "langchain-community>=0.3.0",
  "langchain-groq>=0.2.0",
  "langchain-openai>=0.2.0",
  "langgraph>=0.2.0",
  "langsmith>=0.1.0",
  "pandas>=2.2.0,<3.0.0",
  "python-dotenv>=1.0.1",
  "pydantic>=2.0",
  "chromadb>=0.5.0",
  "sentence-transformers>=3.0.0",
  "requests>=2.31.0",
  "streamlit>=1.35.0",
]
```

### 9.3 Update `.env.example`

Ganti seluruh file dengan:

```bash
# LLM provider: mock, groq, or openrouter
LLM_PROVIDER=mock

# Groq (free tier: https://console.groq.com)
GROQ_API_KEY=
GROQ_MODEL=llama-3.1-8b-instant

# OpenRouter (free tier: https://openrouter.ai)
OPENROUTER_API_KEY=
OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct:free

# LangSmith tracing (free: https://smith.langchain.com)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=satellite-data-agent

# Optional: override telemetry input path
# TELEMETRY_INPUT_PATH=data/telemetry/sample.json

# Optional: override report output path
# REPORT_OUTPUT_PATH=reports/output.md
```

**Perubahan:**
- LangSmith env var pakai `LANGCHAIN_TRACING_V2` (bukan `LANGSMITH_TRACING`). Kedua var diterima oleh LangSmith SDK, tapi `LANGCHAIN_TRACING_V2` adalah yang canonical dan sesuai plan.
- Tambah env var baru untuk path override.

### 9.4 Update `.gitignore`

Append:

```
data/chroma_db/
```

---

## Phase 10: Update `scripts/render_graph.py`

Tidak perlu perubahan besar. File ini sudah import dari `satellite_data_agent.graph`. Setelah graph.py di-rewrite, script ini akan otomatis merender graph baru yang include `rag_retriever` node.

Satu-satunya perubahan: output `.mmd` dan `.png` sekarang akan include 5 nodes (supervisor, data_fetcher, rag_retriever, analyst, report_generator) bukan 4.

**Tidak perlu edit manual.**

---

## Phase 11: Update README.md

Update README.md agar mencerminkan:
1. Quickstart command baru (`python -m satellite_data_agent "query here"`).
2. RAG indexing step (`python -m satellite_data_agent.rag.indexer`).
3. Streamlit command (`streamlit run app.py`).
4. Arsitektur baru dengan 5 agents.
5. Update Project Layout section.

Jangan tulis ulang seluruh README — cukup update sections yang berubah. Pertahankan bagian yang masih relevan (Architecture mermaid diagram perlu diupdate untuk include `rag_retriever`).

---

## Phase 12: Update `plan-satellite-data-agent.md` Roadmap

Update checklist di bagian Roadmap agar akurat setelah implementasi:

```markdown
## Roadmap

- [x] Architecture design + state schema
- [x] Supervisor routing with structured output
- [x] NASA CMR API integration
- [x] ChromaDB RAG pipeline
- [x] Analyst agent with Pydantic output validation
- [x] Report generator with Markdown rendering
- [ ] LangSmith trace integration + screenshots
- [x] Streamlit UI
- [x] Unit tests (pytest)
- [ ] Docker containerization
- [ ] Integration with [`vlm-satellite-image-analyzer`](../vlm-satellite-image-analyzer)
```

**Catatan:** LangSmith trace screenshots belum bisa di-checklist karena memerlukan API key aktif dan run aktual. Docker juga belum.

---

## Verification Plan

Setelah semua Phase selesai, jalankan:

### 1. Unit tests
```powershell
cd d:\GITHUB\satellite-data-agent
python -m pytest tests/ -v
```

Semua 3 test files harus pass. Test `test_graph.py::test_full_pipeline_with_mock_llm` adalah test integrasi paling penting — ia membuktikan seluruh pipeline berjalan dari query → report.

### 2. CLI smoke test
```powershell
python -m satellite_data_agent "Analyze satellite and IoT telemetry health for PSN operations."
```

Output yang diharapkan: agent history log + report preview + file `reports/output.md`.

### 3. RAG indexing
```powershell
python -m satellite_data_agent.rag.indexer
```

Output yang diharapkan: "Indexed N chunks into ChromaDB collection 'satellite_ops_docs'."

### 4. Graph rendering
```powershell
python scripts/render_graph.py
```

Output yang diharapkan: file `.mmd` baru dengan 5 nodes (termasuk `rag_retriever`).

### 5. Streamlit (manual)
```powershell
streamlit run app.py
```

Browser harus membuka UI dengan input area + Run button.

---

## File Summary — Final State

```
satellite-data-agent/
├── src/satellite_data_agent/
│   ├── __init__.py              (UNCHANGED)
│   ├── __main__.py              (NEW)
│   ├── state.py                 (REWRITTEN — AgentState with Annotated)
│   ├── graph.py                 (REWRITTEN — 5 nodes + rag_retriever)
│   ├── llm.py                   (UNCHANGED)
│   ├── cli.py                   (REWRITTEN — positional query arg)
│   ├── agents/
│   │   ├── __init__.py          (NEW)
│   │   ├── supervisor.py        (NEW — LLM-based + deterministic fallback)
│   │   ├── data_fetcher.py      (NEW — local + NASA CMR)
│   │   ├── rag_retriever.py     (NEW — ChromaDB retrieval)
│   │   ├── analyst.py           (NEW — Pydantic validated)
│   │   └── report_gen.py        (NEW — Pydantic + RAG context)
│   ├── tools/
│   │   ├── __init__.py          (NEW)
│   │   └── nasa_api.py          (NEW — NASA CMR wrapper)
│   └── rag/
│       ├── __init__.py          (NEW)
│       ├── indexer.py           (NEW — ChromaDB ingestion)
│       └── retriever.py         (NEW — semantic search)
├── data/
│   ├── sample_telemetry.csv     (KEEP — backward compat)
│   ├── telemetry/
│   │   └── sample.json          (NEW — JSON version of CSV)
│   └── knowledge_base/
│       └── satellite_ops_guide.md (NEW — RAG source document)
├── tests/
│   ├── test_state.py            (NEW)
│   ├── test_agents.py           (NEW)
│   └── test_graph.py            (NEW — replaces test_graph_smoke.py)
├── app.py                       (NEW — Streamlit UI)
├── scripts/render_graph.py      (UNCHANGED)
├── docs/                        (auto-generated by render_graph)
├── reports/                     (auto-generated by pipeline)
├── pyproject.toml               (UPDATED — new deps)
├── requirements.txt             (UPDATED — new deps)
├── requirements-dev.txt         (UNCHANGED)
├── .env.example                 (UPDATED)
├── .gitignore                   (UPDATED — add chroma_db)
├── README.md                    (UPDATED)
└── plan-satellite-data-agent.md (UPDATED — roadmap checkmarks)

DELETED:
├── src/satellite_data_agent/nodes.py        (DELETED — replaced by agents/)
├── tests/test_graph_smoke.py                (DELETED — replaced by test_graph.py)
```

---

## Execution Order

Implementasikan dalam urutan Phase 1 → 12. Setiap phase bergantung pada phase sebelumnya. Jangan skip.

**Phase yang bisa dilakukan paralel:** Phase 3 (RAG) dan Phase 4 (NASA API) bisa dikerjakan bersamaan karena tidak saling bergantung. Tapi keduanya harus selesai SEBELUM Phase 5 (graph.py rewrite).

**Critical path:** Phase 1 → Phase 2 → (Phase 3 + Phase 4) → Phase 5 → Phase 6 → Phase 7 → Phase 8 → Phase 9 → Phase 10 → Phase 11 → Phase 12 → Verification.

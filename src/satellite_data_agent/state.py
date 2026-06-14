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

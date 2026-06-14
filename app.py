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

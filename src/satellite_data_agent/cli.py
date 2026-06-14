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

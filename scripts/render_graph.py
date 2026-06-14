from __future__ import annotations

from pathlib import Path

from satellite_data_agent.graph import build_graph


def main() -> None:
    output_dir = Path("docs")
    output_dir.mkdir(exist_ok=True)

    app = build_graph()
    drawable = app.get_graph()
    mermaid = drawable.draw_mermaid()
    (output_dir / "langgraph_state_graph.mmd").write_text(mermaid, encoding="utf-8")

    try:
        png = drawable.draw_mermaid_png()
    except Exception as exc:  # pragma: no cover - depends on network/render backend
        print(f"PNG render skipped: {exc}")
    else:
        (output_dir / "langgraph_state_graph.png").write_bytes(png)
        print("Wrote docs/langgraph_state_graph.png")

    print("Wrote docs/langgraph_state_graph.mmd")


if __name__ == "__main__":
    main()

from langgraph.graph import StateGraph

from agents import (
    analysis_agent,
    data_agent,
    decision_node,
    emergency_agent,
    knowledge_agent,
    reasoning_agent,
    recommendation_agent,
    route_decision,
)
from state import HealthState


# ---------------------------------------------------------------------------
# Graph Definition
# ---------------------------------------------------------------------------
def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph multi-agent pipeline.

    Flow:
        data → analysis → decision ──► emergency ──► recommendation
                                   └─► knowledge → reasoning ──► recommendation
    """
    builder = StateGraph(HealthState)

    # ── Nodes ────────────────────────────────────────────────────────────────
    builder.add_node("data",           data_agent)
    builder.add_node("analysis",       analysis_agent)
    builder.add_node("decision",       decision_node)
    builder.add_node("knowledge",      knowledge_agent)
    builder.add_node("reasoning",      reasoning_agent)
    builder.add_node("emergency",      emergency_agent)
    builder.add_node("recommendation", recommendation_agent)

    # ── Edges ─────────────────────────────────────────────────────────────────
    builder.set_entry_point("data")
    builder.add_edge("data",      "analysis")
    builder.add_edge("analysis",  "decision")

    builder.add_conditional_edges(
        "decision",
        route_decision,
        {
            "emergency": "emergency",
            "normal":    "knowledge",
        },
    )

    builder.add_edge("knowledge",  "reasoning")
    builder.add_edge("reasoning",  "recommendation")
    builder.add_edge("emergency",  "recommendation")

    builder.set_finish_point("recommendation")

    return builder.compile()


# ---------------------------------------------------------------------------
# Compiled Graph  (imported by app.py and ragas_eval.py)
# ---------------------------------------------------------------------------
graph = build_graph()


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    symptoms = input("Enter symptoms: ").strip()

    if not symptoms:
        print("No symptoms provided.")
    else:
        result = graph.invoke({"symptoms": symptoms})

        print("\n" + "=" * 50)
        print("        🧠  AI Clinical Report")
        print("=" * 50)
        print(f"  Symptoms       : {result.get('symptoms')}")
        print(f"  Risk Level     : {result.get('risk')}")
        print(f"  Diagnosis      : {result.get('diagnosis')}")
        print(f"  Recommendation : {result.get('recommendation')}")
        print("=" * 50)

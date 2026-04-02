from langgraph.graph import StateGraph
from state import HealthState
from agents import (
    data_agent,
    analysis_agent,
    reasoning_agent,
    recommendation_agent,
    decision_node,
    route_decision,
    emergency_agent,
    knowledge_agent   # 🔥 NEW
)

# ------------------------
# BUILD GRAPH
# ------------------------
builder = StateGraph(HealthState)

# Nodes
builder.add_node("data", data_agent)
builder.add_node("analysis", analysis_agent)
builder.add_node("decision", decision_node)
builder.add_node("knowledge", knowledge_agent)   # 🔥 NEW
builder.add_node("reasoning", reasoning_agent)
builder.add_node("emergency", emergency_agent)
builder.add_node("recommendation", recommendation_agent)

# ------------------------
# FLOW
# ------------------------
builder.set_entry_point("data")

builder.add_edge("data", "analysis")
builder.add_edge("analysis", "decision")

# 🔥 Conditional routing
builder.add_conditional_edges(
    "decision",
    route_decision,
    {
        "emergency": "emergency",
        "normal": "knowledge"   # 👈 goes to RAG first
    }
)

# 🔥 RAG flow
builder.add_edge("knowledge", "reasoning")

# Final steps
builder.add_edge("reasoning", "recommendation")
builder.add_edge("emergency", "recommendation")

builder.set_finish_point("recommendation")

# Compile graph
graph = builder.compile()

# ------------------------
# RUN SYSTEM
# ------------------------
if __name__ == "__main__":
    symptoms = input("Enter symptoms: ")

    result = graph.invoke({
        "symptoms": symptoms
    })

    print("\n🧠 AI Clinical Report")
    print("----------------------")
    print(f"Symptoms: {result.get('symptoms')}")
    print(f"Risk: {result.get('risk')}")
    print(f"Diagnosis: {result.get('diagnosis')}")
    print(f"Recommendation: {result.get('recommendation')}")
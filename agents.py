import os

from dotenv import load_dotenv
from groq import Groq

from rag import build_vectorstore
from state import HealthState

# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------
load_dotenv()

_client      = Groq(api_key=os.getenv("GROQ_API_KEY"))
_vectorstore = build_vectorstore()

_LLM_MODEL   = "llama-3.1-8b-instant"
_LLM_TEMP    = 0.2
_RAG_TOP_K   = 3


# ---------------------------------------------------------------------------
# Agent 1 — Data Agent
# ---------------------------------------------------------------------------
def data_agent(state: HealthState) -> dict:
    """Normalise raw symptom input to lowercase."""
    return {"symptoms": state["symptoms"].strip().lower()}


# ---------------------------------------------------------------------------
# Agent 2 — Analysis Agent
# ---------------------------------------------------------------------------
def analysis_agent(state: HealthState) -> dict:
    """
    Assign a risk level based on keyword matching.

    Rules:
        HIGH   → chest pain detected
        MEDIUM → fever detected
        LOW    → all other symptoms
    """
    symptoms = state["symptoms"]

    if "chest pain" in symptoms:
        risk = "HIGH"
    elif "fever" in symptoms:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {"risk": risk}


# ---------------------------------------------------------------------------
# Decision Node + Router
# ---------------------------------------------------------------------------
def decision_node(state: HealthState) -> dict:
    """Passthrough node that triggers conditional routing."""
    return {}


def route_decision(state: HealthState) -> str:
    """
    Route HIGH-risk cases to emergency agent.
    All other cases proceed through the RAG pipeline.
    """
    return "emergency" if state["risk"] == "HIGH" else "normal"


# ---------------------------------------------------------------------------
# Agent 3 — Knowledge Agent  (RAG retrieval)
# ---------------------------------------------------------------------------
def knowledge_agent(state: HealthState) -> dict:
    """
    Retrieve the top-K most relevant medical documents
    from the FAISS vectorstore for the given symptoms.
    """
    docs      = _vectorstore.similarity_search(state["symptoms"], k=_RAG_TOP_K)
    knowledge = [doc.page_content for doc in docs]
    return {"knowledge": knowledge}


# ---------------------------------------------------------------------------
# Agent 4 — Reasoning Agent  (LLM diagnosis)
# ---------------------------------------------------------------------------
def reasoning_agent(state: HealthState) -> dict:
    """
    Send retrieved medical knowledge + symptoms to the LLM
    and generate a structured clinical diagnosis.
    """
    prompt = f"""
You are a clinical AI assistant. Analyse the patient's symptoms using
the relevant medical knowledge provided below.

Patient Symptoms : {state.get("symptoms")}
Risk Level       : {state.get("risk")}

Relevant Medical Knowledge:
{chr(10).join(f"- {k}" for k in state.get("knowledge", []))}

Provide a concise clinical response with:
1. Most likely disease
2. Brief reason based on the knowledge above
3. Any important warnings
"""

    try:
        response = _client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=_LLM_MODEL,
            temperature=_LLM_TEMP,
        )
        diagnosis = response.choices[0].message.content

    except Exception as exc:
        diagnosis = f"Diagnosis unavailable. Error: {exc}"

    return {"diagnosis": diagnosis}


# ---------------------------------------------------------------------------
# Agent 5 — Emergency Agent
# ---------------------------------------------------------------------------
def emergency_agent(state: HealthState) -> dict:
    """
    Bypass the RAG pipeline for HIGH-risk cases and return
    an immediate emergency recommendation.
    """
    return {
        "diagnosis":      "Critical condition suspected based on reported symptoms.",
        "recommendation": "🚨 Seek emergency medical care immediately. Do not delay.",
    }


# ---------------------------------------------------------------------------
# Agent 6 — Recommendation Agent
# ---------------------------------------------------------------------------
def recommendation_agent(state: HealthState) -> dict:
    """
    Return the final patient recommendation.
    HIGH-risk recommendations are already set by the emergency agent.
    """
    if state.get("risk") == "HIGH":
        return {"recommendation": state.get("recommendation")}

    return {
        "recommendation": (
            "Monitor your symptoms closely. "
            "If symptoms persist or worsen, consult a qualified medical professional."
        )
    }

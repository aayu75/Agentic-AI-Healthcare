from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv
import os
from rag import build_vectorstore

# ------------------------
# SETUP
# ------------------------
load_dotenv()

from groq import Groq
import os

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)
vectorstore = build_vectorstore()

# ------------------------
# DATA AGENT
# ------------------------
def data_agent(state):
    return {"symptoms": state["symptoms"].lower()}

# ------------------------
# ANALYSIS AGENT
# ------------------------
def analysis_agent(state):
    symptoms = state["symptoms"]

    if "chest pain" in symptoms:
        risk = "HIGH"
    elif "fever" in symptoms:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {"risk": risk}

# ------------------------
# DECISION NODE
# ------------------------
def decision_node(state):
    print("🧠 decision_node running")
    return {}

# ------------------------
# ROUTER
# ------------------------
def route_decision(state):
    if state["risk"] == "HIGH":
        return "emergency"
    return "normal"

# ------------------------
# KNOWLEDGE AGENT (RAG)
# ------------------------
def knowledge_agent(state):
    print("📚 knowledge_agent running")

    results = vectorstore.similarity_search(state["symptoms"], k=3)

    knowledge = [doc.page_content for doc in results]

    print("📖 Retrieved:", knowledge)

    return {"knowledge": knowledge}

# ------------------------
# REASONING AGENT
# ------------------------
def reasoning_agent(state):
    print("🔥 reasoning_agent running")

    prompt = f"""
    You are a clinical AI assistant.

    Symptoms: {state.get('symptoms')}
    Risk: {state.get('risk')}

    Relevant knowledge:
    {state.get('knowledge')}

    Provide:
    - likely disease
    - short reason
    """

    diagnosis = "Fallback diagnosis"

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",   
            temperature=0.2
        )

        diagnosis = response.choices[0].message.content

    except Exception as e:
        print("❌ LLM ERROR:", e)

    return {"diagnosis": diagnosis}

# ------------------------
# EMERGENCY AGENT
# ------------------------
def emergency_agent(state):
    print("🚨 emergency_agent running")

    return {
        "diagnosis": "Critical condition suspected",
        "recommendation": "🚨 Immediate hospital visit required"
    }

# ------------------------
# RECOMMENDATION AGENT
# ------------------------
def recommendation_agent(state):
    if state.get("risk") == "HIGH":
        return {"recommendation": state.get("recommendation")}

    return {"recommendation": "Monitor at home or consult doctor."}

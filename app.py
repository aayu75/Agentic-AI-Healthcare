import streamlit as st
from dotenv import load_dotenv

from main import graph
from rag import build_vectorstore

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Healthcare Assistant",
    page_icon="🧠",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Preload Vectorstore
# @st.cache_resource in rag.py ensures this runs only once per session
# ---------------------------------------------------------------------------
with st.spinner("Loading knowledge base..."):
    build_vectorstore()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🧠 AI Healthcare Assistant")
st.markdown(
    "Enter your symptoms below to receive an AI-powered clinical assessment. "
    "This tool is for **informational purposes only** and does not replace professional medical advice."
)
st.divider()

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
symptoms = st.text_input(
    label="Describe your symptoms",
    placeholder="e.g. fever, cough, chest pain, blurry vision",
)

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
if st.button("🔍 Analyse Symptoms", use_container_width=True):

    if not symptoms.strip():
        st.warning("Please enter your symptoms before analysing.")

    else:
        try:
            with st.spinner("Analysing symptoms..."):
                result = graph.invoke({"symptoms": symptoms})

            st.success("Analysis complete ✅")
            st.divider()

            # ── Risk Level ──────────────────────────────────────────────────
            st.subheader("📊 Assessment Summary")

            col1, col2 = st.columns(2)

            risk = result.get("risk", "N/A")
            with col1:
                if risk == "HIGH":
                    st.error(f"🔴 Risk Level: **{risk}**")
                elif risk == "MEDIUM":
                    st.warning(f"🟡 Risk Level: **{risk}**")
                else:
                    st.success(f"🟢 Risk Level: **{risk}**")

            with col2:
                st.metric(label="Symptoms Entered", value=result.get("symptoms", "N/A"))

            st.divider()

            # ── Diagnosis ───────────────────────────────────────────────────
            st.subheader("🧾 Diagnosis")
            st.info(result.get("diagnosis", "No diagnosis available."))

            # ── Recommendation ──────────────────────────────────────────────
            st.subheader("💡 Recommendation")
            st.warning(result.get("recommendation", "No recommendation available."))

            st.divider()

            # ── Feedback ────────────────────────────────────────────────────
            st.subheader("💬 Feedback")
            st.caption("Was this assessment helpful?")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍  Helpful", use_container_width=True):
                    st.success("Thank you for your feedback!")
            with col2:
                if st.button("👎  Not Helpful", use_container_width=True):
                    st.info("Thank you. Please consult a qualified doctor for further guidance.")

        except Exception as exc:
            st.error(f"An error occurred: {exc}")
            st.info("Please try again. If the problem persists, consult a medical professional.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "⚠️ This AI assistant is for informational purposes only. "
    "It does not constitute medical advice, diagnosis, or treatment. "
    "Always consult a qualified healthcare professional."
)

import streamlit as st
from main import graph

# Page config
st.set_page_config(
    page_title="AI Healthcare Assistant",
    page_icon="🧠",
    layout="centered"
)

# Title
st.title("🧠 AI Healthcare Assistant")
st.markdown("Enter symptoms to get AI-based medical insights")

# Input
symptoms = st.text_input("Enter symptoms (e.g., fever, cough, chest pain)")

# Button
if st.button("Analyze"):
    if symptoms:
        with st.spinner("Analyzing..."):
            result = graph.invoke({"symptoms": symptoms})

        st.success("Analysis Complete ✅")

        # Results
        st.subheader("📊 Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Risk Level", result.get("risk"))

        with col2:
            st.metric("Symptoms", result.get("symptoms"))

        st.markdown("### 🧾 Diagnosis")
        st.info(result.get("diagnosis"))

        st.markdown("### 💡 Recommendation")
        st.warning(result.get("recommendation"))

    else:
        st.warning("Please enter symptoms first")
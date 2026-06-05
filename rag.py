import os

import pandas as pd
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(BASE_DIR, "medquad.csv")
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")
MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Vectorstore Builder
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_vectorstore() -> FAISS:
    """
    Build (or load) a FAISS vectorstore from the MedQuAD dataset.

    - First run  : reads all rows, embeds them, saves index to disk.
    - Subsequent : loads the saved index instantly (~1 second).
    - Streamlit  : @st.cache_resource ensures this runs only once per session.

    Returns:
        FAISS vectorstore ready for similarity search.
    """
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    # ── Load from disk if index already exists ──────────────────────────────
    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    # ── Build from full dataset (runs only once) ────────────────────────────
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.lower()
    df = df.dropna(subset=["question", "answer"])

    texts = [
        f"Q: {str(row['question'])}  A: {str(row['answer'])}"
        for _, row in df.iterrows()
    ]

    vectorstore = FAISS.from_texts(texts, embeddings)
    vectorstore.save_local(INDEX_PATH)

    return vectorstore

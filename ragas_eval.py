"""
ragas_eval.py
─────────────
Evaluates the RAG pipeline using the RAGAS framework.

Metrics:
    - Faithfulness       : Is the answer grounded in retrieved context?
    - Answer Relevancy   : Is the answer relevant to the question?
    - Context Precision  : Are the most relevant docs ranked highest?
    - Context Recall     : Did retrieval capture all needed information?
"""

import os

import pandas as pd
from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from ragas.run_config import RunConfig

from main import graph
from rag import build_vectorstore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
CSV_PATH       = os.path.join(BASE_DIR, "medquad.csv")
EVAL_SAMPLES   = 10
RANDOM_STATE   = 42
GROQ_MODEL     = "llama-3.1-8b-instant"
EMBED_MODEL    = "all-MiniLM-L6-v2"
RAG_TOP_K      = 3


# ---------------------------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------------------------
print("=" * 60)
print("  RAGAS Evaluation — AI Healthcare Assistant")
print("=" * 60)
print("\n📂 Loading MedQuAD dataset...")

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.lower()
df = df.dropna(subset=["question", "answer"])

# ── Randomly sample diverse questions across whole dataset ──────────────────
# Using sample() instead of head() avoids evaluating only Glaucoma questions
df_eval = df.sample(EVAL_SAMPLES, random_state=RANDOM_STATE)

print(f"✅ Evaluating on {EVAL_SAMPLES} diverse questions:")
for i, q in enumerate(df_eval["question"].tolist(), 1):
    print(f"   {i:02d}. {q}")


# ---------------------------------------------------------------------------
# Load Vectorstore
# ---------------------------------------------------------------------------
print("\n📚 Loading vectorstore...")
vectorstore = build_vectorstore()


# ---------------------------------------------------------------------------
# Evaluator LLM + Embeddings
# ---------------------------------------------------------------------------
print("🧠 Initialising evaluator LLM and embeddings...")

evaluator_llm = LangchainLLMWrapper(
    ChatGroq(model=GROQ_MODEL, api_key=os.getenv("GROQ_API_KEY"))
)

evaluator_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name=EMBED_MODEL)
)

run_config = RunConfig(
    timeout=120,
    max_retries=5,
    max_wait=30,
)


# ---------------------------------------------------------------------------
# Build Evaluation Samples
# ---------------------------------------------------------------------------
samples = []
print("\n🚀 Running pipeline on evaluation questions...\n")

for idx, (_, row) in enumerate(df_eval.iterrows(), 1):
    question     = row["question"]
    ground_truth = row["answer"]

    print(f"[{idx:02d}/{EVAL_SAMPLES}] 🔎 {question}")

    # Retrieve context
    docs     = vectorstore.similarity_search(question, k=RAG_TOP_K)
    contexts = [doc.page_content for doc in docs]

    # Run full agent pipeline
    result          = graph.invoke({"symptoms": question})
    generated_answer = result.get("diagnosis", "No diagnosis generated.")

    print(f"        ✅ {generated_answer[:100]}...\n")

    samples.append(
        SingleTurnSample(
            user_input        = question,
            response          = generated_answer,
            retrieved_contexts= contexts,
            reference         = ground_truth,
        )
    )


# ---------------------------------------------------------------------------
# Evaluate with RAGAS
# ---------------------------------------------------------------------------
dataset = EvaluationDataset(samples=samples)

print("📊 Running RAGAS evaluation...\n")

result = evaluate(
    dataset   = dataset,
    metrics   = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
    ],
    run_config= run_config,
)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
results_df = result.to_pandas()
score_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.4f}".format)

print("\n" + "=" * 60)
print("  🎯  RAGAS Evaluation Results")
print("=" * 60)

print("\n📈 Per-Question Scores:")
print(results_df[["user_input"] + score_cols].to_string(index=False))

print("\n📊 Average Scores (NaN excluded):")
for col in score_cols:
    avg = results_df[col].mean()
    print(f"   {col:<25} : {avg:.4f}")

print("\n" + "=" * 60)

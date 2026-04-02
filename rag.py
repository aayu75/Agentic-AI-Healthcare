import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
def build_vectorstore():
    df = pd.read_csv(r"medquad.csv")

    # normalize column names
    df.columns = df.columns.str.lower()

    # use small sample (important)
    df = df.sample(500)

    texts = []
    for _, row in df.iterrows():
        question = str(row.get("question", ""))
        answer = str(row.get("answer", ""))

        text = f"Q: {question} A: {answer}"
        texts.append(text)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = FAISS.from_texts(texts, embeddings)

    return vectorstore
print("Vectorstore built successfully")

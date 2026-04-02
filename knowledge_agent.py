import kagglehub

# Download latest version
path ='C:\ds lab7\venv'
path = kagglehub.dataset_download("pythonafroz/medquad-medical-question-answer-for-ai-research")

print("Path to dataset files:", path )

import pandas as pd

df = pd.read_csv(r"C:\Users\ayush\.cache\kagglehub\datasets\pythonafroz\medquad-medical-question-answer-for-ai-research\versions\1\medquad.csv")

print(df.columns)

texts = []

for _, row in df.iterrows():
    text = f"Q: {row['Question']} A: {row['Answer']}"
    texts.append(text)

from rag import build_vectorstore
import pandas as pd



texts = [
    f"Q: {row['Question']} A: {row['Answer']}"
    for _, row in df.iterrows()
]

vectorstore = build_vectorstore(texts)

def knowledge_agent(state):
    print("📚 knowledge_agent running")

    query = state["symptoms"]

    results = vectorstore.similarity_search(query, k=3)

    knowledge = [doc.page_content for doc in results]

    return {"knowledge": knowledge}

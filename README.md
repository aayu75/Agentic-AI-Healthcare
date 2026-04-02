# 🧠 Agentic AI Healthcare System

An end-to-end **Agentic AI system for healthcare decision support**, built using multi-agent orchestration, Retrieval-Augmented Generation (RAG), and LLM-based reasoning.

---

## 🚀 Overview

This project demonstrates how autonomous AI agents can collaborate to:

* Analyze patient symptoms
* Assess risk levels
* Retrieve relevant medical knowledge
* Generate AI-assisted diagnosis
* Provide actionable recommendations

It serves as a foundation for **clinical decision support systems** and future research in **distributed and federated AI systems**.

---

## 🏗️ Architecture

The system is built using a **multi-agent pipeline** orchestrated via LangGraph.

### 🔄 Workflow

```
Input Symptoms
      ↓
Data Agent
      ↓
Analysis Agent (Risk Detection)
      ↓
Decision Node
   ↙        ↘
Emergency   Normal
   ↓          ↓
Emergency   Knowledge Agent (RAG)
   ↓          ↓
Recommendation   Reasoning Agent (LLM)
                    ↓
              Recommendation Agent
```

---

## 🤖 Agents

### 1. Data Agent

* Cleans and preprocesses input symptoms

### 2. Analysis Agent

* Determines risk level:

  * HIGH → emergency
  * MEDIUM / LOW → normal flow

### 3. Decision Node

* Routes execution path based on risk

### 4. Knowledge Agent (RAG)

* Retrieves relevant medical context from dataset
* Uses FAISS vector database

### 5. Reasoning Agent

* Uses Groq LLM to generate diagnosis
* Combines symptoms + retrieved knowledge

### 6. Emergency Agent

* Handles critical cases instantly

### 7. Recommendation Agent

* Provides final medical advice

---

## 🧠 Tech Stack

* **LangGraph** → Agent orchestration
* **LangChain** → AI pipeline utilities
* **FAISS** → Vector database
* **MedQuAD Dataset** → Medical knowledge base
* **Groq API (LLaMA 3.1)** → LLM inference
* **Streamlit** → UI
* **Python** → Core implementation

---

## 📊 Features

* Multi-agent architecture
* Conditional routing (emergency vs normal)
* Retrieval-Augmented Generation (RAG)
* Real-time LLM reasoning
* Interactive UI with Streamlit
* Cloud deployment ready

---

## 📁 Project Structure

```
.
├── app.py              # Streamlit UI
├── main.py             # LangGraph pipeline
├── agents.py           # All agents
├── rag.py              # RAG + vectorstore
├── state.py            # State definition
├── medquad.csv         # Dataset
├── requirements.txt
└── .env
```

---

## ⚙️ Installation

### 1. Clone repo

```bash
git clone <your-repo-url>
cd healthcare-agentic-ai
```

---

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Set environment variable

Create `.env` file:

```bash
GROQ_API_KEY=your_api_key_here
```

---

## ▶️ Run Application

```bash
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 🧪 Example Input

```
fever cough
```

### Output:

* Risk Level: MEDIUM
* Diagnosis: AI-generated explanation
* Recommendation: Consult doctor

---

## ⚠️ Limitations

* Not a replacement for professional medical advice
* Dataset may not perfectly match all symptoms
* RAG retrieval may return approximate results

---

## 🔮 Future Work

* Improve dataset quality (clinical datasets)
* Add real-time patient monitoring
* Integrate ECG / wearable data
* Deploy distributed multi-agent system
* Enhance UI/UX

---

## 🧠 Learnings

* Designing agent-based AI systems
* Handling real-world dependency conflicts
* Building RAG pipelines
* Deploying AI systems in production environments

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit PRs.

---

## 📜 License

This project is for educational and research purposes.

---

## 👨‍💻 Author

**Ayush Kumar**
M.S. AI & Data Science

---

⭐ If you found this useful, consider giving it a star!

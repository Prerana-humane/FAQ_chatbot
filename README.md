# 🧠 FAQ Chatbot using RAG (FAISS + Streamlit)

An AI-powered **FAQ Chatbot** built using **Retrieval-Augmented Generation (RAG)**.  
The system performs **semantic search** over FAQ content using **FAISS** and generates accurate, context-aware answers using a lightweight **LLM (Flan-T5)**.

This project demonstrates practical usage of **LLMs, vector databases, and backend design patterns** in a real-world FAQ system.

---

## 🚀 Features

- 🔍 Semantic search using **FAISS + Sentence Transformers**
- 🧠 Retrieval-Augmented Generation (RAG) pipeline
- 📊 Confidence-based answer handling (High / Medium / Low)
- 💡 Suggested questions for better user experience
- 📝 User query logging for analytics & improvement
- 🖥️ Clean and interactive **Streamlit UI**
- 📁 JSON-based storage (simple & lightweight)

---

## 🧱 Tech Stack

- **Python**
- **Streamlit**
- **FAISS**
- **HuggingFace Transformers**
- **Sentence-Transformers**
- **LangChain Embeddings**
- **NumPy**

---

## 🏗️ System Architecture

```text
User Question
     ↓
Embedding Model (Sentence Transformers)
     ↓
FAISS Vector Search (Top-K Retrieval)
     ↓
Relevant FAQ Context
     ↓
LLM (Flan-T5)
     ↓
Final Answer + Confidence Handling

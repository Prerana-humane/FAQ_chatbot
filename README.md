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

## ⚙️ Installation & Setup
1️⃣ Clone the Repository
- git clone https://github.com/your-username/FAQ_chatbot.git
- cd FAQ_chatbot

2️⃣ Create a Virtual Environment (Recommended)
- python3 -m venv venv
 source venv/bin/activate
On Windows:
- venv\Scripts\activate

3️⃣ Install Dependencies
- pip install -r requirements.txt

4️⃣ Prepare FAQ Data
- Ensure your FAQ knowledge base is available in:
- faq_data.json

5️⃣ Run the Application
- streamlit run faq_bot.py

---
## 🧪 Usage
1. Open the Streamlit app in your browser.
2. Select a question from Suggested Questions or type your own.
3. The chatbot retrieves relevant FAQ content using FAISS.
4. The LLM generates an answer based only on retrieved context.
5. Confidence-based filtering prevents hallucinated responses.


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

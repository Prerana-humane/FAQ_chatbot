import streamlit as st
import numpy as np
import faiss
import json
import os
from datetime import datetime
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------
# FILE PATHS
# ---------------------------
FAQ_PATH = "faq_data.json"
USER_LOG_PATH = "user_queries.json"

# ---------------------------
# STREAMLIT CONFIG
# ---------------------------
st.set_page_config(
    page_title="FAQ Chatbot (RAG + FAISS + JSON)",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 FAQ Chatbot")
st.write("Ask questions about returns, shipping, payments, or warranty.")

# ---------------------------
# SESSION STATE
# ---------------------------
if "question" not in st.session_state:
    st.session_state["question"] = ""

# ---------------------------
# UTILITY FUNCTIONS
# ---------------------------
def load_faq_data(path=FAQ_PATH):
    with open(path, "r") as f:
        return json.load(f)


def save_user_query(question, answer, score, path=USER_LOG_PATH):
    entry = {
        "question": question,
        "answer": answer,
        "similarity_score": round(float(score), 3),
        "timestamp": datetime.now().isoformat()
    }

    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump([entry], f, indent=2)
    else:
        with open(path, "r+") as f:
            data = json.load(f)
            data.append(entry)
            f.seek(0)
            json.dump(data, f, indent=2)

# ---------------------------
# MODELS
# ---------------------------
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def get_llm():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=150
    )

embedding_model = get_embedding_model()
llm = get_llm()

# ---------------------------
# LOAD FAQ DATA
# ---------------------------
faq_items = load_faq_data()
chunks = [item["text"] for item in faq_items]

# ---------------------------
# EMBEDDINGS
# ---------------------------
@st.cache_data
def embed_chunks(texts):
    embeddings = embedding_model.embed_documents(texts)
    return np.array(embeddings).astype("float32")

chunk_embeddings = embed_chunks(chunks)

# ---------------------------
# FAISS INDEX
# ---------------------------
@st.cache_resource
def build_faiss_index(embeddings):
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

faiss_index = build_faiss_index(chunk_embeddings)

# ---------------------------
# SUGGESTED QUESTIONS UI
# ---------------------------
st.write("💡 **Suggested questions:**")

suggested_questions = [
    "What is your return policy?",
    "How long does delivery take?",
    "Do you offer international shipping?",
    "What payment methods are accepted?",
    "What if I receive a damaged product?",
    "Is there a warranty on products?"
]

cols = st.columns(3)
for i, q in enumerate(suggested_questions):
    if cols[i % 3].button(q):
        st.session_state["question"] = q

st.write("")

# ---------------------------
# USER INPUT
# ---------------------------
user_input = st.text_input(
    "❓ Your question",
    value=st.session_state["question"]
)

# ---------------------------
# RAG ANSWER LOGIC
# ---------------------------
if user_input:
    query_embedding = embedding_model.embed_query(user_input)
    query_vector = np.array(query_embedding).astype("float32").reshape(1, -1)
    faiss.normalize_L2(query_vector)

    K = 5
    scores, indices = faiss_index.search(query_vector, K)

    top_score = scores[0][0]
    top_indices = indices[0]

    # Debug similarity score
    #st.caption(f"🔍 Similarity score: {top_score:.3f}")

    # Soft confidence handling
    HIGH_CONFIDENCE = 0.55
    LOW_CONFIDENCE = 0.35

    if top_score < LOW_CONFIDENCE:
        st.warning("🤔 I don’t have enough information to answer that.")
        save_user_query(user_input, "No confident answer", top_score)
        st.stop()

    elif top_score < HIGH_CONFIDENCE:
        st.info("⚠️ Answer is based on limited information.")

    retrieved_chunks = [chunks[i] for i in top_indices]
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
Use ONLY the following FAQ content to answer the question.
If the answer is not present, say you don't know.

FAQ Content:
{context}

Question: {user_input}
Answer:
"""

    result = llm(prompt)[0]["generated_text"]
    answer = result.strip()

    st.markdown("### 📘 Answer")
    st.success(answer)

    # Save user interaction
    save_user_query(user_input, answer, top_score)

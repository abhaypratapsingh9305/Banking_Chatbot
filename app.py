import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(page_title="Banking Chatbot", page_icon="💬", layout="centered")

# FAQ Knowledge Base
faq_knowledge_base = [
    {"id": 1, "question": "How do I check my balance?", "answer": "You can check your balance via the mobile app or at any ATM."},
    {"id": 2, "question": "What is the interest rate on savings?", "answer": "Our current savings rate is 2.5% APY."},
    {"id": 3, "question": "I lost my credit card.", "answer": "Please call 1-800-LOST-CARD immediately to freeze your account."},
    {"id": 4, "question": "Can I apply for a loan online?", "answer": "Yes, visit the 'Loans' section of our website to apply."},
    {"id": 5, "question": "What are your opening hours?", "answer": "We are open Mon-Fri from 9 AM to 5 PM."},
    {"id": 6, "question": "How do I change my PIN?", "answer": "You can change your PIN at any branch or ATM."},
    {"id": 7, "question": "Are my deposits insured?", "answer": "Yes, all deposits are insured up to $250,000."},
    {"id": 8, "question": "Do you offer student accounts?", "answer": "Yes, we have a fee-free student checking account."},
    {"id": 9, "question": "What is the routing number?", "answer": "Our routing number is 123456789 for direct deposits."},
    {"id": 10, "question": "How do I report fraud?", "answer": "Contact our fraud department immediately at 1-800-NO-FRAUD."}
]

# Load Model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Precompute FAQ embeddings
faq_questions = [item["question"] for item in faq_knowledge_base]
faq_embeddings = model.encode(faq_questions)

# Function to find the best match
def find_best_match(user_query):
    query_embedding = model.encode([user_query])
    similarities = cosine_similarity(query_embedding, faq_embeddings)[0]
    best_index = np.argmax(similarities)
    return faq_knowledge_base[best_index]

# ---- UI ----
st.title("💬 Banking FAQ Chatbot")
st.write("Ask me anything about banking!")

user_input = st.text_input("Your Query:")

if st.button("Ask"):
    if user_input.strip() != "":
        result = find_best_match(user_input)

        st.success(f"**Answer:** {result['answer']}")

        st.write("---")
        st.write(f"**Matched FAQ ID:** {result['id']}")
        st.write(f"**FAQ Question:** {result['question']}")

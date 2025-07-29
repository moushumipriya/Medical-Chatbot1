import streamlit as st
from utils import load_and_split_pdf, create_vector_store
from rag_bot import generate_answer

st.title("🩺 Medical Chatbot")
st.write("Ask a medical question related to your PDF content.")

user_question = st.text_input("💬 Your question:")

if user_question:
    try:
        docs = load_and_split_pdf("The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf")
        vector_store = create_vector_store(docs)
        answer = generate_answer(user_question, vector_store)
        st.markdown(f"🤖 **Answer:**\n\n{answer}")
    except Exception as e:
        st.error(f"❌ Failed to generate answer: {e}")

import os
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()
API_TOKEN = os.getenv("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")

generator = pipeline("text-generation", 
                     model="openai-community/gpt2",  # মেডিকেল model চাইলে অন্যটা দাও
                     token=API_TOKEN)

def generate_answer(user_question, vector_store):
    docs = vector_store.similarity_search(user_question, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    prompt = f"""You are a helpful medical assistant. Use the following context to answer the question.

Context:
{context}

Question: {user_question}
Answer:
"""
    result = generator(prompt, max_length=256, do_sample=True)[0]["generated_text"]
    return result

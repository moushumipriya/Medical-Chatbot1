from huggingface_hub import InferenceClient

client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    token="hf_klAcDilhdMbcMROaJmQGReIvQokXxvrYe")# তোমার টোকেন এখানে দাও


prompt = """
You are a helpful medical assistant. Answer this question:

Question: What is cancer?
Answer:
"""

response = client.text_generation(prompt, max_new_tokens=200)
print(response)

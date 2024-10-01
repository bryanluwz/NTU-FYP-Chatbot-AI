import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import torch

from src.api.typings import ResponseModel, RequestModel

app = FastAPI()


# Load models at startup
# For scalability, consider lazy loading or using model managers
models = {}


@app.on_event("startup")
def load_models():
    # Example: Load multiple models
    model_names = ["gpt2", "distilgpt2"]  # Replace with desired models
    for name in model_names:
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name)
        models[name] = {"tokenizer": tokenizer, "model": model}
    print("All models loaded.")


# Function to get vector embedding (assuming you have a method)
def get_query_vector(query):
    embedder = SentenceTransformer(
        'all-MiniLM-L6-v2')  # Choose appropriate model
    return embedder.encode(query).tolist()


# Endpoint for chatbot
@app.post("/chat", response_model=ResponseModel)
def chat(query: RequestModel):
    user_input = query.user_input
    model_name = query.model_name

    if model_name not in models:
        raise HTTPException(status_code=400, detail="Model not supported")

    # Step 1: Get query vector
    query_vector = get_query_vector(user_input)

    # Step 2: Retrieve relevant vectors from Express API
    retrieval_response = requests.post("http://localhost:3000/retrieve", json={
        "queryVector": query_vector,
        "topK": 5
    })

    if retrieval_response.status_code != 200:
        raise HTTPException(status_code=500, detail="Retrieval failed")

    retrieved_data = retrieval_response.json().get("results", [])

    # Step 3: Prepare context for the LLM
    context = " ".join([item['text']
                       for item in retrieved_data])  # Assuming 'text' field

    # Step 4: Generate response using the LLM
    tokenizer = models[model_name]['tokenizer']
    model = models[model_name]['model']

    input_text = f"Context: {context}\nUser: {user_input}\nAI:"
    inputs = tokenizer.encode(input_text, return_tensors='pt')

    # Generate response (adjust parameters as needed)
    outputs = model.generate(inputs, max_length=500,
                             do_sample=True, top_p=0.95, top_k=60)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the AI response part
    ai_response = generated_text.split("AI:")[-1].strip()

    return ResponseModel(response=ai_response)

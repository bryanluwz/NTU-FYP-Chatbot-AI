"""
This file contains all the RAG model stuffs
"""

from src.RAG_Model import RAG_Model

import os
from dotenv import dotenv_values
import torch

# Set torch to use the GPU memory at 80% capacity
if torch.cuda.is_available():
    print("[+] GPU found lessgoo..., setting memory fraction to 80%")
    torch.cuda.set_per_process_memory_fraction(0.8)


# Config
config = dotenv_values(".env")

MODEL_NAME = 'meta-llama/Llama-3.2-1B-Instruct'
CROSS_ENCODER_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
TASK = 'text-generation'

EMBEDDING_NAME = 'sentence-transformers/paraphrase-MiniLM-L6-v2'

EMBEDDING_MODEL_PATH = os.path.abspath(config.get(
    'EMBEDDING_MODEL_PATH', './models/embedding_model'))
os.makedirs(EMBEDDING_MODEL_PATH, exist_ok=True)

VECTOR_STORE_PATH = os.path.abspath(config.get(
    'VECTOR_STORE_PATH', './models/vector_store'))
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

LLM_MODEL_PATH = os.path.abspath(
    config.get('LLM_MODEL_PATH', './models/model'))
os.makedirs(LLM_MODEL_PATH, exist_ok=True)

CROSS_ENCODER_MODEL_PATH = os.path.abspath(
    config.get('CROSS_ENCODER_MODEL_PATH', './models/cross_encoder'))
os.makedirs(CROSS_ENCODER_MODEL_PATH, exist_ok=True)

print("[!] Embedding Model path:", EMBEDDING_MODEL_PATH)
print("[!] Vector Store path:", VECTOR_STORE_PATH)
print("[!] LLM Model path:", LLM_MODEL_PATH)
print("[!] Cross Encoder Model path:", CROSS_ENCODER_MODEL_PATH)

DOCUMENT_PARENT_DIR_PATH = os.path.abspath(
    config.get('DOCUMENT_DIR_PATH', './documents'))
DOCUMENT_DIR_NAME = None

os.makedirs(DOCUMENT_PARENT_DIR_PATH, exist_ok=True)


def create_rag_model(debug=False):
    qa_model = RAG_Model(debug=debug)
    qa_model.load_embeddings_model(EMBEDDING_NAME, EMBEDDING_MODEL_PATH)
    qa_model.initialize_cross_encoder(
        model_name=CROSS_ENCODER_NAME, model_path=CROSS_ENCODER_MODEL_PATH)
    qa_model.initialize_llm(model_name=MODEL_NAME,
                            max_new_tokens=512, model_path=LLM_MODEL_PATH, temperature=0.5, task=TASK)
    return qa_model

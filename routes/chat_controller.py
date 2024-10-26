from flask import request, jsonify
from src.QA_Chain import QA_Chain
from src.LLM_Chain import LLM_Chain
from src.Multiple_Chain import Multi_Chain
from src.Multiple_Chain_QA import Multi_Chain_QA

import os
from dotenv import dotenv_values
import torch
import zipfile


"""
Initialise the model
"""
# Set torch to use the GPU memory at 80% capacity
if torch.cuda.is_available():
    print("[+] GPU found lessgoo..., setting memory fraction to 80%")
    torch.cuda.set_per_process_memory_fraction(0.8)

config = dotenv_values(".env")

MODEL_NAME = 'meta-llama/Llama-3.2-1B-Instruct'
# MODEL_NAME = 'meta-llama/Llama-3.2-1B'
# MODEL_NAME = 'google/flan-t5-large'
# MODEL_NAME = 'distilgpt2'
# MODEL_NAME = 'deepset/roberta-base-squad2'
# TASK = 'question-answering'
TASK = 'text-generation'
# TASK = 'text2text-generation'

# SUMMARIZER_MODEL_NAME = 'google/flan-t5-base'
# SUMMARIZER_TASK = 'text2text-generation'
# SUMMARIZER_MODEL_NAME = 'meta-llama/Llama-3.2-1B'
SUMMARIZER_MODEL_NAME = 'meta-llama/Llama-3.2-1B-Instruct'
SUMMARIZER_TASK = 'text-generation'

EMBEDDING_NAME = 'sentence-transformers/paraphrase-MiniLM-L6-v2'

EMBEDDING_MODEL_PATH = os.path.abspath(config.get(
    'EMBEDDING_MODEL_PATH', './models/embedding_model'))

VECTOR_STORE_PATH = os.path.abspath(config.get(
    'VECTOR_STORE_PATH', './models/vector_store'))

LLM_MODEL_PATH = os.path.abspath(
    config.get('LLM_MODEL_PATH', './models/model'))

print("[!] Embedding model path:", EMBEDDING_MODEL_PATH)
print("[!] Vector store path:", VECTOR_STORE_PATH)
print("[!] LLM Model path:", LLM_MODEL_PATH)

DOCUMENT_PARENT_DIR_PATH = os.path.abspath(
    config.get('DOCUMENT_DIR_PATH', './documents'))
DOCUMENT_DIR_NAME = None

os.makedirs(DOCUMENT_PARENT_DIR_PATH, exist_ok=True)


def get_document_dir_path(document_parent_dir_path=DOCUMENT_PARENT_DIR_PATH, document_dir_name=DOCUMENT_DIR_NAME):
    if document_parent_dir_path is None:
        raise ValueError("DOCUMENT_DIR_NAME is not set")

    if document_dir_name is None:
        raise ValueError("DOCUMENT_PARENT_DIR_PATH is not set")

    return (os.path.join(
        document_parent_dir_path, document_dir_name))


def list_all_files(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


# Intialise the QA Chain, but don't load the vector store and init qa chain
qa_chain = Multi_Chain(debug=True)
qa_chain.load_embeddings_model(EMBEDDING_NAME, EMBEDDING_MODEL_PATH)
qa_chain.initialize_llm(model_name=MODEL_NAME,
                        max_new_tokens=512, model_path=LLM_MODEL_PATH, temperature=0.5, task=TASK)

qa_chain.initialize_summarizer(
    model_name=SUMMARIZER_MODEL_NAME, model_path=LLM_MODEL_PATH, task=SUMMARIZER_TASK)


"""
Controlller functions
We are ass uming that the embedding model and the model name is always the same
"""


def query():
    data = request.get_json()
    user_message = data.get('message', '')
    document_src_name = data.get('personaId', None)
    chat_history = data.get('chatHistory', None)

    if document_src_name is None:
        raise ValueError("documentSrc is not set")

    document_dir_name = document_src_name
    vector_store_path = os.path.join(
        VECTOR_STORE_PATH, f"{document_dir_name}_{qa_chain.embeddings.model_name}")

    is_new_vector_store = False

    # Load vector store when necessary
    if qa_chain.vector_store is None or qa_chain.vector_store_path is None or qa_chain.vector_store_path != vector_store_path:
        print("[!] Loading vector store...")

        print("[!] Vector store path:", vector_store_path)

        document_dir_path = get_document_dir_path(
            document_dir_name=document_dir_name)

        # Check if document_dir_path exist, if not exist, return "document_dne", request for document.zip trf
        if not os.path.exists(document_dir_path):
            return jsonify({
                "status": {
                    "code": 201},
                "success": False,
                'data': {
                    'response': f"Document source does not exist. Please contact admin. >:("}
            })

        file_paths = list_all_files(document_dir_path)
        file_paths_abs = [os.path.abspath(file_path)
                          for file_path in file_paths]
        qa_chain.load_vector_store(vector_store_path, file_paths_abs)

        is_new_vector_store = True
    else:
        print("[!] Vector store already loaded, skipping...")

    # Initialize the QA Chain when necessary
    if isinstance(qa_chain, LLM_Chain):
        if qa_chain.qa_chain is None or is_new_vector_store:
            print("[!] Initializing QA Chain...")
            qa_chain.initialize_qa_chain(top_k=4)
        else:
            print("[!] QA Chain already initialized, skipping...")
    elif isinstance(qa_chain, Multi_Chain):
        if qa_chain.qa_chain is None or is_new_vector_store:
            print("[!] Initializing QA Chain...")
            qa_chain.initialize_qa_chain(top_k=2)
        else:
            print("[!] QA Chain already initialized, skipping...")

    # bot_response = qa_chain.query(
    #     user_message, chat_history, chat_history_truncate_num=0)
    # answer = bot_response['answer']

    answer = qa_chain.query_without_pipeline(
        user_message, chat_history, chat_history_truncate_num=4)

    return jsonify({
        "success": True,
        "data": {
            'response': answer}})


def transferDocumentSrc():
    """Upload and transfer documents from BE server to this server"""
    data = request.form
    personaId = data.get('personaId', None)

    print(request.files, data)

    if 'documentSrc' not in request.files:
        return jsonify({
            "status": {
                "code": 400},
            "success": False,
            "data": {
                'response': 'No file part'}})

    file = request.files['documentSrc']

    if file.filename == '':
        return jsonify({
            "status": {
                "code": 400},
            "success": False,
            "data": {
                'response': 'No selected file'}})

    if file:
        document_src_name = personaId
        dir_name = get_document_dir_path(document_dir_name=document_src_name)
        os.makedirs(dir_name, exist_ok=True)
        filename = os.path.join(dir_name, file.filename)
        file.save(filename)

        unzip_dir = os.path.dirname(filename)
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(unzip_dir)

        return jsonify({
            "status": {
                "code": 200},
            "success": True,
            "data": {
                'response': 'Document uploaded successfully'}})

    return jsonify({
        "status": {
            "code": 400},
        "success": False,
        'data': {
            'response': 'Document upload failed'}})

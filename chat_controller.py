from flask import request, jsonify, json
from src.functions import list_all_files
import shutil

import os
import zipfile

from model import create_rag_model, DOCUMENT_PARENT_DIR_PATH, DOCUMENT_DIR_NAME, VECTOR_STORE_PATH

from dotenv import dotenv_values
config = dotenv_values(".env")

"""
Initialise some stuff
"""

TEMP_STORAGE_PATH = os.path.abspath(
    config.get('TEMP_STORAGE_PATH', './temp_storage'))
os.makedirs(TEMP_STORAGE_PATH, exist_ok=True)


"""
Helper functions
"""


def get_document_dir_path(document_parent_dir_path=DOCUMENT_PARENT_DIR_PATH, document_dir_name=DOCUMENT_DIR_NAME):
    if document_parent_dir_path is None:
        raise ValueError("DOCUMENT_DIR_NAME is not set")

    if document_dir_name is None:
        raise ValueError("DOCUMENT_PARENT_DIR_PATH is not set")

    return (os.path.join(
        document_parent_dir_path, document_dir_name))


"""
Controlller functions
"""

global qa_model
qa_model = None


def query():
    # Get info
    message_info = json.loads(request.form.get("messageInfo"))

    user_message = message_info['messageText']
    document_src_name = message_info['personaId']
    chat_history = message_info['chatHistory']

    # Files of images and documents
    files = request.files.getlist("files")
    filepaths = []

    # Save files to temp storage
    for file in files:
        file.save(os.path.join(TEMP_STORAGE_PATH, file.filename))
        filepaths.append(os.path.join(TEMP_STORAGE_PATH, file.filename))
        print("[+] File saved to temp storage:", file.filename)

    # Create RAG model if none
    global qa_model

    if qa_model is None:
        print("[!] Creating RAG model beep beep boop...")
        qa_model = create_rag_model(debug=True)

    if document_src_name is None:
        raise ValueError("documentSrc is not set")

    document_dir_name = document_src_name
    vector_store_path = os.path.join(
        VECTOR_STORE_PATH, f"{document_dir_name}_{qa_model.embeddings.model_name}")

    # Load vector store when necessary
    if qa_model.vector_store is None or qa_model.vector_store_path is None or qa_model.vector_store_path != vector_store_path:
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
        qa_model.load_vector_store(vector_store_path, file_paths_abs)
    else:
        print("[!] Vector store already loaded, skipping...")

    # Here should also pass the files too
    answer = qa_model.query(
        user_message, chat_history, chat_history_truncate_num=4, attached_file_paths=filepaths)

    # If no error (hopefully), remove temp files, before returning answer
    for filepath in filepaths:
        os.remove(filepath)
        pass

    return jsonify({
        "success": True,
        "data": {
            'response': answer}})


def transferDocumentSrc():
    """Upload and transfer documents from BE server to this server"""
    data = request.form
    personaId = data.get('personaId', None)

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

        # Clear directory and create new one
        os.makedirs(dir_name, exist_ok=True)
        shutil.rmtree(dir_name)
        os.makedirs(dir_name, exist_ok=True)

        filename = os.path.join(dir_name, file.filename)
        file.save(filename)

        unzip_dir = os.path.dirname(filename)
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(unzip_dir)

        # Clear vector store path if it exists
        global qa_model

        if qa_model is None:
            print("[!] Creating RAG model beep beep boop...")
            qa_model = create_rag_model(debug=True)

        vector_store_path = os.path.join(
            VECTOR_STORE_PATH, f"{document_src_name}_{qa_model.embeddings.model_name}")
        shutil.rmtree(vector_store_path, ignore_errors=True)

        # Stonks
        print("[+] Document uploaded successfully")

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

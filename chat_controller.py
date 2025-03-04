from flask import after_this_request, request, jsonify, json, send_file
import shutil
import os
import zipfile
from dotenv import dotenv_values

from src.functions import list_all_files
from model import create_rag_model, DOCUMENT_PARENT_DIR_PATH, DOCUMENT_DIR_NAME, VECTOR_STORE_PATH, create_stt_model, create_tts_model
from src.TTS_Model_API import TTS_Model_Map as TTS_API_Model_Map

config = dotenv_values(".env")
chat_config = {"debug": False, "api_mode": False}


"""
Initialise some stuff
"""

TEMP_STORAGE_PATH = os.path.abspath(
    config.get('TEMP_STORAGE_PATH', './temp_storage'))
os.makedirs(TEMP_STORAGE_PATH, exist_ok=True)

TTS_MODEL_PATH = os.path.abspath(
    config.get('TTS_MODEL_PATH', './models/tts_model'))
os.makedirs(TTS_MODEL_PATH, exist_ok=True)


"""
Helper functions
"""


def init_chat_config(init_config: dict):
    """
    Update config dict based on the passed config (only update the config keys that are present in the passed config)
    """
    for key in init_config:
        if key in chat_config:
            chat_config[key] = init_config[key]


def get_document_dir_path(document_parent_dir_path=DOCUMENT_PARENT_DIR_PATH, document_dir_name=DOCUMENT_DIR_NAME):
    if document_parent_dir_path is None:
        raise ValueError("DOCUMENT_DIR_NAME is not set")

    if document_dir_name is None:
        raise ValueError("DOCUMENT_PARENT_DIR_PATH is not set")

    return (os.path.normpath(os.path.join(
        document_parent_dir_path, document_dir_name)))


"""
Global variables
"""
global qa_model
qa_model = None

global tts_model
tts_model = None

global stt_model
stt_model = None


"""
Controlller functions
"""


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
        print(
            f"[!] Creating RAG model... Debug: {chat_config.get('debug', False)}, API Mode: {chat_config.get('api_mode', False)}")
        qa_model = create_rag_model(debug=chat_config.get(
            "debug", False), api_mode=chat_config.get("api_mode", False))

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
    answer, image_paths = qa_model.query(
        user_message, chat_history, attached_file_paths=filepaths)

    # If no error (hopefully), remove temp files, before returning answer
    for filepath in filepaths:
        os.remove(filepath)
        pass

    return jsonify(
        {"status": {
            "code": 400},
            "data": {
            'response': answer, 'image_paths': image_paths}})


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
            print(
                f"[!] Creating RAG model... Debug: {chat_config.get('debug', False)}, API Mode: {chat_config.get('api_mode', False)}")
            qa_model = create_rag_model(debug=chat_config.get(
                "debug", False), api_mode=chat_config.get("api_mode", False))

        vector_store_path = os.path.join(
            VECTOR_STORE_PATH, f"{document_src_name}_{qa_model.embeddings.model_name}")
        shutil.rmtree(vector_store_path, ignore_errors=True)

        # Force create new vector store
        qa_model.load_vector_store(vector_store_path, [os.path.abspath(
            file_path) for file_path in list_all_files(dir_name)])

        # For some reason it would be better to unload the vector store here
        qa_model.vector_store = None

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


def tts():
    # Get info
    text = request.form.get("text")
    voice = request.form.get("ttsName") or 'default'

    # Create TTS model if none
    global tts_model

    if tts_model is None:
        print(
            f"[!] Creating TTS model... Debug: {chat_config.get('debug', False)}, API Mode: {chat_config.get('api_mode', False)}")
        tts_model = create_tts_model(debug=chat_config.get(
            "debug", False), api_mode=chat_config.get("api_mode", False))

    # TTS
    file_path = tts_model.tts(voice, text)

    @after_this_request
    def remove_file(response):
        try:
            # TODO: Remove file, currently not working cause file is still in use for some reason
            os.remove(file_path)
        except Exception as e:
            print(f"Error removing file: {e}")
        return response

    return send_file(file_path, as_attachment=True)


def stt():
    # Get info
    audio = request.files.get("audio")
    if audio is None:
        return jsonify({
            "success": False,
            "data": {
                'response': 'No audio file'}})

    audio = audio.stream.read()

    # Create STT model if none
    global stt_model

    if stt_model is None:
        print(
            f"[!] Creating STT model... Debug: {chat_config.get('debug', False)}, API Mode: {chat_config.get('api_mode', False)}")
        stt_model = create_stt_model(debug=chat_config.get(
            "debug", False), api_mode=chat_config.get("api_mode", False))

    # STT
    text = stt_model.stt(audio)

    return jsonify({
        "success": True,
        "data": {
            'response': text}})


def query_voices():
    api_voices = TTS_API_Model_Map.model_map

    voices = api_voices.keys()

    return jsonify({
        "success": True,
        "data": {
            'response': list(voices)}})


def post_query_image():
    filename = request.form.get("filename")
    filename = filename.replace("\\", os.sep)  # stupid thing waste my hour
    try:
        return send_file(os.path.join(filename), as_attachment=True)
    except Exception as e:
        # Try to remove the 'app' part
        try:
            new_path = filename.replace("app/", "")
            return send_file(new_path, as_attachment=True)
        except Exception as f:
            raise Exception(new_path)

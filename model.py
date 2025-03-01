"""
This file contains all the RAG model stuffs
"""

from src.RAG_Model import RAG_Model
from src.RAG_Model_API import RAG_Model_API
from src.STT_Model import STT_Model
from src.TTS_Model import TTS_Model

import os
from dotenv import dotenv_values
import torch

# Set torch to use the GPU memory at 80% capacity
if torch.cuda.is_available():
    print("[+] GPU found lessgoo..., setting memory fraction to 80%")
    torch.cuda.set_per_process_memory_fraction(0.8)
else:
    print("[!] GPU not found, using CPU... sadge")


# Config
config = dotenv_values(".env")

# Only local mode
LLM_MODEL_NAME = os.getenv(
    "LLM_MODEL", "meta-llama/Llama-3.2-1B-Instruct")  # Default if not set
LLM_MODEL_NAME_API = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
LLM_TASK = 'text-generation'

BLIP_MODEL_NAME = 'Salesforce/blip-image-captioning-base'
BLIP_TASK = 'image-to-text'

CROSS_ENCODER_NAME = 'cross-encoder/ms-marco-MiniLM-L-12-v2'

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

BLIP_MODEL_PATH = os.path.abspath(
    config.get('BLIP_MODEL_PATH', './models/blip_model'))
os.makedirs(BLIP_MODEL_PATH, exist_ok=True)

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


HUGGINGFACE_TOKEN = None
if os.path.exists("/run/secrets/hf_token"):
    with open("/run/secrets/hf_token") as f:
        print("[+] Found Huggingface token in secrets: /run/secrets/hf_token")
        HUGGINGFACE_TOKEN = f.read().strip()
else:
    print("[!] Huggingface token not found in secrets: /run/secrets/hf_token, using the .env file")
    HUGGINGFACE_TOKEN = config.get('HUGGINGFACE_TOKEN', None)

TOGETHER_API_KEY = None
if os.path.exists("/run/secrets/together_api_key"):
    with open("/run/secrets/together_api_key") as f:
        print("[+] Found Together API key in secrets: /run/secrets/together_api_key")
        TOGETHER_API_KEY = f.read().strip()
else:
    print("[!] Together API key not found in secrets: /run/secrets/together_api_key, using the .env file")
    TOGETHER_API_KEY = config.get('TOGETHER_API_KEY', None)

AZURE_API_ENDPOINT = None
if os.path.exists("/run/secrets/azure_api_endpoint"):
    with open("/run/secrets/azure_api_endpoint") as f:
        print("[+] Found Azure API endpoint in secrets: /run/secrets/azure_api_endpoint")
        AZURE_API_ENDPOINT = f.read().strip()
else:
    print("[!] Azure API endpoint not found in secrets: /run/secrets/azure_api_endpoint, using the .env file")
    AZURE_API_ENDPOINT = config.get('AZURE_AI_ENDPOINT', None)

AZURE_API_KEY = None
if os.path.exists("/run/secrets/azure_api_key"):
    with open("/run/secrets/azure_api_key") as f:
        print("[+] Found Azure API key in secrets: /run/secrets/azure_api_key")
        AZURE_API_KEY = f.read().strip()
else:
    print("[!] Azure API key not found in secrets: /run/secrets/azure_api_key, using the .env file")
    AZURE_API_KEY = config.get('AZURE_AI_API_KEY', None)


def create_rag_model(debug=False, api_mode=False):
    if not api_mode:
        print("[!] API mode is not enabled, using the LLM module in local mode")
        qa_model = RAG_Model(debug=debug, huggingface_token=HUGGINGFACE_TOKEN)
        qa_model.load_embeddings_model(EMBEDDING_NAME, EMBEDDING_MODEL_PATH)
        qa_model.initialize_cross_encoder(
            model_name=CROSS_ENCODER_NAME, model_path=CROSS_ENCODER_MODEL_PATH)
        qa_model.initialize_llm(model_name=LLM_MODEL_NAME,
                                max_new_tokens=512, model_path=LLM_MODEL_PATH, temperature=0.8, task=LLM_TASK)
        qa_model.intialize_image_pipeline(model_name=BLIP_MODEL_NAME,
                                          model_path=BLIP_MODEL_PATH, task=BLIP_TASK)
        return qa_model
    else:
        print("[!] API mode is enabled, using the LLM module in API mode")
        qa_model = RAG_Model_API(
            debug=debug, together_api_key=TOGETHER_API_KEY, azure_api_key=AZURE_API_KEY, azure_api_endpoint=AZURE_API_ENDPOINT)
        qa_model.load_embeddings_model(EMBEDDING_NAME, EMBEDDING_MODEL_PATH)
        qa_model.initialize_cross_encoder(
            model_name=CROSS_ENCODER_NAME, model_path=CROSS_ENCODER_MODEL_PATH)
        qa_model.initialize_llm(model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                                max_new_tokens=512, temperature=0.8)
        qa_model.intialize_image_pipeline()
        return qa_model


def create_tts_model(debug=False):
    tts_model = TTS_Model(debug=debug)
    return tts_model


STT_MODEL_PATH = os.path.abspath(
    config.get('STT_MODEL_PATH', './models/stt_model'))
os.makedirs(STT_MODEL_PATH, exist_ok=True)

STT_MODEL_NAME = 'openai/whisper-small.en'


def create_stt_model(debug=False):
    stt_model = STT_Model(debug=debug)
    stt_model.initialise_stt(STT_MODEL_NAME, STT_MODEL_PATH)
    return stt_model

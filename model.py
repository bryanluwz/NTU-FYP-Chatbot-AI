"""
This file contains all the RAG model stuffs
"""
import os
from dotenv import dotenv_values
from src.RAG_Model_API import RAG_Model_API
from src.TTS_Model_API import TTS_Model_API
from src.STT_Model_API import STT_Model_API

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

EMBEDDING_NAME = 'text-embedding-ada-002'

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
TOGETHER_API_KEY = None

if os.path.exists("/run/secrets/rag_tokens"):
    with open("/run/secrets/rag_tokens") as f:
        print("[+] Found RAG tokens in secrets: /run/secrets/rag_tokens")
        lines = f.readlines()
        if len(lines) >= 2:
            HUGGINGFACE_TOKEN = lines[0].strip()
            TOGETHER_API_KEY = lines[1].strip()
        else:
            raise Exception(
                "[!] RAG tokens file is malformed: /run/secrets/rag_tokens")
else:
    print("[!] RAG tokens not found in secrets: /run/secrets/rag_tokens, using the .env file")
    HUGGINGFACE_TOKEN = config.get('HUGGINGFACE_TOKEN', None)
    TOGETHER_API_KEY = config.get('TOGETHER_API_KEY', None)

AZURE_API_ENDPOINT = None
AZURE_API_KEY = None
AZURE_OPENAI_ENDPOINT = None
AZURE_OPENAI_API_KEY = None

if os.path.exists("/run/secrets/azure_api"):
    with open("/run/secrets/azure_api") as f:
        print("[+] Found Azure API credentials in secrets: /run/secrets/azure_api")
        lines = f.readlines()
        if len(lines) >= 4:
            AZURE_API_ENDPOINT = lines[0].strip()
            AZURE_API_KEY = lines[1].strip()
            AZURE_OPENAI_ENDPOINT = lines[2].strip()
            AZURE_OPENAI_API_KEY = lines[3].strip()
        else:
            raise Exception(
                "[!] Azure API credentials file is malformed: /run/secrets/azure_api")
else:
    print("[!] Azure API credentials not found in secrets: /run/secrets/azure_api, using the .env file")
    AZURE_API_ENDPOINT = config.get('AZURE_API_ENDPOINT', None)
    AZURE_API_KEY = config.get('AZURE_API_KEY', None)
    AZURE_OPENAI_ENDPOINT = config.get('AZURE_OPENAI_ENDPOINT', None)
    AZURE_OPENAI_API_KEY = config.get('AZURE_OPENAI_API_KEY', None)


GOOGLE_CLOUD_API_KEY = None
# Actually should be a json file but i dont care
if os.path.exists("/run/secrets/google_cloud_api_key"):
    print(
        "[+] Found Google Cloud API key in secrets: /run/secrets/google_cloud_api_key")
    GOOGLE_CLOUD_API_KEY = "/run/secrets/google_cloud_api_key"
else:
    print("[!] Google Cloud API key not found in secrets: /run/secrets/google_cloud_api_key, using the secrets/google_cloud_api_key.json file")
    GOOGLE_CLOUD_API_KEY = os.path.abspath(
        '../secrets/google_cloud_api_key.json')
    # Check if the file exists
    if not os.path.exists(GOOGLE_CLOUD_API_KEY):
        GOOGLE_CLOUD_API_KEY = None
        raise Exception(
            "[!] Google Cloud API key file not found: secrets/google_cloud_api_key.json, please provide the key in the secrets folder or in the .env file")


def create_rag_model(debug=False, api_mode=False):
    print("[!] API mode is enabled, using the LLM module in API mode")
    qa_model = RAG_Model_API(
        debug=debug, together_api_key=TOGETHER_API_KEY, azure_api_key=AZURE_API_KEY,
        azure_api_endpoint=AZURE_API_ENDPOINT, azure_openai_api_key=AZURE_OPENAI_API_KEY,
        azure_openai_endpoint=AZURE_OPENAI_ENDPOINT)
    qa_model.load_embeddings_model(EMBEDDING_NAME)
    qa_model.initialize_llm(model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                            max_new_tokens=512, temperature=0.8)
    qa_model.initialize_image_pipeline()
    return qa_model


def create_tts_model(debug=False, api_mode=False):
    print("[!] API mode is enabled, using the TTS module in API mode")
    tts_model = TTS_Model_API(
        debug=debug, tts_api_key=GOOGLE_CLOUD_API_KEY)
    tts_model.initialise_tts()
    return tts_model


STT_MODEL_PATH = os.path.abspath(
    config.get('STT_MODEL_PATH', './models/stt_model'))
os.makedirs(STT_MODEL_PATH, exist_ok=True)

STT_MODEL_NAME = 'openai/whisper-small.en'


def create_stt_model(debug=False, api_mode=False):
    print("[!] API mode is enabled, using the STT module in API mode")
    stt_model = STT_Model_API(
        debug=debug, stt_api_key=GOOGLE_CLOUD_API_KEY)
    stt_model.initialise_stt()
    return stt_model

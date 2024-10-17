import os
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.retrieval import create_retrieval_chain
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
import gc
import torch

from .document_loader import load_documents


def load_embeddings_model(model_name: str = "paraphrase-MiniLM-L6-v2", embedding_model_path: str = "embedding_models"):
    """Initialize HuggingFace embeddings."""
    # Check if the model already exists in the cache
    local_model_path = os.path.join(
        embedding_model_path, model_name)

    os.makedirs(local_model_path, exist_ok=True)

    # Load the embeddings model from the cache directory or download it
    return HuggingFaceEmbeddings(model_name=model_name, show_progress=True, cache_folder=local_model_path)


def load_vector_store(embeddings: HuggingFaceEmbeddings = None, vector_store_path: str = "vector_store/<your_vector_store_name>"):
    """Load the FAISS vector store if it exists."""
    try:
        faiss_index_path = os.path.join(vector_store_path, "index.faiss")
        faiss_pkl_path = os.path.join(vector_store_path, "index.pkl")

        if os.path.exists(faiss_index_path) and os.path.exists(faiss_pkl_path):
            # Load persisted vector store
            persisted_vectorstore = FAISS.load_local(
                vector_store_path, embeddings, allow_dangerous_deserialization=True)
            print("✅ Loaded vector store from local storage.")
            return persisted_vectorstore
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        return None


def create_and_save_vector_store(embeddings, vector_store_path, file_paths):
    """Create a new FAISS vector store from the given PDF and save it."""
    print("⚠️ Creating a new vector store, if one already exists it will be overwritten.")

    if os.path.exists(vector_store_path):
        shutil.rmtree(vector_store_path)
        print("🗑️ Removed existing vector store.")

    os.makedirs(vector_store_path, exist_ok=True)

    # Load document using PyPDFLoader
    documents = load_documents(file_paths)

    # Split document into chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=30,
        separator="\n"
    )
    docs = text_splitter.split_documents(documents)

    # Create vectors using FAISS
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Persist the vectors locally on disk
    vectorstore.save_local(vector_store_path)
    print("💾 Vector store saved locally.")

    return vectorstore


def initialize_llm(model_name: str = 'distilgpt2', max_new_tokens: int = 1024, temperature: float = 0.7, model_path: str = None, device: int = -1):
    """Initialize the HuggingFace pipeline for text generation, and save/load the model."""
    model_save_path = os.path.join(model_path, model_name)

    # Check if the model is already saved
    if os.path.exists(model_save_path):
        print(f"🔄 Loading model from {model_save_path}...")
        text_gen_pipeline = pipeline(
            task="text-generation",
            model=model_save_path,
            tokenizer=model_save_path,
            max_new_tokens=max_new_tokens,
            framework="pt",
            device_map="balanced_low_0",
            return_full_text=False
        )
        # assert text_gen_pipeline.tokenizer.vocab_size == text_gen_pipeline.model.config.vocab_size, "Tokenizer and model vocab size mismatch."
    else:
        # Get the model size before downloading
        print(
            f"⬇️ Downloading and saving model '{model_name}' to {model_save_path}...")
        text_gen_pipeline = pipeline(
            task="text-generation",
            model=model_name,
            tokenizer=model_name,
            framework="pt",
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            device_map="balanced_low_0",
            return_full_text=False
        )
        # assert text_gen_pipeline.tokenizer.vocab_size == text_gen_pipeline.model.config.vocab_size, "Tokenizer and model vocab size mismatch."

        # Save the model and tokenizer
        text_gen_pipeline.model.save_pretrained(model_save_path)
        text_gen_pipeline.tokenizer.save_pretrained(model_save_path)
        print(f"✅ Model '{model_name}' saved to {model_save_path}.")

    return HuggingFacePipeline(pipeline=text_gen_pipeline)


def initialize_qa_chain(llm, vectorstore: FAISS, prompt_template: ChatPromptTemplate | None = None, top_k: int = 5):
    """Initialize the RetrievalQA chain with the given LLM and vectorstore."""
    if prompt_template is None:
        system_prompt = (
            "You are an teachin assistant at a university. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

    document_chain = create_stuff_documents_chain(
        llm, prompt_template)
    return create_retrieval_chain(vectorstore.as_retriever(search_kwargs={"k": top_k}), document_chain)


def unload_model(text_gen_pipeline):
    try:
        torch.cuda.empty_cache()
    except:
        pass

    # Delete the model and tokenizer
    del text_gen_pipeline.model
    del text_gen_pipeline.tokenizer

    # Delete the pipeline
    del text_gen_pipeline

    # Run garbage collection
    gc.collect()

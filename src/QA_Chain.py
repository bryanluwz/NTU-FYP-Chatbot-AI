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

from datetime import datetime


class QA_Chain:
    """
    Retrieval QA Chain for answering queries given context
    """

    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self.vector_store_path = None
        self.llm = None
        self.qa_chain = None
        self.text_gen_pipeline = None

    def load_embeddings_model(self, model_name: str = "paraphrase-MiniLM-L6-v2", embedding_model_path: str = "embedding_models"):
        """Initialize HuggingFace embeddings."""
        # Check if the model already exists in the cache
        local_model_path = os.path.join(
            embedding_model_path, model_name)

        os.makedirs(local_model_path, exist_ok=True)

        # Load the embeddings model from the cache directory or download it
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name, show_progress=True, cache_folder=local_model_path)

    def load_vector_store(self, vector_store_path: str = "vector_store/<your_vector_store_name>", file_paths=[]):
        """Load the FAISS vector store if it exists."""
        assert self.embeddings is not None, "Embeddings model not initialized."

        try:
            faiss_index_path = os.path.join(vector_store_path, "index.faiss")
            faiss_pkl_path = os.path.join(vector_store_path, "index.pkl")

            if os.path.exists(faiss_index_path) and os.path.exists(faiss_pkl_path):
                # Load persisted vector store
                persisted_vectorstore = FAISS.load_local(
                    vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
                print("✅ Loaded vector store from local storage.")
                self.vector_store = persisted_vectorstore

                self.vector_store_path = vector_store_path

            else:
                raise FileNotFoundError
        except FileNotFoundError:
            self.vector_store = None
            self.create_and_save_vector_store(
                vector_store_path, file_paths)

    def create_and_save_vector_store(self, vector_store_path, file_paths):
        """Create a new FAISS vector store from the given PDF and save it."""
        assert self.embeddings is not None, "Embeddings model not initialized."

        print(
            "⚠️ Creating a new vector store, if one already exists it will be overwritten.")

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
        vectorstore = FAISS.from_documents(docs, self.embeddings)

        # Persist the vectors locally on disk
        vectorstore.save_local(vector_store_path)
        print("💾 Vector store saved locally.")

        self.vector_store_path = vector_store_path
        self.vector_store = vectorstore

    def initialize_llm(self, model_name: str = 'distilgpt2', max_new_tokens: int = 1024, temperature: float = 0.7, model_path: str = None, device: int = -1):
        """Initialize the HuggingFace pipeline for text generation, and save/load the model."""
        model_save_path = os.path.join(model_path, model_name)

        if self.llm is not None:
            del self.llm
            gc.collect()

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

        self.text_gen_pipeline = text_gen_pipeline

        self.llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

    def initialize_qa_chain(self, prompt_template: ChatPromptTemplate | None = None, top_k: int = 5):
        """Initialize the RetrievalQA chain with the given LLM and vectorstore."""
        assert self.llm is not None, "LLM model not initialized."
        assert self.vector_store is not None, "Vector store not initialized."

        if self.qa_chain is not None:
            del self.qa_chain
            gc.collect()

        if prompt_template is None:
            system_prompt = (
                "You are an teaching assistant at a university. "
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
            self.llm, prompt_template)
        self.qa_chain = create_retrieval_chain(
            self.vector_store.as_retriever(search_kwargs={"k": top_k}), document_chain)

    def self_detox(self):
        try:
            torch.cuda.empty_cache()
        except:
            pass

        try:
            # Delete the model and tokenizer
            del self.text_gen_pipeline.model
            del self.text_gen_pipeline.tokenizer

            # Delete the pipeline
            del self.text_gen_pipeline
            del self.llm
            del self.qa_chain

            # Set all to None
            self.text_gen_pipeline = None
            self.llm = None
            self.qa_chain = None
            self.embeddings = None
            self.vector_store = None
            self.vector_store_path = None
        except Exception as e:
            print(f"[!] Error while destroying the QA chain: {e}")
            pass

        # Run garbage collection
        gc.collect()

    def yeet_qa_chain(self):
        """Delete the QA chain and free up memory."""
        if self.qa_chain is not None:
            del self.qa_chain
            self.qa_chain = None

        gc.collect

    def query(self, query):
        """Query the QA chain with the given input."""
        assert self.qa_chain is not None, "QA chain not initialized."
        timestamp = datetime.now().strftime("%Y%m%d %H%M%S")
        print(f"[{timestamp}] Querying the QA chain...")
        output = self.qa_chain.invoke({
            "input": query,
        })
        timestamp = datetime.now().strftime("%Y%m%d %H%M%S")
        print(f"[{timestamp}] Query complete.")
        return output

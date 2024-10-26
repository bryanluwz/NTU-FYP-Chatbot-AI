"""
This is the implementation of the RAG Chatbot model

(Beta release)
Known Issues:
- EVERYTHING!!! fja;fladkl;faj;dfjasldfj
"""

from datetime import datetime
import os
import shutil
from typing import List
from langchain_core.documents.base import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from transformers import pipeline
import torch

from .document_loader import load_documents


class RAG_Model_beta:
    def __init__(self, debug=False):
        self.embeddings = None
        self.vector_store = None
        self.vector_store_path = None

        self.llm_pipeline = None
        self.summarizer_pipeline = None

        self.document_reranker_pipeline = None

        self.debug = debug

    def _debug_print(self, *msg):
        """
        Print debug messages
        """
        if self.debug:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}]", *msg, end="\n\n")

    def _convert_to_pipeline_inputs(self, messages: dict[str, str]):
        """
        Convert ChatMessageModel to pipeline inputs
        """
        return [{
            "role": message['userType'],
            "content": message['message']
        } for message in messages]

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

    def initialize_llm(self, model_name: str = 'distilgpt2', max_new_tokens: int = 1024, temperature: float = 0.7, model_path: str = None, task: str = "text-generation"):
        """
        Initialize the pipeline for text generation, and save/load the model.
        Both the LLM and summarizer pipelines are the same.
        """
        model_save_path = os.path.join(model_path, model_name)

        # Check if the model is already saved
        if os.path.exists(model_save_path):
            print(f"🔄 Loading model from {model_save_path}...")
            text_gen_pipeline = pipeline(
                task=task,
                model=model_save_path,
                tokenizer=model_save_path,
                max_new_tokens=max_new_tokens,
                framework="pt",
                device_map="auto"
            )
        else:
            # Get the model size before downloading
            print(
                f"⬇️ Downloading and saving model '{model_name}' to {model_save_path}...")
            text_gen_pipeline = pipeline(
                task=task,
                model=model_name,
                tokenizer=model_name,
                framework="pt",
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                device_map="auto"
            )

            # Save the model and tokenizer
            text_gen_pipeline.model.save_pretrained(model_save_path)
            text_gen_pipeline.tokenizer.save_pretrained(model_save_path)
            print(f"✅ Model '{model_name}' saved to {model_save_path}.")

        self.llm_pipeline = text_gen_pipeline

        self.summarizer_pipeline = text_gen_pipeline

    def initialize_cross_encoder(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2', model_path: str = None):
        """
        Initialize the cross encoder model for ranking documents.
        """
        model_save_path = os.path.join(model_path, model_name)

        # Check if the model is already saved
        if os.path.exists(model_save_path):
            print(f"🔄 Loading cross-encoder model from {model_save_path}...")
            reranker_pipeline = pipeline(
                task="text-classification",
                model=model_save_path,
                tokenizer=model_save_path,
            )
        else:
            # Get the model size before downloading
            print(
                f"⬇️ Downloading and saving cross-encoder model '{model_name}' to {model_save_path}...")
            reranker_pipeline = pipeline(
                task="text-classification",
                model=model_name,
                tokenizer=model_name,
            )

            # Save the model
            reranker_pipeline.save_pretrained(model_save_path)
            print(
                f"✅ Cross-encoder model '{model_name}' saved to {model_save_path}.")

        self.document_reranker_pipeline = reranker_pipeline

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize the scores to a range of 0 to 1.
        """
        min_score = min(scores)
        max_score = max(scores)

        return [(score - min_score) / (max_score - min_score) for score in scores]

    def _rerank_documents(self, documents: list[Document], query: str, top_k=2, score_threshold=0.5):
        """
        Rerank the documents based on the relevance to the query.
        """
        assert self.document_reranker_pipeline is not None, "Cross-encoder model not initialized."

        # Prepare query-document pairs
        query_doc_pairs = [{"text": query, "text_pair": doc.page_content}
                           for doc in documents]

        # Get relevance scores
        scores = self.document_reranker_pipeline(
            query_doc_pairs)

        scores = self._normalize_scores([score["score"] for score in scores])

        # Combine scores with documents and sort by score, highest first
        scored_docs = [(doc, score)
                       for doc, score in zip(documents, scores) if score > score_threshold]

        self._debug_print(
            f"Scored documents: {[score for score in scores]}")

        reranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in reranked_docs][:top_k]

    def query(self, query: str, chat_history: list[any], chat_history_truncate_num=5, search_k=10, top_k=2):
        """
        Query the QA chain with the given input.

        Search for the most relevant documents based on the query and chat history.
        Return a bunch of documents based on the search results.
        Then, rank the documents based on relevance to the query and return the top-k documents.
        """
        assert self.vector_store is not None, "Vector store not initialized."
        assert self.llm_pipeline is not None, "LLM pipeline not initialized."
        assert self.summarizer_pipeline is not None, "Summarizer pipeline not initialized."
        assert self.document_reranker_pipeline is not None, "Cross-encoder model not initialized."

        # Get reformulated query based on chat history and user query
        truncated_chat_history = chat_history[:chat_history_truncate_num]

        self._debug_print("Querying the QA chain...")

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference prior context, rephrase the question "
            "as a standalone question understandable without prior context. "
            "Do NOT add new information, make assumptions, or create your own questions. "
            "If rephrasing is not possible without assumptions, return the question exactly as it is. "
        )

        reformulated_query = self.summarizer_pipeline(
            [
                {"role": "system", "content": contextualize_q_system_prompt},
                *self._convert_to_pipeline_inputs(truncated_chat_history),
                {"role": "human", "content": query}
            ]
        )[0]['generated_text'][-1]

        self._debug_print(f"Reformulated query: {reformulated_query}")

        # Check if the reformulated query need context or not
        # WIP or not idk

        # Get Context based on reformulated query
        context: List[Document] = self.vector_store.similarity_search(
            reformulated_query["content"], k=search_k)

        # Rank documents based on usefulness via Cross Encoder
        context_ranked = self._rerank_documents(
            context, query, top_k=top_k)

        # Restructure the final prompt passed into LLM
        system_prompt = f"""
You are a friendly teaching assistant at a university. 
Use the following pieces of retrieved context to answer the question as accurately as possible. 
Keep your responses concise, using a maximum of three sentences. 
1. Provide the best possible response based on the information available. 
2. If you're unsure, respond in a friendly manner and share any relevant insights you have.
3. If the question isn't clear or doesn't align with the context, feel free to provide your general knowledge while mentioning that you are doing so.
4. For greetings or casual conversation, respond warmly and engage in a friendly dialogue.
Context: {context_ranked}""".replace("\n", " ")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "human", "content": query}
        ]

        self._debug_print(f"Query Message: {messages}")

        output = self.llm_pipeline(
            messages
        )[0]['generated_text'][-1]

        self._debug_print(f"Generated answer: {output}")

        self._debug_print("Query complete.")

        return output["content"]

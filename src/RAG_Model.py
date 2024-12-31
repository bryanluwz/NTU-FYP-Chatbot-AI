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

from .document_loader import load_documents
from .functions import download_tesseract

from PIL import Image

from torch import cuda

import pytesseract


tesseract_path = download_tesseract(os.getcwd())
pytesseract.pytesseract.tesseract_cmd = tesseract_path


class RAG_Model:
    def __init__(self, debug=False, device=None):
        self.embeddings = None
        self.vector_store = None
        self.vector_store_path = None

        self.llm_pipeline = None
        self.summarizer_pipeline = None
        self.blip_pipeline = None

        self.document_reranker_pipeline = None

        self.debug = debug
        self.device = device or (1 if cuda.is_available() else 0)

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
        # TODO: Need to add support for images here that are in the documents
        documents = load_documents(
            file_paths, describe_image_callback=self._describe_image)

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

    def initialize_blip(self, model_name: str = 'Salesforce/blip-image-captioning-base', model_path: str = None, task: str = "image-to-text"):
        """
        Initialize the BLIP pipeline for image and text encoding.
        """
        model_save_path = os.path.join(model_path, model_name)

        # Check if the model is already saved
        # `device` cannot be set to 'auto' for BLIP, so we need to specify the device manually, if GPU cannot be used, fall back to CPU
        if os.path.exists(model_save_path):
            try:
                print(f"🔄 Loading BLIP model from {model_save_path}...")
                blip_pipeline = pipeline(
                    task=task,
                    model=model_save_path,
                    tokenizer=model_save_path,
                    framework="pt",
                    device=self.device  # Use self.device to specify the device
                )

            except RuntimeError as e:
                if "CUDA error: invalid device ordinal" in str(e):
                    print(
                        "⚠️ Idk there's some weird GPU device behaviour. Falling back to CPU.")
                    self.device = -1  # Use CPU
                    blip_pipeline = pipeline(
                        task=task,
                        model=model_save_path,
                        tokenizer=model_save_path,
                        framework="pt",
                        device=self.device  # Use CPU
                    )
                else:
                    raise e
        else:
            # Get the model size before downloading
            print(
                f"⬇️ Downloading and saving BLIP model '{model_name}' to {model_save_path}...")
            try:
                blip_pipeline = pipeline(
                    task=task,
                    model=model_name,
                    tokenizer=model_name,
                    framework="pt",
                    device=self.device  # Use self.device to specify the device
                )
            except RuntimeError as e:
                if "CUDA error: invalid device ordinal" in str(e):
                    print(
                        "⚠️ Idk there's some weird GPU device behaviour. Falling back to CPU.")
                    self.device = -1  # Use CPU
                    blip_pipeline = pipeline(
                        task=task,
                        model=model_name,
                        tokenizer=model_name,
                        framework="pt",
                        device=self.device  # Use CPU
                    )
                else:
                    raise e

            # Save the model
            blip_pipeline.save_pretrained(model_save_path)
            print(f"✅ BLIP model '{model_name}' saved to {model_save_path}.")

        self.blip_pipeline = blip_pipeline

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

    def _describe_images(self, image_paths: list[str]):
        """
        Describe the image using the BLIP model. (and also OCR)
        """
        assert self.blip_pipeline is not None, "BLIP pipeline not initialized."

        descs = []

        for image_path in image_paths:
            image_desc, ocr_text = self._describe_image(image_path)
            descs.append({'description': image_desc, 'ocr_text': ocr_text})

        return descs

    def _describe_image(self, image_path: str):
        """
        Describe the image using the BLIP model. (and also OCR)
        """
        assert self.blip_pipeline is not None, "BLIP pipeline not initialized."

        image = Image.open(image_path)

        # Get description
        image_description = self.blip_pipeline(image)
        image_description_text = image_description[0]['generated_text']

        # Get OCR text
        try:
            ocr_text = pytesseract.image_to_string(image_path)
        except Exception as e:
            print(f"❌ OCR failed (-_-): {e}")
            ocr_text = ""

        return {"description": image_description_text, "text": ocr_text}

    def _load_input_documents(self, file_paths: list[str]):
        """
        Load the input documents from the given file paths.
        """
        # Load document using PyPDFLoader
        documents = load_documents(
            file_paths, describe_image_callback=self._describe_image)

        # Split document into chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=30,
            separator="\n"
        )
        docs = text_splitter.split_documents(documents)

        return docs

    def query(self, query: str, chat_history: list[any], chat_history_truncate_num=5, search_k=10, top_k=2, attached_file_paths=[]):
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
            "As a friendly university teaching assistant, given a chat history and the latest user question "
            "which might reference prior context, rephrase the question as a standalone question understandable "
            "without prior context. Do NOT add new information, make assumptions, or create your own questions. "
            "If rephrasing is not possible without assumptions, return the question exactly as it is."
            "Do NOT answer the question. Just rephrase it."
        )

        reformulated_query = self.summarizer_pipeline(
            [
                {"role": "system", "content": contextualize_q_system_prompt},
                *self._convert_to_pipeline_inputs(truncated_chat_history),
                {"role": "human", "content": f"User question: {query}"}
            ]
        )[0]['generated_text'][-1]

        self._debug_print(f"Reformulated query: {reformulated_query}")

        # TODO: Just check if the reformulated query needs context or not

        # Get Context based on reformulated query
        context: List[Document] = self.vector_store.similarity_search(
            reformulated_query["content"], k=search_k)

        # Rank documents based on usefulness via Cross Encoder
        try:
            context_ranked = self._rerank_documents(
                context, reformulated_query["content"], top_k=top_k)
        except:
            try:
                context_ranked = self._rerank_documents(
                    context, query, top_k=top_k)
            except:
                context_ranked = context

        # Load the attached files (and images)
        attached_files_docs = self._load_input_documents(attached_file_paths)

        # Rank attached files based on usefulness via Cross Encoder
        try:
            attached_files_ranked = self._rerank_documents(
                attached_files_docs, reformulated_query["content"], top_k=top_k)
        except:
            try:
                attached_files_ranked = self._rerank_documents(
                    attached_files_docs, query, top_k=top_k)
            except:
                attached_files_ranked = attached_files_docs

        # Restructure the final prompt passed into LLM
        # TODO: Add attached files and images to the prompt
        # TODO: Add conversation history to the prompt (if needed, and if so, how much past history?)
        system_prompt = f"""
You are a friendly teaching assistant at a university.
Always use the given context to answer questions as accurately as possible.
Use the minimum amount of sentences needed to provide a correct answer. Do not be to verbose.
1. Provide the best answer based on the provided context. 
2. If there's no context available, kindly mention that clarification or additional details are needed. Do not mention the lack of context in the answer.
3. For questions that don't require specific context (general questions), answer using your general knowledge.
4. For greetings or casual conversation, respond warmly and engage in a friendly dialogue.
Context: {context_ranked} 
User also attached the following files:
{attached_files_ranked}
""".replace("\n", " ")

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

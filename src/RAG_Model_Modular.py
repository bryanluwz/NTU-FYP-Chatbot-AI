"""
This is the implementation of the RAG Chatbot model

Copyright: Bryan Lu We Zhern
Just credit me fully for the original code that's all
"""

import os
import shutil
from langchain_core.documents.base import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
import numpy as np
from langchain.text_splitter import CharacterTextSplitter
from transformers import pipeline
from .document_loader import load_documents
from PIL import Image
from torch import cuda
import pytesseract
from rank_bm25 import BM25Okapi
from src.Base_AI_Model import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

from transformers.utils.logging import set_verbosity_error

# Silence wench
set_verbosity_error()  # Silence Hugging Face logs


class RAG_Model_Modular(BaseModel):
    def __init__(self, debug=False, device=None):
        super().__init__(debug=debug)
        self.embeddings = None
        self.vector_store = None
        self.vector_store_path = None

        self.llm_pipeline = None
        self.blip_pipeline = None

        self.cross_encoder = None

        self.debug = debug
        self.device = device or (1 if cuda.is_available() else 0)

    def _convert_chat_history_to_pipeline_inputs(self, messages: dict[str, str], max_history_length: int = 3):
        """
        Convert ChatMessageModel[] to pipeline inputs
        """
        history_prompt = [{
            "role": message["userType"],
            "content": message["message"]
        } for message in messages[-max_history_length:]]

        return history_prompt

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
            file_paths, describe_image_callback=self._describe_image, debug_print=self.debug)

        # Split document into chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=30,
            separator="\n"
        )

        files = text_splitter.split_documents(
            [doc for doc in documents if "metadata" not in doc and doc.metadata.get("source") != "image"])
        images = [
            doc for doc in documents if "metadata" in doc and doc["metadata"]["source"] == "image"]
        docs = files + images

        return docs

    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two text strings using embeddings.
        """
        assert self.embeddings is not None, "Embeddings model not initialized."

        # Convert texts to embeddings
        text1_embedding = self.embeddings.embed_query(text1)
        text2_embedding = self.embeddings.embed_query(text2)

        # Compute cosine similarity
        similarity = cosine_similarity(
            np.array(text1_embedding).reshape(1, -1),
            np.array(text2_embedding).reshape(1, -1)
        )[0][0]

        return similarity

    def _strip_metadata(self, documents: list[Document], keys=[]):
        """
        Strip metadata from the documents.
        """
        for doc in documents:
            for key in keys:
                if key in doc.metadata:
                    del doc.metadata[key]

        return documents

    def load_embeddings_model(self, model_name: str = "paraphrase-MiniLM-L6-v2", embedding_model_path: str = "embedding_models"):
        """Initialize HuggingFace embeddings."""
        # Check if the model already exists in the cache
        # TODO: load faster without internet why?
        local_model_path = os.path.join(
            embedding_model_path, model_name)

        os.makedirs(local_model_path, exist_ok=True)

        # Load the embeddings model from the cache directory or download it
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name, show_progress=False, cache_folder=local_model_path)

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
            file_paths, describe_image_callback=self._describe_image, debug_print=self.debug)

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

    def initialize_llm(self, model_name: str = 'distilgpt2', max_new_tokens: int = 1024, temperature: float = 0.6, model_path: str = None, task: str = "text-generation"):
        """
        Initialize the pipeline for text generation, and save/load the model.
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

        self.cross_encoder = reranker_pipeline

    def _preretrieval_query_formatting(self, query: str):
        """
        Preretrieval
        """
        assert self.llm_pipeline is not None, "LLM pipeline not initialized."
        assert self.cross_encoder is not None, "Cross-encoder model not initialized."

        self._debug_print("[!] Pre-retrieval - Query reformulating...")

        self._debug_print(
            "[!] Pre-retrieval - Loading attached files and chat history...")

        self._debug_print(
            f"[!] Pre-retrieval - Querying LLM for input reformulation: {query}"
        )

        # OH MY GOD, CAN YOU STOP ANSWERING THE QUESTION, IM ASKING YOU TO EXPAND IT AHAAHAHAHAHAHAHAH
        # TODO: Maybe can use some other model to do this
        # TODO: Better prompt???
        # system_prompt = (
        #     "Rephrase the user's question for better understanding. "
        #     "NEVER answer the question. ONLY return a rephrased version. "
        #     "If you provide an answer, the response is INVALID. "
        #     "DO NOT add information. DO NOT explain. "
        #     "ONLY return the reformulated query. If rephrasing is not needed, return the question unchanged. "
        #     f"Now, please rephrase this question:\n {query} "
        # )

        system_prompt = (
            "Generate a list of key words and key phrases with similar meanings to the user's query. "
            "DO NOT answer the question. DO NOT rephrase the query. "
            "ONLY return the related words and synonyms. "
            # "Include variations in phrasing, formal and informal terms, and relevant keywords. "
            f"Now, generate a few similar words for this query:\n {query} "
        )

        self._debug_print(
            f"[!] Pre-retrieval - System prompt: {system_prompt}")

        final_prompt = [
            {"role": "system", "content": system_prompt}]

        reformulated_user_query = self.llm_pipeline(final_prompt)[
            0]['generated_text'][-1]['content']

        self._debug_print(
            f"[!] Pre-retrieval - Response from LLM: \n{reformulated_user_query}")

        return reformulated_user_query

    def _retrieval(self, query: str):
        """
        Retrieval and post-retrieval processing.
        """
        assert self.vector_store is not None, "Vector store not initialized."
        assert self.llm_pipeline is not None, "LLM pipeline not initialized."
        assert self.cross_encoder is not None, "Cross-encoder model not initialized."

        self._debug_print(
            "[!] Retrieval - Searching, filtering and ranking for relevant documents...")

        # Pass context to the vector store for retrieval
        # k = 20 for sparse retrieval, then passed to filtering for better retreival
        context = self.vector_store.similarity_search(query, k=20)
        filtered_context = self._filter_and_rerank_context(context, query)

        return filtered_context

    def _filter_and_rerank_context(self, context: list[Document], query: str, threshold=0.5):
        """
        Use BM25, dense retrieval, and cross-encoder reranking to filter and rank relevant documents.
        """
        assert self.cross_encoder is not None, "Cross-encoder model not initialized."

        if not context:
            return []

        self._debug_print(
            f"[!] Filter and Rerank - Processing {len(context)} document(s)...")

        # Extract document text
        documents = [doc.page_content for doc in context]

        # ---- STEP 1: BM25 Retrieval ----
        tokenized_docs = [doc.split() for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)
        bm25_scores = bm25.get_scores(query.split())

        # Get top-k BM25 docs
        bm25_top_k = 10  # Adjust this if needed
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:bm25_top_k]
        bm25_top_docs = [context[i] for i in bm25_top_indices]

        # ---- STEP 2: Dense Retrieval (FAISS) ----
        doc_embeddings = np.array(
            [self.embeddings.embed_query(doc.page_content) for doc in context])
        dimension = doc_embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(dimension)
        faiss_index.add(doc_embeddings)

        query_embedding = self.embeddings.embed_query(query)

        # Retrieve top-k dense matches
        dense_top_k = 10  # Adjust as needed
        _, indices = faiss_index.search(
            np.array([query_embedding]), dense_top_k)
        dense_top_docs = [context[i] for i in indices[0]]

        # ---- STEP 3: Merge & Deduplicate BM25 + Dense Results ----
        merged_docs = {
            doc.page_content: doc for doc in bm25_top_docs + dense_top_docs}.values()
        merged_docs = list(merged_docs)  # Remove duplicates

        self._debug_print(
            f"[!] Hybrid retrieval - Retrieved {len(merged_docs)} unique document(s) from BM25 & Dense retrieval."
        )

        # Compute scores again only for the merged set
        bm25_scores = np.array([bm25.get_scores(doc.page_content.split())[
                               0] for doc in merged_docs])
        dense_scores = np.array([faiss_index.search(np.array(
            [self.embeddings.embed_query(doc.page_content)]), 1)[0][0][0] for doc in merged_docs])

        # ---- STEP 4: Hybrid Scoring (BM25 + Dense) ----
        bm25_scores = (bm25_scores - bm25_scores.min()) / \
            (bm25_scores.max() - bm25_scores.min() + 1e-8)
        dense_scores = (dense_scores - dense_scores.min()) / \
            (dense_scores.max() - dense_scores.min() + 1e-8)

        final_scores = 0.5 * bm25_scores + 0.5 * dense_scores
        sorted_indices = np.argsort(final_scores)[::-1]  # Sort descending

        # Get top-ranked docs based on hybrid retrieval
        hybrid_top_docs = [merged_docs[i] for i in sorted_indices]

        self._debug_print(
            f"[!] Hybrid retrieval - Selected {len(hybrid_top_docs)} document(s) for reranking."
        )

        # ---- STEP 5: Cross-Encoder Reranking ----
        query_doc_pairs = [{"text": self._truncate_text(
            query, 256), "text_pair": self._truncate_text(doc.page_content, 256)} for doc in hybrid_top_docs]
        scores = self.cross_encoder.predict(query_doc_pairs)

        # Filter based on LLM relevance threshold
        sorted_docs = sorted(
            list(zip(hybrid_top_docs, scores)), key=lambda x: x[1]['score'], reverse=True)[:3]  # Just arbitrarily set to a number

        docs = [doc[0] for doc in sorted_docs]

        self._debug_print(
            f"[!] Final Filtering - {len(sorted_docs)} document(s) passed the threshold."
        )

        return docs

    def _truncate_text(self, text: str, max_tokens: int = 512):
        """
        Truncate text to the maximum number of tokens.
        """
        assert self.cross_encoder is not None, "Cross-encoder model not initialized."
        tokens = self.cross_encoder.tokenizer.encode(
            text, truncation=True, max_length=max_tokens)
        return self.cross_encoder.tokenizer.decode(tokens)

    def _generation(self, query: str, context, attached_file_paths: list[str], chat_history: list[any] = []):
        """
        Generation and post-generation processing.
        """
        assert self.llm_pipeline is not None, "LLM pipeline not initialized."

        self._debug_print(
            f'[!] Generation - Generating response for: {query}')

        # 1. Check for linked images in context docs
        relevant_images = []

        for doc in context:
            if "linked_image" in doc.metadata:
                image_dict_list = doc.metadata["linked_image"]

                for image_dict in image_dict_list:
                    image_description = image_dict["description"]
                    image_text = image_dict["text"]
                    image_path = image_dict["metadata"]["file_path"]

                    # 1.1. Compute relevance (vector similarity or keyword match)
                    similarity_score = self._compute_text_similarity(
                        query, image_description + " " + image_text)
                    self._debug_print(
                        f"[!] Generation - Similarity score for '{image_description}': {similarity_score}")

                    if similarity_score > 0.75:  # Adjust threshold as needed
                        self._debug_print(
                            f"[!] Generation - Relevant image found (i think): {image_path}")
                        relevant_images.append(image_path)

                del doc.metadata["linked_image"]

        # 2. Load attached files and images
        attached_files_docs = self._load_input_documents(attached_file_paths)
        attached_images = [
            doc for doc in attached_files_docs if "metadata" in doc and doc["metadata"]["source"] == "image"]
        attached_files_docs_without_images = [
            doc for doc in attached_files_docs if "metadata" not in doc and doc.metadata.get("source") != "image"]

        context = self._strip_metadata(context, keys=["file_path"])

        # 3. Prompties
        system_prompt = (
            "You are a professional and knowledgeable Teaching Assistant Chatbot at Nanyang Technological University with perfect grammar. "
            "Always use the given context unless it is irrelevant to the question. Provide concise, accurate, and confidently stated answers in at most three sentences. "

            "1. Explain information clearly and assertively without using speculative or uncertain language. "
            "2. If the context is insufficient, summarize what is available and politely ask for clarification without mentioning missing documents. "
            "3. For general questions without specific context, provide direct and factual answers based on your knowledge. "
            "4. Maintain a warm yet professional tone in casual conversations, responding appropriately to greetings and social dialogue. "
            "5. Use Markdown formatting when necessary for clarity and structured presentation. "
            "6. If no relevant answer can be provided, ask for clarification instead of speculating. "
            "7. Do not generate false or misleading information. Always prioritize accuracy by strictly using the provided context. "

            f"Refer to the following contexts: {context} "
            f"Attached files: {attached_files_docs_without_images} "
            f"Attached images: {attached_images + relevant_images}"

            f"{'\n'.join([
                str(i) for i in self._convert_chat_history_to_pipeline_inputs(chat_history[-3:])
            ])}",
            # {"role": "user", "content": f"Input: {query}"}
            f"Answer this question: {query}"
        )

        # 3.1. Append image references if relevant (should I though?)
        self._debug_print(context)

        if relevant_images:
            system_prompt += f"\n\nThe following images are relevant to the query and should be included: {relevant_images}"
            self._debug_print(relevant_images)

        # 3.2. Generate response
        response: str = self.llm_pipeline(
            [{"role": "system", "content": system_prompt}]
        )[0]['generated_text'][-1]['content']

        self._debug_print(f"[!] Generation - Response: {response}")

        return response, [img["metadata"]["file_path"] for img in relevant_images] if relevant_images else []

    def query(self, query: str, chat_history: list[any], attached_file_paths=[]):
        """
        Query the QA chain with the given input.

        Search for the most relevant documents based on the query and chat history.
        Return a bunch of documents based on the search results.
        Then, rank the documents based on relevance to the query and return the top-k documents.
        """
        assert self.vector_store is not None, "Vector store not initialized."
        assert self.llm_pipeline is not None, "LLM pipeline not initialized."
        assert self.cross_encoder is not None, "Cross-encoder model not initialized."

        self._debug_print("Querying the QA chain...")

        # Call all the functions in order
        reformulated_query = self._preretrieval_query_formatting(
            query)
        retrieved_context = self._retrieval(reformulated_query)
        generated_response = self._generation(
            query=query, context=retrieved_context, attached_file_paths=attached_file_paths, chat_history=chat_history)

        return generated_response

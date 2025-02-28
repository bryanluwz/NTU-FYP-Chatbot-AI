"""
This is the implementation of the RAG Chatbot model

Copyright: Bryan Lu We Zhern
Just credit me fully for the original code that's all
"""

import os
import shutil
import numpy as np
from PIL import Image
import pytesseract
import faiss
from torch import cuda

from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents.base import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

from transformers import pipeline
from transformers.utils.logging import set_verbosity_error

from .document_loader import load_documents
from src.Base_AI_Model import BaseModel

# Silence wench
set_verbosity_error()  # Silence Hugging Face logs i think, maybe it's not even working


class RAG_Model(BaseModel):
    def __init__(self, debug=False, device=None, huggingface_token=None):
        super().__init__(debug=debug)
        self.embeddings = None
        self.vector_store = None
        self.vector_store_path = None

        self.llm_pipeline = None
        self.blip_pipeline = None

        self.cross_encoder = None

        self.debug = debug
        self.device = device or (1 if cuda.is_available() else 0)
        self.huggingface_token = huggingface_token

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
            file_paths,
            describe_image_callback=self._describe_image,
            debug_print=self.debug)

        # Split document into chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=30,
            separator="\n"
        )

        files = text_splitter.split_documents(
            [doc for doc in documents if doc.metadata and doc.metadata.get("source") != "image"])
        images = [doc for doc in documents if doc.metadata and doc.metadata.get(
            "source") == "image"]

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

    def _get_page_content(self, documents: list[Document]):
        """
        Strip metadata from the documents.
        """
        return [f"Page {doc.metadata.get('page', 'N/A')}: {doc.page_content}" for doc in documents]

    def load_embeddings_model(self, model_name: str = "paraphrase-MiniLM-L6-v2", embedding_model_path: str = "embedding_models"):
        """Initialize HuggingFace embeddings."""
        # Check if the model already exists in the cache
        # TODO: load faster without internet why?
        local_model_path = os.path.join(
            embedding_model_path, model_name)

        os.makedirs(local_model_path, exist_ok=True)

        # Load the embeddings model from the cache directory or download it
        # NOTE: Cache got problem, so no cache in local model path. why? idk
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name, show_progress=False)

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
        documents = load_documents(
            file_paths, describe_image_callback=self._describe_image,
            debug_print=self.debug)

        # Split document into chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=690,
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

    def initialize_llm(self, model_name: str = 'distilgpt2', max_new_tokens: int = 512, temperature: float = 0.6, model_path: str = None, task: str = "text-generation"):
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
                do_sample=True,
                framework="pt",
                device_map="auto",
                torch_dtype="auto"
            )
        else:
            # Get the model size before downloading
            print(
                f"⬇️ Downloading and saving model '{model_name}' to {model_save_path}...")
            text_gen_pipeline = pipeline(
                task=task,
                model=model_name,
                tokenizer=model_name,
                do_sample=True,
                framework="pt",
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                device_map="auto",
                torch_dtype="auto",
                token=self.huggingface_token
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
        print(f"🔄 Loading BLIP model from {model_save_path}...")

        if os.path.exists(model_save_path):
            try:
                print(f"🔄 Loading BLIP model from {model_save_path}...")
                self.blip_pipeline = pipeline(
                    task=task,
                    model=model_save_path,
                    tokenizer=model_save_path,
                    framework="pt",
                    device=self.device
                )
            except RuntimeError as e:
                if "CUDA error: invalid device ordinal" in str(e):
                    print(
                        "⚠️ Idk there's some weird GPU device behaviour. Falling back to CPU.")
                    self.blip_pipeline = pipeline(
                        task=task,
                        model=model_save_path,
                        tokenizer=model_save_path,
                        framework="pt",
                        device=-1  # Use CPU
                    )
                else:
                    raise e
        else:
            print(
                f"⬇️ Downloading and saving BLIP model '{model_name}' to {model_save_path}...")

            try:
                self.blip_pipeline = pipeline(
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
                    self.blip_pipeline = pipeline(
                        task=task,
                        model=model_name,
                        tokenizer=model_name,
                        framework="pt",
                        device=-1  # Use CPU
                    )
                else:
                    raise e

            # Save the model
            self.blip_pipeline.save_pretrained(model_save_path)
            print(f"✅ BLIP model '{model_name}' saved to {model_save_path}.")

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

        # System prompt for LLM for query reformulation / expansion / whatever you call it
        system_prompt = (
            "You are a teaching assistant at Nanyang Technological University in Singapore. "
            "You have a vast knowledge of the course material. "
            "You are tasked to reformulate the user's query to generate five similar words and phrases"
            "\n"
            "DO NOT answer the question, but ensure the reformulated query is relevant to the user's question. "
            "ONLY return the related words and synonyms, without additional text and numberings. "
        )

        final_prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Help me generate five similar words and phrases for: {query}"}
        ]

        reformulated_user_query = self.llm_pipeline(final_prompt)[
            0]['generated_text'][-1]['content']

        self._debug_print(
            f"[!] Pre-retrieval - Reformulated \n{query}\n    to\n{reformulated_user_query}")

        return reformulated_user_query

    def _retrieval(self, *query: str):
        """
        Retrieval and post-retrieval processing.
        """
        assert self.vector_store is not None, "Vector store not initialized."
        assert self.llm_pipeline is not None, "LLM pipeline not initialized."
        assert self.cross_encoder is not None, "Cross-encoder model not initialized."

        self._debug_print(
            "[!] Retrieval - Searching, filtering and ranking for relevant documents...")

        # Pass context to the vector store for retrieval
        # k = 10 for sparse retrieval, then passed to filtering for better retreival
        full_query = "\n".join([q for q in query])
        reformulated_queries = full_query.split("\n")

        context = [self.vector_store.similarity_search(
            q, k=20) for q in reformulated_queries]
        flattened_context = [doc for sublist in context for doc in sublist]
        filtered_context = self._filter_and_rerank_context(
            flattened_context, full_query)

        return filtered_context

    def _filter_and_rerank_context(self, context: list[Document], query: str, num_return=4):
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
            list(zip(hybrid_top_docs, scores)), key=lambda x: x[1]['score'], reverse=True)[:num_return]

        docs = [doc[0] for doc in sorted_docs]

        self._debug_print(
            f"[!] Final Filtering - {len(sorted_docs)} document(s) passed the threshold."
        )

        return docs

    def _rank_relevant_images(self, images: list[dict], generated_response: str):
        """
        Rank relevant images based on the generated response and images description + text
        Only with cross encoder
        """
        assert self.cross_encoder is not None, "Cross-encoder model not initialized."

        for image in images:
            image["content"] = image["description"] + " " + image["text"]
            del image["description"]
            del image["text"]

        query_image_pairs = [{"text": self._truncate_text(
            generated_response, 256), "text_pair": self._truncate_text(image["content"], 256)} for image in images]

        scores = self.cross_encoder.predict(query_image_pairs)

        # Normalise scores
        if not scores or len(scores) == 0:
            return []

        max_score = max([score['score'] for score in scores])
        min_score = min([score['score'] for score in scores])

        for score in scores:
            score['score'] = (score['score'] - min_score) / \
                (max_score - min_score + 1e-8)

        sorted_images = sorted(
            list(zip(images, scores)), key=lambda x: x[1]['score'], reverse=True)

        return sorted_images

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
            f'[!] Generation - Generating response for: \n{query}\nwith {len(context)} context document(s) and {len(attached_file_paths)} attached file(s).')

        # 1. Check for linked images in context docs
        relevant_images = []

        for doc in context:
            # self._debug_print(doc)
            if "linked_image" in doc.metadata:
                if len(attached_file_paths) > 0:
                    self._debug_print(
                        "[!] Generation - Attached files are present, ignoring linked images.")
                    # Remove linked images from context if attached files are present
                    # i.e. ignore linked images if attached files are present
                    del doc.metadata["linked_image"]
                    continue

                image_dict_list = doc.metadata["linked_image"]

                for image_dict in image_dict_list:
                    image_description = image_dict["description"]
                    image_text = image_dict["text"]
                    image_path = image_dict["metadata"]["file_path"]

                    # 1.1. Append image to relevant images
                    # Adjust threshold as needed
                    relevant_images.append({
                        "description": image_description,
                        "text": image_text,
                        "metadata": {
                            "file_path": image_path
                        }
                    })

                del doc.metadata["linked_image"]

        # 2. Load attached files and images
        attached_files_docs = self._load_input_documents(attached_file_paths)
        attached_images = [
            doc for doc in attached_files_docs if doc.metadata and doc.metadata.get("source") == "image"]
        attached_files_docs_without_images = [
            doc for doc in attached_files_docs if doc.metadata and doc.metadata.get("source") != "image"]

        context = self._get_page_content(context)

        # 3. Prompties
        system_prompt = (
            "Goal:\n"
            "You are a teaching assistant at Nanyang Technological University in Singapore with vast knowledge of the course material.\n"
            "Your role is to provide short and concise answers to user questions, with a maximum of three sentences.\n"
            "Always maintain this role unless explicitly instructed otherwise.\n"
            "\n"
            "Return Format:\n"
            "- Provide a direct answer or ask for clarification.\n"
            "- Do not exceed three sentences.\n"
            "- Use markdown only when necessary.\n"
            "\n"
            "Warnings:\n"
            "- Do not hallucinate or fabricate information.\n"
            "- Use only provided context and general knowledge.\n"
            "- Ignore context when responding to greetings, farewells, or thanks; respond warmly instead.\n"
            "- Provide only relevant answers. Avoid unnecessary elaboration.\n"
            "- Do not include numbering or extra text from references.\n"
            "\n"
        )

        # 3.1. Append stuffs
        messages = [{"role": "system", "content": system_prompt}]

        if chat_history:
            messages.append(
                {"role": "user", "content": f"Chat History Context: {self._convert_chat_history_to_pipeline_inputs(chat_history[-3:])}"})

        if context:
            context_str = '\n'.join(context)
            # self._debug_print(
            #     f"[!] Generation - Context: {context_str}")
            messages.append(
                {"role": "user", "content": f"Reference Document Context: {context_str}"})

        user_message = ""
        if attached_files_docs_without_images:
            attached_content = self._get_page_content(
                attached_files_docs_without_images)
            attached_content = [
                f"**File {i+1}.** {content}" for i, content in enumerate(attached_content)]
            user_message += f"Refer to these attached files: {attached_content}\n"

        if attached_images:
            attached_content = self._get_page_content(attached_images)
            attached_content = [
                f"**Image {i+1}.** {content}" for i, content in enumerate(attached_content)]
            user_message += f"Refer to these attached images: {attached_content}\n"

        messages.append(
            {"role": "user", "content": user_message + f"Answer in a concise manner, The user input is '{query}'"})

        # 3.2. Generate response
        response: str = self.llm_pipeline(
            messages, eos_token_id=self.llm_pipeline.tokenizer.eos_token_id
        )[0]['generated_text'][-1]['content']

        self._debug_print(f"[!] Generation - Response: {response}")

        # 4. Rank relevant images based on the generated response
        scored_relevant_images = self._rank_relevant_images(
            relevant_images, response)

        filtered_images = []

        for img, score in scored_relevant_images:
            # Adjust threshold as needed
            if score['score'] > 0.9:
                filtered_images.append(img)

        # if len(filtered_images) == 0:
        #     # Get the highest scored image
        #     if scored_relevant_images:
        #         filtered_images.append(scored_relevant_images[0][0])

        if filtered_images:
            self._debug_print(
                f"[!] Generation - Found {len(filtered_images)} relevant image(s) based on the generated response.")

        return response, [img["metadata"]["file_path"] for img in filtered_images] if filtered_images else []

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
        retrieved_context = None
        if len(attached_file_paths) == 0:
            reformulated_query = self._preretrieval_query_formatting(query)
            retrieved_context = self._retrieval(reformulated_query, query)
        else:
            self._debug_print(
                f"[!] Query - Attached files are present, ignoring RAG model...")

        # TEMP: Chat history stuff
        if chat_history:
            if chat_history[0] != chat_history[-1] and chat_history[0] != chat_history[-2]:
                chat_history = [chat_history[0],
                                chat_history[-1], chat_history[-2]]
            else:
                chat_history = [chat_history[0]]

        generated_response = self._generation(
            query=query, context=retrieved_context or [], attached_file_paths=attached_file_paths, chat_history=chat_history)

        return generated_response

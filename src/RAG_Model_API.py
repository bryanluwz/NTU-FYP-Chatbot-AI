"""
This is the implementation of the RAG Chatbot model
but with ✨API✨ for firee
but like i spent quite a lot of time developing for local, why i change to using API

Copyright: Bryan Lu We Zhern
Just credit me fully for the original code that's all
"""

import os
import shutil
import numpy as np
import faiss
import re

from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents.base import Document
from langchain_community.vectorstores import FAISS

from together import Together

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

from src.Base_AI_Model import BaseModel
from .document_loader import load_documents


from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity


from .document_loader import load_documents
from src.Base_AI_Model import BaseModel
from openai import AzureOpenAI


def _softmax(scores, temperature=0.5):
    """Apply softmax with temperature scaling."""
    scores = np.array(scores) / temperature
    # Subtract max for numerical stability
    exp_scores = np.exp(scores - np.max(scores))
    return (exp_scores / np.sum(exp_scores)).tolist()


class RAG_Model_API(BaseModel):
    """Notice how this class is a subclass of RAG_Model, very saving re-implementing stuffs, very mindful, very demure"""

    def __init__(self, debug=False, device=None, together_api_key=None, azure_api_endpoint=None, azure_api_key=None, hf_hub_api_key=None, azure_openai_endpoint=None, azure_openai_api_key=None):
        super().__init__(debug=debug)
        self.embeddings_model = None
        self.embeddings_pipeline = None
        self.azure_openai_endpoint = azure_openai_endpoint
        self.azure_openai_api_key = azure_openai_api_key

        self.vector_store = None
        self.vector_store_path = None

        self.llm_pipeline = None
        self.llm_pipeline_kwargs = None
        self.together_api_key = together_api_key

        self.image_pipeline = None
        self.azure_api_endpoint = azure_api_endpoint
        self.azure_api_key = azure_api_key
        self.hf_hub_api_key = hf_hub_api_key

        self.debug = debug
        self.device = device or 0

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
        Describe the image using the image pipeline. (and also OCR)
        """
        assert self.image_pipeline is not None, "Image pipeline not initialized."

        image_bytes = open(image_path, "rb").read()

        image_description_response = self.image_pipeline.analyze(
            image_bytes, visual_features=[VisualFeatures.CAPTION]
        )

        image_description = image_description_response.get(
            'captionResult').get('text')

        # Get OCR text

        return {"description": image_description, "text": ""}

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

    def _get_page_content(self, documents: list[Document]):
        """
        Strip metadata from the documents.
        """
        return [f"Page {doc.metadata.get('page', 'N/A')}: {doc.page_content}" for doc in documents]

    def load_embeddings_model(self, model_name: str = "text-embedding-ada-002"):
        """Initialize embeddings."""
        self.embeddings_model = AzureOpenAI(
            api_key=self.azure_openai_api_key, api_version="2024-06-01", azure_endpoint=self.azure_openai_endpoint)
        self.embeddings_pipeline = lambda text: self.embeddings_model.embeddings.create(
            input=text, model=model_name).data[0].embedding

    def load_vector_store(self, vector_store_path: str = "vector_store/<your_vector_store_name>", file_paths=[]):
        """Load the FAISS vector store if it exists."""
        assert self.embeddings_model is not None and self.embeddings_pipeline is not None, "Embeddings model not initialized."

        try:
            faiss_index_path = os.path.join(vector_store_path, "index.faiss")
            faiss_pkl_path = os.path.join(vector_store_path, "index.pkl")

            self._debug_print(
                f"🔍 Loading vector index and pickle files from {faiss_index_path} and {faiss_pkl_path}..")

            if os.path.exists(faiss_index_path) and os.path.exists(faiss_pkl_path):
                # Load persisted vector store
                persisted_vectorstore = FAISS.load_local(
                    vector_store_path, self.embeddings_pipeline, allow_dangerous_deserialization=True)
                print("✅ Loaded vector store from local storage.")
                self.vector_store = persisted_vectorstore
                self.vector_store_path = vector_store_path

            else:
                raise FileNotFoundError
        except FileNotFoundError:
            # raise FileNotFoundError(vector_store_path, faiss_index_path, faiss_pkl_path, os.path.exists(
            #     faiss_index_path), os.path.exists(faiss_pkl_path))
            self.vector_store = None
            self.create_and_save_vector_store(vector_store_path, file_paths)

    def create_and_save_vector_store(self, vector_store_path, file_paths):
        """Create a new FAISS vector store from the given PDF and save it."""
        assert self.embeddings_model is not None, "Embeddings model not initialized."

        self._debug_print(
            "⚠️ Creating a new vector store, if one already exists it will be overwritten.")

        if os.path.exists(vector_store_path):
            shutil.rmtree(vector_store_path)
            print("🗑️ Removed existing vector store.")

        os.makedirs(vector_store_path, exist_ok=True)

        # Load document using PyPDFLoader
        documents = load_documents(
            file_paths, describe_image_callback=self._describe_image,
            debug_print=self.debug, is_rate_limit=True)

        # Split document into chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=690,
            chunk_overlap=30,
            separator="\n"
        )
        docs = text_splitter.split_documents(documents)
        doc_texts = [doc.page_content for doc in docs]

        embeddings = [self.embeddings_pipeline(
            doc.page_content) for doc in docs]

        # Create vectors using FAISS
        vectorstore = FAISS.from_embeddings(
            zip(doc_texts, embeddings), self.embeddings_pipeline)

        # Persist the vectors locally on disk
        vectorstore.save_local(vector_store_path)
        self._debug_print("💾 Vector store saved locally.")

        self.vector_store_path = vector_store_path
        self.vector_store = vectorstore

    def _load_input_documents(self, file_paths: list[str]):
        """
        Load the input documents from the given file paths.
        """
        # Load document using PyPDFLoader
        documents = load_documents(
            file_paths,
            describe_image_callback=self._describe_image,
            debug_print=self.debug, is_rate_limit=True)

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

    def initialize_llm(self, model_name: str = 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free', max_new_tokens: int = 512, temperature: float = 0.6):
        """
        Initialize the Together.AI client for text generation
        """
        self.llm_pipeline_kwargs = {
            "model": model_name,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }
        self.llm_pipeline = Together(api_key=self.together_api_key)

    def initialize_image_pipeline(self):
        """
        Initialize the image pipeline for image and text encoding.
        """
        assert self.azure_api_endpoint is not None, "Azure API endpoint not initialized."
        assert self.azure_api_key is not None, "Azure API key not initialized."

        client = ImageAnalysisClient(
            endpoint=self.azure_api_endpoint,
            credential=AzureKeyCredential(self.azure_api_key)
        )
        self.image_pipeline = client

    def _preretrieval_query_formatting(self, query: str):
        """
        Preretrieval
        """
        assert self.llm_pipeline is not None, "LLM pipeline not initialized."

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
            "ONLY return the related words and synonyms, without additional text and numberings, each in a new line. "
        )

        final_prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Help me generate five similar words and phrases for: {query}"}
        ]

        completion = self.llm_pipeline.chat.completions.create(
            model=self.llm_pipeline_kwargs["model"], messages=final_prompt, max_tokens=self.llm_pipeline_kwargs["max_new_tokens"], temperature=self.llm_pipeline_kwargs["temperature"])

        reformulated_user_query = completion.choices[0].message.content

        self._debug_print(
            f"[!] Pre-retrieval - Reformulated \n{query}\n   to \n{reformulated_user_query}")

        return reformulated_user_query

    def _retrieval(self, *query: str):
        """
        Retrieval and post-retrieval processing.
        """
        assert self.vector_store is not None, "Vector store not initialized."
        assert self.llm_pipeline is not None, "LLM pipeline not initialized."

        self._debug_print(
            "[!] Retrieval - Searching, filtering and ranking for relevant documents...")

        # Pass context to the vector store for retrieval
        # k = idk for sparse retrieval, then passed to filtering for better retreival
        full_query = "\n".join([q for q in query])
        reformulated_queries = re.split(r'\n+', full_query)

        context = [self.vector_store.similarity_search(
            q, k=max(1, 20 // len(reformulated_queries))) for q in reformulated_queries]
        flattened_context = [doc for sublist in context for doc in sublist]
        filtered_context = self._filter_and_rerank_context(
            flattened_context, full_query)

        return filtered_context

    def _filter_and_rerank_context(self, context: list[Document], query: str, num_return=4):
        """
        FUCK THIS FUNCTION
        Use BM25, dense retrieval, and cross-encoder reranking to filter and rank relevant documents.
        """
        return context
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
        bm25_top_k = 4  # Adjust this if needed
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:bm25_top_k]
        bm25_top_docs = [context[i] for i in bm25_top_indices]

        # ---- STEP 2: Dense Retrieval (FAISS) ----
        # Instead of recalculating embeddings, use the ones from the vector store
        # Get pre-stored embeddings from the vector store
        doc_embeddings = [doc.embedding for doc in context]
        # Assuming all embeddings have the same dimension
        dimension = doc_embeddings[0].shape[0]

        # Initialize FAISS index and add the pre-stored embeddings
        faiss_index = faiss.IndexFlatIP(dimension)
        # Add all embeddings to the FAISS index
        faiss_index.add(np.array(doc_embeddings))

        # Get query embedding
        query_embedding = self.embeddings_pipeline(query)

        # Perform FAISS search to retrieve top-k dense matches
        dense_top_k = 4  # Adjust as needed
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

        # Instead of recomputing embeddings, use the embeddings stored in context
        dense_scores = np.array([
            faiss_index.search(np.array([doc.embedding]), 1)[0][0][0] for doc in merged_docs])

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
        # Query embedding
        query_embedding = self.embeddings_pipeline(query)

        # Calculate similarity scores between the query and documents using their embeddings
        scores = np.array([faiss_index.search(np.array([doc.embedding]), 1)[
            0][0][0] for doc in hybrid_top_docs])

        # Filter based on LLM relevance threshold
        sorted_docs = sorted(
            list(zip(hybrid_top_docs, scores)), key=lambda x: x[1], reverse=True)[:num_return]

        # Extract documents from the sorted results
        docs = [doc[0] for doc in sorted_docs]

        self._debug_print(
            f"[!] Final Filtering - {len(sorted_docs)} document(s) passed the threshold."
        )

        return docs

    def _rank_relevant_images(self, images: list[dict], generated_response: str):
        """
        Rank relevant images based on the generated response and images description + text
        Actually there's also some filtering here hmmm
        """

        # Merge description and text
        for image in images:
            image["content"] = image.pop(
                "description") + " " + image.pop("text")

        # Make sure image embeddings and query embedding are 2D
        # Ensure query_embedding and image_embeddings are 2D
        query_embedding = self.embeddings_pipeline(
            generated_response)
        image_embeddings = [self.embeddings_pipeline(image["content"]).reshape(
            1, -1) for image in images]  # Shape (1, feature_dim) for each image

        # Compute cosine similarity
        # Now this works, returns a 1D array of scores
        scores = []
        if len(image_embeddings) > 0:
            scores = cosine_similarity([query_embedding], image_embeddings)

        if not scores or len(scores) == 0:
            return []

        raw_scores = scores[0]

        self._debug_print(raw_scores)

        # Handle single image case
        if len(raw_scores) == 1:
            # Raw cutoff at 0.whatever
            return [(images[0], raw_scores[0])] if raw_scores[0] > 0.1 else []
        else:
            # If theres one that is very high score already then just return that one ba
            if max(raw_scores) > 0.8:
                return [(images[i], raw_scores[i]) for i, score in enumerate(raw_scores) if score > 0.8]

            # Cutoff low scores if multiple images at threshold 0.whatever, set as 0
            for i, score in enumerate(raw_scores):
                if score < 0.02:
                    raw_scores[i] = 0

        # Apply softmax with temperature
        softmax_scores: list[float] = _softmax(raw_scores, temperature=0.1)

        self._debug_print(softmax_scores)

        # Compute dynamic cutoff
        best_score = max(softmax_scores)
        # Dynamic threshold with small buffer
        cutoff = max(best_score - 0.05, 0)

        # Pair images with scores and filter using cutoff
        sorted_images = sorted(
            [(img, score)
             for img, score in zip(images, softmax_scores) if score >= cutoff],
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_images

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
            "Your role is to provide short and concise answers to user questions. Answer to the best of your knowledge. \n"
            "Always maintain this role unless explicitly instructed otherwise.\n"
            "\n"
            "Return Format:\n"
            "- Provide a direct answer or ask for clarification.\n"
            "- Use markdown when necessary, such as but not limited to, code, links, headers, list, etc.\n"
            "\n"
            "Warnings:\n"
            "- Do not hallucinate or fabricate information.\n"
            "- Use only provided context and general knowledge.\n"
            "- Ignore context when responding to greetings, farewells, or thanks; respond warmly instead.\n"
            "- Provide only relevant answers. Avoid unnecessary elaboration.\n"
            "- Do not include numbering or extra text from references.\n"
            "- Do not mention any reference to the context or attached documents explicitly, unless necessary.\n"
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
        completion = self.llm_pipeline.chat.completions.create(
            messages=messages, max_tokens=self.llm_pipeline_kwargs["max_new_tokens"], temperature=self.llm_pipeline_kwargs["temperature"], model=self.llm_pipeline_kwargs["model"])

        response = completion.choices[0].message.content

        self._debug_print(f"[!] Generation - Response: {response}")

        # 4. Rank relevant images based on the generated response
        scored_relevant_images = self._rank_relevant_images(
            relevant_images, response)

        filtered_images = [
            img for (img, score) in scored_relevant_images if score > 0.5]

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

        self._debug_print("Querying the QA chain...")

        # Call all the functions in order
        retrieved_context = None
        if len(attached_file_paths) == 0:
            reformulated_query = self._preretrieval_query_formatting(query)
            retrieved_context = self._retrieval(reformulated_query, query)
        else:
            self._debug_print(
                f"[!] Query - Attached files are present, ignoring RAG model...")

        # Chat history stuff only last n messages + first message
        if chat_history:
            if len(chat_history) > 2:
                chat_history = [chat_history[0]] + chat_history[-2:]
            else:
                chat_history = [chat_history[0]] + chat_history[1:]

        generated_response = self._generation(
            query=query, context=retrieved_context or [], attached_file_paths=attached_file_paths, chat_history=chat_history)

        return generated_response

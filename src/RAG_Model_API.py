"""
This is the implementation of the RAG Chatbot model
but with ✨API✨ for firee
but like i spent quite a lot of time developing for local, why i change to using API

Copyright: Bryan Lu We Zhern
Just credit me fully for the original code that's all
"""

import os
import shutil
from langchain.text_splitter import CharacterTextSplitter
from .document_loader import load_documents
from langchain_community.vectorstores import FAISS

from src.RAG_Model import RAG_Model

from together import Together

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential


class RAG_Model_API(RAG_Model):
    """Notice how this class is a subclass of RAG_Model, very saving re-implementing stuffs, very mindful, very demure"""

    def __init__(self, debug=False, device=None, together_api_key=None, azure_api_endpoint=None, azure_api_key=None):
        super().__init__(debug=debug, device=device)
        self.llm_pipeline_kwargs = None

        self.together_api_key = together_api_key
        self.azure_api_endpoint = azure_api_endpoint
        self.azure_api_key = azure_api_key

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

    def create_and_save_vector_store(self, vector_store_path, file_paths):
        """Create a new FAISS vector store from the given PDF and save it."""
        assert self.embeddings is not None, "Embeddings model not initialized."

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

        # Create vectors using FAISS
        vectorstore = FAISS.from_documents(docs, self.embeddings)

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

        completion = self.llm_pipeline.chat.completions.create(
            model=self.llm_pipeline_kwargs["model"], messages=final_prompt, max_tokens=self.llm_pipeline_kwargs["max_new_tokens"], temperature=self.llm_pipeline_kwargs["temperature"])

        reformulated_user_query = completion.choices[0].message.content

        self._debug_print(
            f"[!] Pre-retrieval - Reformulated \n{query}\n   to \n{reformulated_user_query}")

        return reformulated_user_query

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

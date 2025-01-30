"""
This is the implementation of the RAG Chatbot model
"""


from datetime import datetime
import os
import regex as re
import shutil
from typing import List
from langchain_core.documents.base import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from transformers import pipeline
from .document_loader import load_documents
from PIL import Image
from torch import cuda
import pytesseract
import json
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser


class SubinputClass(BaseModel):
    subinput: str = Field(description="The subinput to be processed")
    retrieval_needed: bool = Field(
        description="Whether retrieval is needed based on the input and context")


class SubinputListClass(BaseModel):
    subinputs: List[SubinputClass] = Field(
        description="List of subinputs to be processed")


def find_first_json_array(text, required_keys=[]):
    # Regex to find JSON arrays
    json_candidates = re.findall(r'\[.*?\]', text, re.DOTALL)

    for candidate in json_candidates:
        print(candidate)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list) and all(isinstance(obj, dict) and all(k in obj for k in required_keys) for obj in parsed):
                return parsed  # Return the first valid match
        except json.JSONDecodeError:
            continue

    # If no valid JSON array is found, search for JSON objects
    json_candidates = re.findall(r'\{.*?\}', text, re.DOTALL)
    for candidate in json_candidates:
        print(candidate)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and all(k in parsed for k in required_keys):
                return [parsed]  # Return the first valid match
        except json.JSONDecodeError:
            continue

    return None


class RAG_Model_Modular:
    def __init__(self, debug=False, device=None):
        # Models
        self.embeddings = None
        self.vector_store = None
        self.vector_store_path = None

        self.llm_pipeline = None
        self.blip_pipeline = None

        self.document_reranker_pipeline = None

        self.debug = debug
        self.device = device or (1 if cuda.is_available() else 0)
        self.preretrieval_parser = PydanticOutputParser(
            pydantic_object=SubinputListClass)

    def _debug_print(self, *msg):
        """
        Print debug messages
        """
        if self.debug:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")
            print(f"[{timestamp}]", *msg, end="\n\n")

    def _convert_chat_history_to_string(self, messages: dict[str, str]):
        """
        Convert ChatMessageModel to pipeline inputs
        """
        return_string_list = []

        for message in messages:
            user_role = message['userType']
            user_message = message['message']
            return_string_list.append(
                f"{'{'}{user_role}: {user_message}{'}'}'")

        return "[" + "\n".join(return_string_list) + "]"

    def load_embeddings_model(self, model_name: str = "paraphrase-MiniLM-L6-v2", embedding_model_path: str = "embedding_models"):
        """Initialize HuggingFace embeddings."""
        # Check if the model already exists in the cache
        # TODO: load faster without internet why?
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

        self.document_reranker_pipeline = reranker_pipeline

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

        files = text_splitter.split_documents(
            [doc for doc in documents if "metadata" not in doc and doc.metadata.get("source") != "image"])
        images = [
            doc for doc in documents if "metadata" in doc and doc["metadata"]["source"] == "image"]
        docs = files + images

        return docs

    def _preretrieval_query_formatting(self, query: str, chat_history: list[any], attached_file_paths: list[str]):
        """
        Preretrieval
        """
        assert self.llm_pipeline is not None, "LLM pipeline not initialized."
        assert self.document_reranker_pipeline is not None, "Cross-encoder model not initialized."

        self._debug_print("[!] Pre-retrieval - Query formatting...")

        self._debug_print(
            "[!] Pre-retrieval - Loading attached files and chat history...")

        # Load attached files and
        attached_files_docs = self._load_input_documents(attached_file_paths)
        attached_images = [
            doc for doc in attached_files_docs if "metadata" in doc and doc["metadata"]["source"] == "image"]
        attached_files_docs_without_images = [
            doc for doc in attached_files_docs if "metadata" not in doc and doc.metadata.get("source") != "image"]

        # Load the chat history
        # chat_history

        # Merging everything into one single user query
        formatted_query = [
            {"role": "system", "content": (
                f"Attached files: {attached_files_docs_without_images}"
                f"Attached images: {attached_images}"
                f"Chat History: {self._convert_chat_history_to_string(chat_history)}"
            )},
        ]

        self._debug_print(
            f"[!] Pre-retrieval - Querying LLM for input decomposition: {query}"
        )

        # TODO: cant  output json ffs
        system_prompt = (
            "You are an input decomposition expert at converting user questions into smaller inputs. "
            "You have access to a collection of course materials for a university course. "
            "You have access to the chat history and any attached files and images, but these may not be relevant"
            ""
            "Perform input decomposition. Break down the user input into smaller, more manageable subinputs, when necessary. "
            "Only output between 1 to 3 subinputs. "
            "If you think the input is clear and does not need decomposition, return the input as it is as a single subinput."
            ""
            "Do NOT add new information, make assumptions, or create your own questions. "
            "If rephrasing is not possible without assumptions, return the question exactly as it is."
            "Do NOT answer the question. Just rephrase it."
            "Do NOT change the meaning or context of the question."
            ""
            f"{self.preretrieval_parser.get_format_instructions()}"
        )
        # system_prompt = (
        #     "You are an input decomposition expert at converting user questions into smaller inputs. "
        #     "You have access to a collection of course materials for a university course. "
        #     "You have access to the chat history and any attached files and images, but these may not be relevant"
        #     ""
        #     "Perform input decomposition. Break down the user input into smaller, more manageable subinputs, when necessary. "
        #     "Only output between 1 to 3 subinputs. "
        #     "If you think the input is clear and does not need decomposition, return the input as it is as a single subinput."
        #     ""
        #     "Do NOT add new information, make assumptions, or create your own questions. "
        #     "If rephrasing is not possible without assumptions, return the question exactly as it is."
        #     "Do NOT answer the question. Just rephrase it."
        #     "Do NOT change the meaning or context of the question."
        #     ""
        #     "IMPORTANT: Your output MUST ONLY be a JSON array of valid dictionary object with exactly two keys:"
        #     '"subinput": A string subinput.'
        #     '"retrieval_needed": A true or false boolean indicating whether retrieval is needed based on the input and context. only set it to true when necessary.'
        # )

        reformulated_query = [
            {"role": "system", "content": system_prompt}] + formatted_query + [{
                "role": "user",
                "content": query
            }]

        response = self.llm_pipeline(reformulated_query)[
            0]['generated_text'][-1]['content']

        self._debug_print(
            f"[!] Pre-retrieval - Response from LLM: \n{response}")

        try:
            parsed_json = find_first_json_array(
                response, ["subinput", "retrieval_needed"])

            self._debug_print(
                f"[!] Pre-retrieval - Parsed JSON: \n{parsed_json}")

            if parsed_json is None:
                raise ValueError("Response is not a list")

            # Check if is array / list
            if not isinstance(parsed_json, list):
                raise ValueError("Response is not a list")

            # Check if each element is a dictionary that contains the keys "subinput" and "retrieval_needed"
            for subquery_object in parsed_json:
                if not ("subinput" in subquery_object and "retrieval_needed" in subquery_object):
                    raise ValueError(
                        "Subquery object does not contain the required keys")

                if not isinstance(subquery_object["subinput"], str):
                    raise ValueError("subinput is not a string")

                if not isinstance(subquery_object["retrieval_needed"], bool):
                    # Try to convert to boolean, if cannot then set to false

                    try:
                        subquery_object["retrieval_needed"] = bool(
                            subquery_object["retrieval_needed"])
                    except ValueError:
                        subquery_object["retrieval_needed"] = False

            subqueries_dict = []
            for subquery_object in parsed_json:
                subinput = subquery_object["subinput"].strip()
                retrieval_needed = subquery_object["retrieval_needed"]
                subqueries_dict.append(
                    {"subquery": subinput, "retrieval_needed": retrieval_needed})

        except Exception as e:
            self._debug_print(
                f'[!] Pre-retrieval - Error parsing response: {e}'
            )
            self._debug_print(
                '[!] Pre-retrieval - Error parsing response, returning default subqueries')
            subqueries_dict = [
                {"subquery": response, "retrieval_needed": False}]

        self._debug_print(
            f"[!] Pre-retrieval - Reformulated Subqueries: \n{subqueries_dict}")

        return subqueries_dict

    def _retrieval(self, subqueries: list[dict]):
        """
        Retrieval and post-retrieval processing.
        """
        assert self.vector_store is not None, "Vector store not initialized."
        assert self.llm_pipeline is not None, "LLM pipeline not initialized."
        assert self.document_reranker_pipeline is not None, "Cross-encoder model not initialized."

        self._debug_print(
            "[!] Retrieval - Searching, filtering and ranking for relevant documents...")

        # Sparse and dense retrieval
        for subquery_object in subqueries:
            if not ("subquery" in subquery_object and "retrieval_needed" in subquery_object):
                continue

            subquery = subquery_object["subquery"]
            retrieval_needed = subquery_object["retrieval_needed"]

            # Retrieve relevant documents based on the subquery
            if retrieval_needed:
                context = self.vector_store.similarity_search(
                    subquery, k=4)  # Retrieve top-k
                filtered_context = self._filter_rerank_with_llm(
                    context, subquery)
                subquery_object["context"] = filtered_context

        self._debug_print(
            "[!] Retrieval - Reranking and filtering complete. Subqueries: \n", subqueries)

        return subqueries

    def _filter_rerank_with_llm(self, context: list[Document], subquery: str):
        """
        Use the LLM to judge the relevance of each document in the context and rerank them.
        """
        if not context:
            return []

        self._debug_print(
            f"[!] Filter and Rerank - Filtering and reranking {len(context)} documents...")

        # Prepare prompt for LLM to evaluate documents
        # TODO: Fix this shit that cant evaluate for shit
        system_prompt = (
            "You are given a subinput and a list of university courses documents. For each document, decide if it is relevant to the subinput."
            "Each document has an index number."
            "Return a JSON object containing the document and a relevance score (0.0 to 0.99). "
            "If it is not relevant, then return 0"
            "List the documents in a JSON array of which contains a dictionary object with the following"
            "keys: 'document': the index number, and 'relevance_score': the score."
            "Only return a JSON array, do NOT return anything else"
            "Do NOT add any other words, headings, explanations, just return the array"
        )

        formatted_documents = "\n".join(
            [f"{i}: {doc.page_content}" for i,
                doc in enumerate(context)]
        )

        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"The List of Documents: {formatted_documents}"},
            {"role": "user", "content": f"The Subinput: {subquery}"}
        ]

        # Get response from LLM
        response = self.llm_pipeline(
            prompt)[0]['generated_text'][-1]['content']
        self._debug_print(
            f"[!] Filter and Rerank - LLM response: \n{response}")

        try:
            parsed_response = find_first_json_array(
                response, ['document', 'relevance_score'])
            if parsed_response is None:
                raise ValueError("Response is not a list")

            final_context = []

            for doc in parsed_response:
                index_number = doc['document']
                relevance_score = doc['relevance_score']
                document = context[index_number]

                if relevance_score > 0.5:
                    final_context.append(document)

            return final_context

        except (json.JSONDecodeError, KeyError):
            self._debug_print(
                f"[!] Failed to parse relevance data: {response}")
            return []

    def _generation(self, subqueries: list[dict]):
        """
        Generation and post-generation processing.
        """
        assert self.llm_pipeline is not None, "LLM pipeline not initialized."

        self._debug_print(
            "[!] Generation - Generating and combining responses...")

        # Generate responses for each subquery
        for subquery_object in subqueries:
            if not ("subquery" in subquery_object):
                self._debug_print(
                    "[!] Generation - Subquery not found in subquery object. Weird :/")
                continue

            subquery = subquery_object["subquery"]

            if not ("context" in subquery_object):
                context = []
            else:
                context = subquery_object["context"]

            # Generate response based on the subquery and context
            self._debug_print(
                f'[!] Generation - Generating response for subquery: {subquery}')

            system_prompt = (
                "You are a knowledgeable and professional Teaching Assistant Chatbot at a university with perfect grammar."
                "Always use the given context unless it is irrelevant to the question. Give concise answers using at most three sentences."
                "Always provide answers that are concise, accurate, and confidently stated, without referencing the source document or context explicitly."
                ""
                "1. Always explain information clearly and assertively. Avoid tentative or overly speculative language."
                "2. If there is insufficient context, summarise what is available and politely ask for more specific details, avoiding mention of a missing document or guide."
                "3. For general questions without specific context, provide direct and accurate answers using your knowledge."
                "4. For casual conversations, maintain a warm and professional tone, responding appropriately to greetings and social dialogue."
                "5. Format all responses in markdown when necessary to ensure clarity and proper presentation."
                "6. If no relevant answer can be provided, respond with a friendly greeting or ask for clarification."
                ""
            )
            response = self.llm_pipeline(
                [{"role": "system", "content": system_prompt},
                    {"role": "system", "content": f"Context: {context}"},
                    {"role": "user", "content": f"Input: {subquery}"}]
            )[0]['generated_text'][-1]['content']

            self._debug_print(f"[!] Generation - Response: {response}")

            subquery_object["response"] = response

        self._debug_print(
            "[!] Generation - Responses generated. Subqueries: \n", subqueries)

        return subqueries

    def _post_generation_processing(self, subqueries: list[dict]):
        """
        Post-generation processing and formatting.
        """
        self._debug_print(
            "[!] Post-generation - Processing and formatting responses...")

        # Combine responses with Subquery - response pair
        combined_responses = []
        for subquery_object in subqueries:
            if not ("subquery" in subquery_object and "response" in subquery_object):
                self._debug_print(
                    "[!] Post-generation - Subquery or response not found in subquery object. Weird :/")
                continue

            combined_responses.append(subquery_object["response"])

        # Final prompt to LLM to combine all responses
        # TODO: Fix this fcking problem where the fcking LLM keep on mess shit up on the final output
        system_prompt = (
            "You are a knowledgeable and professional Teaching Assistant Chatbot at a university with perfect grammar."
            "Always use the given context unless it is irrelevant to the question. Give concise answers using at most three sentences."
            "Always provide answers that are concise, accurate, and confidently stated, without referencing the source document or context explicitly."
            ""
            "1. Always explain information clearly and assertively. Avoid tentative or overly speculative language."
            "2. If there is insufficient context, summarise what is available and politely ask for more specific details, avoiding mention of a missing document or guide."
            "3. For general questions without specific context, provide direct and accurate answers using your knowledge."
            "4. For casual conversations, maintain a warm and professional tone, responding appropriately to greetings and social dialogue."
            "5. Format all responses in markdown when necessary to ensure clarity and proper presentation."
            "6. If no relevant answer can be provided, respond with a friendly greeting or ask for clarification."
            ""
        )

        user_query = f"Given a list of responses, help me combine all into one response. You are to only answer with the combined response. Do not add headers."

        self._debug_print(
            "[!] Post-generation - Combine query: \n", combined_responses)

        response = self.llm_pipeline(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": (
                 f"{user_query} Refer to the list of Responses: {combined_responses} "
                 )}
            ]
        )[0]['generated_text'][-1]['content']

        self._debug_print(f"[!] Post-generation - Final response: {response}")

        return response

    def query(self, query: str, chat_history: list[any], chat_history_truncate_num=5, search_k=10, top_k=2, attached_file_paths=[]):
        """
        Query the QA chain with the given input.

        Search for the most relevant documents based on the query and chat history.
        Return a bunch of documents based on the search results.
        Then, rank the documents based on relevance to the query and return the top-k documents.
        """
        assert self.vector_store is not None, "Vector store not initialized."
        assert self.llm_pipeline is not None, "LLM pipeline not initialized."
        assert self.document_reranker_pipeline is not None, "Cross-encoder model not initialized."

        self._debug_print("Querying the QA chain...")

        # Call all the functions in order
        pre_retrieval_subqueries = self._preretrieval_query_formatting(
            query, chat_history[-chat_history_truncate_num:], attached_file_paths)
        retrieval_subqueries = self._retrieval(pre_retrieval_subqueries)
        generated_responses = self._generation(retrieval_subqueries)
        final_response = self._post_generation_processing(generated_responses)

        self._debug_print(f"Final response: {final_response}")

        return final_response

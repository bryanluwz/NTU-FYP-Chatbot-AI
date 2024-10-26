"""
1. Retrieve Context
	1.0 (Extra Feature) If user input files, convert into text using Clip for images, and pdf/txt loader to convert into text
	1.1 Uses text2text-generation to summarise chat_hisory[:5], other relevant context, and user query
	1.2 Uses summary to retrieve context 
	1.3 If no context is found, then nevermind
2. Retrieve Answer
	2.1 Uses question-answering to retrieve answer based on context
	2.2 Alternatively, uses text-generation to generate answer based on context
"""

from datetime import datetime
import os
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
import gc
import torch


from src.functions import convert_to_langchain_messages

from .document_loader import load_documents


class Multi_Chain_QA:
    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self.vector_store_path = None
        self.llm = None
        self.summarizer = None

        self.text_gen_pipeline = None
        self.summarizer_pipeline = None

        self.qa_chain = None

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

    def initialize_llm(self, model_name: str = 'distilgpt2', max_new_tokens: int = 1024, temperature: float = 0.7, model_path: str = None, task: str = "question-answering"):
        """Initialize the HuggingFace pipeline for text generation, and save/load the model."""
        model_save_path = os.path.join(model_path, model_name)

        if self.llm is not None:
            del self.llm
            gc.collect()

        # Check if the model is already saved
        if os.path.exists(model_save_path):
            print(f"🔄 Loading model from {model_save_path}...")
            text_gen_pipeline = pipeline(
                task='question-answering',
                model=model_save_path,
                tokenizer=model_save_path,
                max_new_tokens=max_new_tokens,
                framework="pt",
                device_map="balanced_low_0"
            )
        else:
            # Get the model size before downloading
            print(
                f"⬇️ Downloading and saving model '{model_name}' to {model_save_path}...")
            text_gen_pipeline = pipeline(
                task='question-answering',
                model=model_name,
                tokenizer=model_name,
                framework="pt",
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                device_map="balanced_low_0"
            )

            # Save the model and tokenizer
            text_gen_pipeline.model.save_pretrained(model_save_path)
            text_gen_pipeline.tokenizer.save_pretrained(model_save_path)
            print(f"✅ Model '{model_name}' saved to {model_save_path}.")

        self.text_gen_pipeline = text_gen_pipeline

        # self.llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
        self.llm = text_gen_pipeline

    def initialize_summarizer(self, model_name: str = "google/flan-t5-base", model_path: str = None, task="text2text-generation"):
        """Initialize the HuggingFace pipeline for summarization."""
        model_save_path = os.path.join(model_path, model_name)

        if self.summarizer is not None:
            del self.summarizer
            gc.collect()

        if os.path.exists(model_save_path):
            print(f"🔄 Loading model from {model_save_path}...")
            summarizer = pipeline(
                task=task,
                model=model_save_path,
                tokenizer=model_save_path,
                framework="pt",
                device_map="balanced_low_0"
            )

        else:
            print(
                f"⬇️ Downloading and saving model '{model_name}' to {model_save_path}...")
            summarizer = pipeline(
                task=task,
                model=model_name,
                tokenizer=model_name,
                framework="pt",
                device_map="balanced_low_0"
            )

            summarizer.model.save_pretrained(model_save_path)
            summarizer.tokenizer.save_pretrained(model_save_path)
            print(f"✅ Model '{model_name}' saved to {model_save_path}.")

        self.summarizer_pipeline = summarizer

        self.summarizer = HuggingFacePipeline(pipeline=summarizer)

    def initialize_qa_chain(self, top_k: int = 2):
        """Initialize the RetrievalQA chain with the given LLM and vectorstore."""
        assert self.llm is not None, "LLM model not initialized."
        assert self.summarizer is not None, "Summarizer model not initialized."
        assert self.vector_store is not None, "Vector store not initialized."

        if self.qa_chain is not None:
            del self.qa_chain
            gc.collect()

        # History + context + user input -> similar question
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            self.summarizer, self.vector_store.as_retriever(
                search_kwargs={"k": top_k,
                               "search_type": "similarity_score_threshold",
                               "score_threshold": 0.8}), contextualize_q_prompt
        )

        # Text gen
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

        self.qa_chain = history_aware_retriever, prompt_template

    def query(self, query, chat_history, chat_history_truncate_num=5):
        """Query the QA chain with the given input."""
        assert self.qa_chain is not None, "QA chain not initialized."

        truncated_chat_history = chat_history[:chat_history_truncate_num]

        timestamp = datetime.now().strftime("%Y%m%d %H%M%S")
        print(f"[{timestamp}] Querying the QA chain...")

        output = self.llm({
            "context": "\n\n".join([doc.content for doc in self.qa_chain[0].invoke({
                "chat_history": convert_to_langchain_messages(truncated_chat_history),
                "input": query
            })]),
            "question": self.qa_chain[1].invoke({
                "input": query
            }).to_string()
        })

        timestamp = datetime.now().strftime("%Y%m%d %H%M%S")
        print(f"[{timestamp}] Query complete.")
        return output

    def kys(self):
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
            del self.summarizer
            del self.summarizer_pipeline

            # Set all to None
            self.__init__()
        except Exception as e:
            print(f"[!] Error while destroying the QA chain: {e}")
            pass

        # Run garbage collection
        gc.collect()

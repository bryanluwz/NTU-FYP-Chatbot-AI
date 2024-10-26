import os
import shutil
from typing import Literal
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.retrieval import create_retrieval_chain
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
import gc
import torch

from .document_loader import load_documents


class ChatMessageModel:
    message: str
    userType: Literal['system', 'human', 'ai']


def convert_to_langchain_messages(messages: list[ChatMessageModel]):
    return [(m['userType'], m['message']) for m in messages]


def convert_to_langchain_messages_alt(messages: list[dict]):
    return [{
        "role": m['userType'],
        "content": m['message']
    } for m in messages]

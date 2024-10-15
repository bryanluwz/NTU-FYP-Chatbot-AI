import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader


def load_documents(file_paths):
    """Load multiple documents from various formats into a single list of documents."""
    documents = []

    for file_path in file_paths:
        if (not os.path.exists(file_path)):
            print(f"❌ File not found: {file_path}")
            continue
        print(f"📄 Loading: {file_path}")
        ext = os.path.splitext(file_path)[1].lower()  # Get file extension

        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.docx':
            loader = Docx2txtLoader(file_path)
        elif ext == '.txt':
            loader = TextLoader(file_path)
        else:
            print(f"❌ Unsupported file format: {ext}")
            continue

        documents.extend(loader.load())  # Load document and add to list

    return documents

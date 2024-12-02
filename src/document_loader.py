import os

from langchain_core.documents.base import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader


def load_documents(file_paths, describe_image_callback=None):
    """Load multiple documents from various formats into a single list of documents with metadata."""
    documents = []

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue
        print(f"📄 Loading: {file_path}")
        ext = os.path.splitext(file_path)[1].lower()  # Get file extension

        is_image = False

        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.docx':
            loader = Docx2txtLoader(file_path)
        elif ext == '.txt':
            loader = TextLoader(file_path)
        elif ext in ['.png', '.jpg', '.jpeg']:
            if describe_image_callback is not None:
                description = describe_image_callback(file_path)
                # Should be true if it works
                is_image = "text" in description and "description" in description
                print(
                    f"[?] Image file is under testing for support, weird things might happen: {ext}")
            else:
                print(
                    f"❌ Unsupported file format: {ext} (Image not supported)")
                continue
        else:
            print(f"❌ Unsupported file format: {ext}")
            continue

        if is_image:
            # Convert the description into a document format
            doc = Document()
            doc.text = description["description"] + description["text"]
            doc.metadata = {
                "file_name": os.path.basename(file_path),
                "file_path": file_path,
                "description": description["description"],
                # Add other metadata as needed
            }
            documents.append(doc)

        else:
            loaded_docs = loader.load()  # Load document
            for doc in loaded_docs:
                doc.metadata = {
                    "file_name": os.path.basename(file_path),
                    "file_path": file_path,
                    # Add other metadata as needed
                }
                documents.append(doc)

    return documents

import os

from langchain_core.documents.base import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

FILE_TYPES = {
    'pdf': 'document',
    'docx': 'document',
    'txt': 'document',
    'py': 'document',
    'csv': 'document',
    'json': 'document',
    'xml': 'document',
    'html': 'document',
    'css': 'document',
    'js': 'document',
    'ts': 'document',
    'c': 'document',
    'cpp': 'document',
    'h': 'document',
    'hpp': 'document',
    'java': 'document',
    'kt': 'document',
    'swift': 'document',
    'rb': 'document',
    'php': 'document',
    'go': 'document',
    'rs': 'document',
    'pl': 'document',
    'sh': 'document',
    'bat': 'document',
    'ps1': 'document',
    'psm1': 'document',
    'psd1': 'document',
    'ps1xml': 'document',
    'pssc': 'document',
    'psc1': 'document',
    'png': 'image',
    'jpg': 'image',
    'jpeg': 'image',
    'gif': 'image',
    'bmp': 'image',
    'tiff': 'image',
    'webp': 'image',
    'mp3': 'audio',
    'wav': 'audio',
    'flac': 'audio',
    'ogg': 'audio',
    'm4a': 'audio',
    'wma': 'audio',
    'aac': 'audio',
    'aiff': 'audio',
    'alac': 'audio',
    'amr': 'audio',
    'au': 'audio',
    'awb': 'audio',
    'dct': 'audio',
    'dss': 'audio',
    'dvf': 'audio',
    'gsm': 'audio',
    'iklax': 'audio',
    'ivs': 'audio',
    'm4p': 'audio',
    'mmf': 'audio',
    'mpc': 'audio',
    'msv': 'audio',
    'nmf': 'audio',
    'nsf': 'audio',
    'oga': 'audio',
    'mogg': 'audio',
    'opus': 'audio',
    'ra': 'audio',
    'rm': 'audio',
    'raw': 'audio',
    'rf64': 'audio',
    'sln': 'audio',
    'tta': 'audio',
    'voc': 'audio',
    'vox': 'audio',
    'wv': 'audio',
    'webm': 'audio',
    '8svx': 'audio',
    'cda': 'audio',
    'mid': 'audio',
    'midi': 'audio',
    'rmi': 'audio',
    'kar': 'audio',
    'mka': 'audio',
    'mpa': 'audio',
    'mp2': 'audio',
    'm2a': 'audio',
    'm3a': 'audio',
    'm4b': 'audio',
    'm4r': 'audio',
    'm4v': 'audio',
    '3ga': 'audio',
    'aa': 'audio',
    'aax': 'audio',
    'act': 'audio',
    'ape': 'audio',
}


# TODO: Is there a better way to do file extension checking?
def load_documents(file_paths, describe_image_callback=None):
    """Load multiple documents from various formats into a single list of documents with metadata."""
    documents = []

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue
        print(f"📄 Loading: {file_path}")
        ext = os.path.splitext(file_path)[1].lower()  # Get file extension

        # Remove '.' if it exists I'm not too sure either
        ext = ext.replace('.', '')

        file_type = FILE_TYPES.get(ext)
        loader = None

        if file_type == 'document':
            if ext == 'pdf':
                loader = PyPDFLoader(file_path)
            elif ext == 'docx':
                loader = Docx2txtLoader(file_path)
            elif ext == 'txt' or ext in ['py', 'csv', 'json', 'xml', 'html', 'css', 'js', 'ts', 'c', 'cpp', 'h', 'hpp', 'java', 'kt', 'swift', 'rb', 'php', 'go', 'rs', 'pl', 'sh', 'bat', 'ps1', 'psm1', 'psd1', 'ps1xml', 'pssc', 'psc1']:
                loader = TextLoader(file_path)
            if loader:
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata = {
                        "file_name": os.path.basename(file_path),
                        "file_path": file_path,
                    }
                    documents.append(doc)
        elif file_type == 'image':
            if describe_image_callback is not None:
                description = describe_image_callback(file_path)
                is_image = "text" in description and "description" in description
                if is_image:
                    print(
                        f"[?] Image file is under testing for support, weird things might happen: {ext}")
                    doc = Document(
                        f'description: {description["description"]}, ocr text: {description["text"]}')
                    doc.metadata = {
                        "file_name": os.path.basename(file_path),
                        "file_path": file_path,
                    }
                    documents.append(doc)
                else:
                    print(f"❌ Unsupported image format: {ext}")
            else:
                print(
                    f"❌ Unsupported image format: {ext} (Image not supported)")
        elif file_type == 'audio':
            print(f"❌ Unsupported file format: {ext} (Audio not supported)")
        else:
            print(f"❌ Unsupported file format: {ext}")

    return documents

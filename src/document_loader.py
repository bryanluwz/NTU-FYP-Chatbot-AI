from docx import Document
import fitz
import os

import fitz  # PyMuPDF
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


def extract_images_from_pdf(pdf_path):
    """Extract images from a PDF and return a list of (page_number, image_path)."""
    doc = fitz.open(pdf_path)
    base_dir = os.path.splitext(pdf_path)[0] + "_images"
    os.makedirs(base_dir, exist_ok=True)
    image_data = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_data = base_image["image"]
            img_ext = base_image["ext"]
            img_path = os.path.join(
                base_dir, f"page_{page_num + 1}_img_{img_index + 1}.{img_ext}")
            with open(img_path, "wb") as f:
                f.write(img_data)
            image_data.append((page_num + 1, img_path))

    return image_data


def extract_images_from_docx(docx_path):
    """Extract images from a DOCX file with their order of appearance."""
    document = Document(docx_path)
    base_dir = os.path.splitext(docx_path)[0] + "_images"
    os.makedirs(base_dir, exist_ok=True)
    image_data = []

    for i, rel in enumerate(document.part.rels):
        if "image" in document.part.rels[rel].target_ref:
            img_path = os.path.join(base_dir, f"image_{i + 1}.png")
            with open(img_path, "wb") as f:
                f.write(document.part.rels[rel].target_part.blob)
            image_data.append((i + 1, img_path))

    return image_data


def load_documents(file_paths, describe_image_callback=None, debug_print=False):
    """Load multiple documents from various formats into a single list of documents with metadata."""
    text_docs = []

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue
        print(f"📄 Loading: {file_path}")
        ext = os.path.splitext(file_path)[1].lower().replace('.', '')

        file_type = FILE_TYPES.get(ext)
        loader = None
        image_data = []

        if file_type == 'document':
            if ext == 'pdf':
                loader = PyPDFLoader(file_path)
                image_data = extract_images_from_pdf(file_path)
            elif ext == 'docx':
                loader = Docx2txtLoader(file_path)
                image_data = extract_images_from_docx(file_path)
            elif ext in ['txt', 'py', 'csv', 'json', 'xml', 'html', 'css', 'js', 'ts', 'c', 'cpp', 'java', 'kt', 'swift']:
                loader = TextLoader(file_path)

            if loader:
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata = {"file_name": os.path.basename(
                        file_path), "file_path": file_path}
                    text_docs.append(doc)

            image_docs = []
            for i, (order, img_path) in enumerate(image_data):
                if describe_image_callback:
                    if debug_print:
                        print(
                            f"🖼️ Describing image progress: {i} / {len(image_data)}")
                    description = describe_image_callback(img_path)
                    image_doc = {
                        "description": description["description"],
                        "text": description["text"],
                        "metadata": {"file_name": os.path.basename(img_path), "file_path": img_path, "source": "image"}
                    }
                    image_docs.append((order, image_doc))

            # Link images to text by page/order
            for order, image_doc in image_docs:
                closest_text_doc = None
                closest_distance = float("inf")

                for text_doc in text_docs:
                    text_page = text_doc.metadata.get("page", 0)
                    if abs(text_page - order) < closest_distance:
                        closest_distance = abs(text_page - order)
                        closest_text_doc = text_doc

                if closest_text_doc:
                    if "linked_image" not in closest_text_doc.metadata:
                        closest_text_doc.metadata["linked_image"] = []
                    closest_text_doc.metadata["linked_image"].append(image_doc)

            if debug_print:
                print(
                    f"🖼️ Created {len(image_docs)} images and {len(text_docs)} text Document objects")
        elif file_type == 'image':
            if describe_image_callback is not None:
                description = describe_image_callback(file_path)
                if "text" in description and "description" in description:
                    print(
                        f"[?] Image file is under testing for support: {ext}")
                    doc = {"description": description["description"], "text": description["text"], "metadata": {
                        "file_name": os.path.basename(file_path), "file_path": file_path, "source": "image"}}
                    text_docs.append(doc)
                else:
                    print(f"❌ Unsupported image format: {ext}")
            else:
                print(
                    f"❌ Unsupported image format: {ext} (Image not supported)")
        elif file_type == 'audio':
            print(f"❌ Unsupported file format: {ext} (Audio not supported)")
        else:
            print(f"❌ Unsupported file format: {ext}")

    return text_docs

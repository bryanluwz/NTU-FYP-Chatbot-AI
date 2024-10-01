import faiss
import numpy as np


def create_faiss_index(dimension):
    index = faiss.IndexFlatL2(dimension)
    return index


def add_vectors_to_index(index, vectors):
    vectors = np.array(vectors).astype('float32')
    index.add(vectors)


def save_faiss_index(index, file_path):
    faiss.write_index(index, file_path)


def load_faiss_index(file_path):
    index = faiss.read_index(file_path)
    return index

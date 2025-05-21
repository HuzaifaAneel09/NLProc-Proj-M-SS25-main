import faiss
import os
import pickle
from typing import List
from sentence_transformers import SentenceTransformer
from utils.utils import load_txt_file, chunk_text

class Retriever:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)
        self.index = None
        self.documents = []
        self.embeddings = []

    def add_documents(self, filepath: str):
        """Loads, chunks, embeds, and indexes the text from a .txt file."""
        text = load_txt_file(filepath)
        chunks = chunk_text(text)
        self.documents.extend(chunks)

        embeddings = self.model.encode(chunks)
        self.embeddings.extend(embeddings)

        if self.index is None:
            self.index = faiss.IndexFlatL2(len(embeddings[0]))
        self.index.add(embeddings)

    def query(self, question: str, top_k: int = 3) -> List[str]:
        """Returns top-k most relevant chunks for a given question."""
        if self.index is None:
            raise ValueError("No documents have been added yet.")
        question_embedding = self.model.encode([question])
        distances, indices = self.index.search(question_embedding, top_k)
        return [self.documents[i] for i in indices[0]]

    def save(self, path: str):
        """Saves FAISS index and document list to disk."""
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)

    def load(self, path: str):
        """Loads FAISS index and document list from disk."""
        self.index = faiss.read_index(os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "documents.pkl"), "rb") as f:
            self.documents = pickle.load(f)

    def initialize_or_load(self, data_path: str, save_path: str):
        """Load index if available, else create from data and save."""
        if os.path.exists(os.path.join(save_path, "faiss.index")):
            print("Loading existing index...")
            self.load(save_path)
        else:
            print("Creating new index...")
            self.add_documents(data_path)
            self.save(save_path)

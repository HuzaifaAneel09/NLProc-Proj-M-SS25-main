import os
from typing import List

def load_txt_file(filepath: str) -> str:
    """Loads plain text from a .txt file."""
    if not os.path.isfile(filepath) or not filepath.endswith('.txt'):
        raise ValueError("Only .txt files are supported.")
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def chunk_text(text: str, chunk_size: int = 100, overlap: int = 50) -> List[str]:
    """Splits text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

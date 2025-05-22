import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

text_chunks = [
    "The Eiffel Tower is located in Paris, France.",
    "The Great Wall of China stretches over 13,000 miles.",
    "The capital of Japan is Tokyo.",
    "Python is a popular programming language for data science.",
    "Pandas is a library used for data manipulation in Python.",
    "Mount Everest is the tallest mountain in the world.",
    "The Taj Mahal was built in the 17th century in India.",
    "The Amazon rainforest is the largest tropical rainforest.",
    "Basketball is a sport played between two teams of five players.",
    "Water boils at 100 degrees Celsius at sea level."
]

model = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = model.encode(text_chunks)

dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(chunk_embeddings))

queries = [
    "Where is the Eiffel Tower?",
    "What city is the Eiffel Tower in?",
    "Tell me the location of the Eiffel Tower."
]

query_embeddings = model.encode(queries)

top_k = 3

for i, query in enumerate(queries):
    print(f"\nQuery {i+1}: {query}")
    D, I = index.search(np.array([query_embeddings[i]]), top_k)

    for rank, idx in enumerate(I[0]):
        print(f"  Rank {rank+1}: {text_chunks[idx]} (distance: {D[0][rank]:.2f})")

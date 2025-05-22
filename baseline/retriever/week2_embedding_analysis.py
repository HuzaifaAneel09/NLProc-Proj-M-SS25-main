import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# 1. Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Sample sentences
sentences = [
    "Who created Python?",
    "Who invented the Python programming language?",
    "Tell me about Guido van Rossum.",
    "What is the capital of France?",
    "Where is Paris located?",
    "Explain how a car engine works.",
    "Describe how an engine functions.",
    "What is machine learning?",
    "Define artificial intelligence.",
    "What is the speed of light?"
]

# 3. Generate embeddings
embeddings = model.encode(sentences)

# 4. Compute cosine similarity matrix
cos_sim_matrix = cosine_similarity(embeddings)

# 5. Plot cosine similarity matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cos_sim_matrix, xticklabels=sentences, yticklabels=sentences, annot=True, fmt=".2f", cmap="Blues")
plt.title("Cosine Similarity Between Sentences")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# 6. PCA visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
for i, label in enumerate(sentences):
    x, y = pca_result[i]
    plt.scatter(x, y)
    plt.text(x + 0.01, y + 0.01, label, fontsize=9)
plt.title("PCA of Sentence Embeddings")
plt.tight_layout()
plt.show()

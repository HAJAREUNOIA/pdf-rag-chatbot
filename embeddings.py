
from sentence_transformers import SentenceTransformer

# Load a pretrained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Example sentences
sentences = [
    "iCaRL learns new classes incrementally.",
    "Incremental learning allows models to learn over time.",
    "The weather is sunny today."
]

# Convert sentences into numerical vectors
embeddings = model.encode(sentences)

print("Number of embeddings:", len(embeddings))
print("Dimensions of each embedding:", len(embeddings[0]))

print("\nFirst 10 values of the first embedding:")
print(embeddings[0][:10])

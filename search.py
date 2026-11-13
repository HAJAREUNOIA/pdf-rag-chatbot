
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Connect to our existing database
client = chromadb.PersistentClient(path="./chroma_db")

# 3. Open the collection
collection = client.get_collection(name="pdf_documents")

# 4. Define a question
question = "How does iCaRL prevent catastrophic forgetting?"

# 5. Convert the question into an embedding
question_embedding = model.encode(question).tolist()

# 6. Search for the 3 most relevant chunks
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3
)

# 7. Display the retrieved chunks
for i, document in enumerate(results["documents"][0]):
    print(f"\n--- Result {i + 1} ---")
    print(document)

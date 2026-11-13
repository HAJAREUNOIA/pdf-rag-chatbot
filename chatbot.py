
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from ollama import chat

# 1. Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
# Load the reranking model
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
# 2. Connect to the existing vector database
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="pdf_documents")


# 3. Start an interactive question-answering loop
while True:

    question = input("\nAsk a question about your PDF (or type 'exit'): ")

    # Stop the chatbot when the user types exit
    if question.strip().lower() == "exit":
        print("Goodbye!")
        break

    # Ignore empty questions
    if not question.strip():
        continue

    # 4. Convert the question into an embedding
    question_embedding = model.encode(question).tolist()

    # 5. Retrieve the 5 nearest chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=10
    )

    # 6. Prepare the retrieved context
    
    # Get the 10 candidate chunks from ChromaDB
    candidate_chunks = results["documents"][0]

    # Create question-passage pairs
    pairs = [(question, chunk) for chunk in candidate_chunks]

    # Calculate a relevance score for each chunk
    scores = reranker.predict(pairs)

    # Combine each chunk with its score
    scored_chunks = list(zip(candidate_chunks, scores))

    # Sort chunks by relevance score (highest first)
    scored_chunks.sort(
    key=lambda item: float(item[1]),
    reverse=True
)

    # Keep the 5 highest-scoring chunks
    top_chunks = scored_chunks[:5]

    # Build the context for Llama
    context = ""

    for i, (chunk, score) in enumerate(top_chunks):
       context += f"\n--- Source passage {i + 1} ---\n"
       context += chunk + "\n"


    # 7. Prepare the prompt
    prompt = f"""
You are a scientific research assistant.

Answer the question using only the provided passages.

Important:
- Distinguish iCaRL from comparison methods.
- Do not invent information.
- If the passages are insufficient, say so.
- Give a clear and accurate explanation.

SOURCE PASSAGES:
{context}

QUESTION:
{question}

ANSWER:
"""

    # 8. Generate an answer with Llama 3.2
    response = chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # 9. Display the answer
    print("\nANSWER:")
    print(response.message.content)

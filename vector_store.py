
import chromadb
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# 1. Read the PDF
reader = PdfReader("data/sample.pdf")

full_text = ""

for page in reader.pages:
    text = page.extract_text()
    if text:
        full_text += text + "\n"

# 2. Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_text(full_text)

# 3. Generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

# 4. Create a persistent ChromaDB database
client = chromadb.PersistentClient(path="./chroma_db")

# 5. Create or open a collection
collection = client.get_or_create_collection(
    name="pdf_documents"
)

# 6. Store the chunks and embeddings
collection.upsert(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings.tolist()
)

# 7. Verify the number of stored chunks
print("Number of stored chunks:", collection.count())

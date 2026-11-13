########     PDF extraction
from pypdf import PdfReader

# Load the pdfs paper
pdf_path = "data/sample.pdf"
reader = PdfReader(pdf_path)

print("Number of pages:", len(reader.pages))

# Extract text from all pages
full_text = ""

for page in reader.pages:
    text = page.extract_text()

    if text:
        full_text += text + "\n"

print("\nExtracted text:\n")
print(full_text[:1000])

###############  Split the PDF text into chunks

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Create a text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Split the extracted text into chunks
chunks = text_splitter.split_text(full_text)

# Display the results
print("\nNumber of chunks:", len(chunks))

print("\nFirst chunk:\n")
print(chunks[0])

print("\nSecond chunk:\n")
print(chunks[1])


###########  Convert chunks into embeddings
from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings for all PDF chunks
chunk_embeddings = model.encode(chunks)##chunk_embeddings is a NumPy array with shape (63, 384)

# Display the results
print("\nNumber of chunk embeddings:", len(chunk_embeddings))
print("Dimensions per embedding:", len(chunk_embeddings[0]))

print("\nFirst 10 values of the first chunk embedding:")
print(chunk_embeddings[0][:10])

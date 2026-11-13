
import hashlib
from io import BytesIO
import re
import ollama
from rank_bm25 import BM25Okapi
import chromadb
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
from ollama import chat


st.set_page_config(
    page_title="DocMind AI | PDF Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)




st.markdown(
    """
    <style>
    /* Overall app */
    .stApp {
        background: #F8FAFC;
        color: #1E293B;
    }

    /* Main content width */
    .block-container {
        max-width: 1050px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }

    /* Headings */
    h1, h2, h3 {
        color: #172554;
        letter-spacing: -0.5px;
    }

    /* Main header */
    .hero {
        background: linear-gradient(
            120deg,
            #FFFFFF 0%,
            #EEF2FF 100%
        );
        border: 1px solid #E0E7FF;
        border-radius: 22px;
        padding: 30px 32px;
        margin-bottom: 24px;
    }

    .hero-label {
        color: #6366F1;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .hero-title {
        color: #1E1B4B;
        font-size: 34px;
        font-weight: 750;
        line-height: 1.2;
        margin-bottom: 12px;
    }

    .hero-description {
        color: #64748B;
        font-size: 15px;
        line-height: 1.7;
        max-width: 620px;
    }

    /* Sidebar branding */
    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 8px 0 20px;
    }

    .brand-icon {
        background: #EEF2FF;
        color: #4F46E5;
        padding: 12px;
        border-radius: 14px;
        font-size: 24px;
    }

    .brand-title {
        color: #1E1B4B;
        font-size: 21px;
        font-weight: 750;
    }

    .brand-subtitle {
        color: #94A3B8;
        font-size: 12px;
    }

    /* Small labels */
    .section-label {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #64748B;
        margin: 16px 0 10px;
    }

    /* Status card */
    .status-card {
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-radius: 12px;
        padding: 13px 15px;
        margin: 14px 0;
        color: #166534;
        font-size: 13px;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 12px 16px;
    }

    /* Buttons */
    .stButton > button {
        background: #FFFFFF;
        color: #4338CA;
        border: 1px solid #C7D2FE;
        border-radius: 11px;
        font-weight: 500;
        min-height: 42px;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: #EEF2FF;
        color: #3730A3;
        border-color: #818CF8;
    }

    /* Chat messages */
    div[data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 16px 18px;
        margin-bottom: 12px;
        border: 1px solid #E2E8F0;
        background: #FFFFFF;
    }

    div[data-testid="stChatMessage"]:has(
        [data-testid="stChatMessageAvatarUser"]
    ) {
        background: #EEF2FF;
        border-color: #C7D2FE;
    }

    /* Chat input */
    div[data-testid="stChatInput"] {
        border-radius: 15px;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        overflow: hidden;
    }

    /* File uploader */
    section[data-testid="stFileUploaderDropzone"] {
        background: #F8FAFF;
        border: 1.5px dashed #A5B4FC;
        border-radius: 14px;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        color: #94A3B8;
        font-size: 12px;
        padding-top: 30px;
        margin-top: 30px;
        border-top: 1px solid #E2E8F0;
    }

    /* Hide Streamlit's default footer */
    footer {
        visibility: hidden;
    }

    @media (max-width: 768px) {
        .hero {
            padding: 22px;
        }

        .hero-title {
            font-size: 27px;
        }

        .block-container {
            padding-top: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_models():

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    return embedding_model, reranker


@st.cache_resource
def load_database():

    return chromadb.PersistentClient(
        path="./chroma_db"
    )


embedding_model, reranker = load_models()
client = load_database()


if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_pdf" not in st.session_state:
    st.session_state.active_pdf = None

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None



def process_pdf(pdf_bytes, pdf_name):

    # Identify each PDF by its contents
    pdf_id = hashlib.sha256(pdf_bytes).hexdigest()

    collection_name = f"pdf_{pdf_id[:32]}"

    collection = client.get_or_create_collection(
        name=collection_name
    )

    # Reuse previously generated embeddings
    if collection.count() > 0:
        return collection, collection.count(), None

    # Extract text
    reader = PdfReader(BytesIO(pdf_bytes))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = []
    metadatas = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        page_text = page.extract_text()

        if not page_text or not page_text.strip():
            continue

        page_chunks = splitter.split_text(page_text)

        for chunk in page_chunks:

            chunks.append(chunk)

            metadatas.append({
                "page": page_number,
                "filename": pdf_name
            })

    if not chunks:
        raise ValueError(
            "No readable text found. "
            "This PDF may require OCR."
        )

    # Convert chunks to embeddings
    embeddings = embedding_model.encode(
        chunks,
        show_progress_bar=False
    )

    # Save chunks in batches
    batch_size = 100

    for start in range(0, len(chunks), batch_size):

        end = min(start + batch_size, len(chunks))

        collection.upsert(
            ids=[
                f"chunk_{i}"
                for i in range(start, end)
            ],
            documents=chunks[start:end],
            embeddings=embeddings[start:end].tolist(),
            metadatas=metadatas[start:end]
        )

    return collection, len(chunks), len(reader.pages)


#def debug_retrieval(question, collection):
    if collection.count() == 0:
        return

    # 1. Semantic search
    embedding = embedding_model.encode(question).tolist()

    semantic = collection.query(
        query_embeddings=[embedding],
        n_results=min(20, collection.count()),
        include=["documents", "metadatas"]
    )

    # 2. BM25 search
    data = collection.get(
        include=["documents", "metadatas"]
    )

    documents = data["documents"]
    metadatas = data["metadatas"]
    ids = data["ids"]

    bm25 = BM25Okapi([
        tokenize(doc) for doc in documents
    ])

    bm25_scores = bm25.get_scores(tokenize(question))

    bm25_order = sorted(
        range(len(documents)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:20]

    # 3. Combine candidates using chunk IDs
    candidates = {}

    for chunk_id, doc, meta in zip(
        semantic["ids"][0],
        semantic["documents"][0],
        semantic["metadatas"][0]
    ):
        candidates[chunk_id] = (doc, meta)

    for i in bm25_order:
        if bm25_scores[i] > 0:
            candidates[ids[i]] = (
                documents[i], metadatas[i]
            )

    # 4. CrossEncoder scores
    candidate_list = list(candidates.values())

    scores = reranker.predict([
        (question, doc)
        for doc, meta in candidate_list
    ])

    reranked = sorted(
        zip(candidate_list, scores),
        key=lambda item: float(item[1]),
        reverse=True
    )

    # 5. Display diagnostic results
    with st.expander("🔍 Retrieval diagnostics"):

        st.write("Semantic search — Top 5")
        for doc, meta in zip(
            semantic["documents"][0][:5],
            semantic["metadatas"][0][:5]
        ):
            st.write(
                f"Page {meta['page']}: {doc[:180]}..."
            )

        st.write("BM25 search — Top 5")
        for i in bm25_order[:5]:
            st.write(
                f"Page {metadatas[i]['page']} "
                f"(BM25: {bm25_scores[i]:.2f}): "
                f"{documents[i][:180]}..."
            )

        st.write("CrossEncoder — Top 5")
        for (doc, meta), score in reranked[:5]:
            st.write(
                f"Page {meta['page']} "
                f"(Score: {float(score):.2f}): "
                f"{doc[:180]}..."
            )

def detect_question_type(question):

    prompt = f"""
You are a question classifier for a PDF research assistant.

Determine whether the user wants:
- SUMMARY: an overview of the entire document
- SPECIFIC: an answer about a particular topic, method,
  result, comparison, or section

Return only SUMMARY or SPECIFIC.

User question: {question}
"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        options={"temperature": 0}
    )

    result = response.message.content.strip().upper()

    if result == "SUMMARY":
        return "SUMMARY"

    return "SPECIFIC"

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())
#"How does Knowledge Distillation work?"
#["how", "does", "knowledge", "distillation", "work"]
#BM25 compares these tokens against the tokens in the PDF chunks.

def select_diverse_passages(scored_chunks, top_k=5):
    selected = []
    remaining = scored_chunks.copy()

    while remaining and len(selected) < top_k:

        if not selected:
            selected.append(remaining.pop(0))
            continue

        best_index = 0
        best_score = float("-inf")

        for i, (doc, meta, relevance) in enumerate(remaining):

            # Compare this passage with already selected passages
            candidate_words = set(tokenize(doc))

            max_similarity = 0

            for selected_doc, _, _ in selected:
                selected_words = set(tokenize(selected_doc))

                union = candidate_words | selected_words

                similarity = (
                    len(candidate_words & selected_words) / len(union)
                    if union else 0
                )

                max_similarity = max(
                    max_similarity, similarity
                )

            # Combine relevance with diversity
            diversity_score = relevance - 5 * max_similarity

            if diversity_score > best_score:
                best_score = diversity_score
                best_index = i

        selected.append(remaining.pop(best_index))

    return selected

def retrieve_passages(question, collection):

    if collection.count() == 0:
        return []

    # Keep our special strategy for summary questions
    
    if detect_question_type(question) == "SUMMARY":

        all_data = collection.get(
            include=["documents", "metadatas"]
        )

        # Keep the original order of chunks
        chunks = list(zip(
            all_data["ids"],
            all_data["documents"],
            all_data["metadatas"]
        ))

        # Our chunk IDs follow the format chunk_0, chunk_1...
        chunks.sort(
            key=lambda item: int(item[0].split("_")[-1])
        )

        # Sample passages across the entire PDF
        max_passages = 12
        step = max(1, len(chunks) // max_passages)

        selected = chunks[::step][:max_passages]

        return [
            (document, metadata, 0.0)
            for _, document, metadata in selected
        ]


    # STEP 1: Semantic search using ChromaDB
    question_embedding = embedding_model.encode(
        question
    ).tolist()

    semantic_results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(20, collection.count()),
        include=["documents", "metadatas"]
    )

    # STEP 2: BM25 keyword search
    all_data = collection.get(
        include=["documents", "metadatas"]
    )

    documents = all_data["documents"]
    metadatas = all_data["metadatas"]
    ids = all_data["ids"]

    tokenized_documents = [
        tokenize(doc) for doc in documents
    ]

    bm25 = BM25Okapi(tokenized_documents)

    query_tokens = tokenize(question)

    bm25_scores = bm25.get_scores(query_tokens)

    # Get the indices of the 20 best BM25 results
    ranked_indices = sorted(
        range(len(documents)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )

    bm25_indices = [
        i for i in ranked_indices
        if bm25_scores[i] > 0
    ][:20]

    # STEP 3: Merge results without duplicates
    candidates = {}

    for chunk_id, doc, meta in zip(
        semantic_results["ids"][0],
        semantic_results["documents"][0],
        semantic_results["metadatas"][0]
    ):
        candidates[chunk_id] = (doc, meta)

    for i in bm25_indices:
        candidates[ids[i]] = (
            documents[i],
            metadatas[i]
        )

    # STEP 4: Rerank using CrossEncoder
    candidate_chunks = list(candidates.values())

    if not candidate_chunks:
        return []

    pairs = [
        (question, doc)
        for doc, meta in candidate_chunks
    ]

    scores = reranker.predict(pairs)

    scored_chunks = [
        (doc, meta, float(score))
        for (doc, meta), score in zip(
            candidate_chunks, scores
        )
    ]

    scored_chunks.sort(
        key=lambda item: item[2],
        reverse=True
    )

    # STEP 5: Return the 5 best passages
    return select_diverse_passages(scored_chunks, top_k=5)



def generate_answer(question, top_chunks):

    context = ""

    for i, (chunk, metadata, score) in enumerate(
        top_chunks,
        start=1
    ):

        page_number = metadata.get("page", "Unknown")

        context += (
            f"\n--- Source {i} "
            f"(Page {page_number}) ---\n"
            f"{chunk}\n"
        )

    prompt = f"""
You are DocMind AI, a helpful research assistant.

Answer the user's question using only the
information provided in the PDF passages.
RULES:
1. Give a clear and accurate answer.
2. Explain important mechanisms when relevant.
3. Do not invent facts.
4. Distinguish the proposed method from methods
   mentioned only for comparison.
5. If the passages are insufficient, say so.
6. Mention relevant page numbers when appropriate.
7. Do not use outside knowledge.
8. Use simple, understandable language.


SOURCE PASSAGES:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    response = chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.message.content


collection = None
chunk_count = None
page_count = None

with st.sidebar:

    st.title("⚙️ Workspace")

    st.caption("Manage your research session")

    st.divider()

    st.subheader("📄 Current document")

    # Document information will be displayed
    # after the PDF is uploaded on the main page.

    document_info = st.empty()

    st.divider()

    st.subheader("💬 Conversation")

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

    st.divider()

    st.subheader("🤖 AI Engine")

    st.caption("Model: Llama 3.2 (3B)")
    st.caption("Embeddings: MiniLM-L6-v2")
    st.caption("Vector database: ChromaDB")
    st.caption("Retrieval: Semantic search + reranking")

    st.success("Local AI")


st.title("📚 DocMind AI")

st.write(
    "Your intelligent research assistant. "
    "Upload a PDF, ask questions, and explore "
    "your document with locally powered AI."
)

st.divider()

st.subheader("📄 Upload your document")

uploaded_file = st.file_uploader(
    "Choose a PDF to start your research",
    type=["pdf"],
    help="Upload a text-based PDF document."
)

if uploaded_file is not None:

    pdf_bytes = uploaded_file.getvalue()

    pdf_id = hashlib.sha256(pdf_bytes).hexdigest()

    if st.session_state.active_pdf != pdf_id:

        st.session_state.active_pdf = pdf_id
        st.session_state.messages = []
        st.session_state.pending_question = None

    try:

        with st.spinner("Preparing your document..."):

            collection, chunk_count, page_count = process_pdf(
                pdf_bytes,
                uploaded_file.name
            )

        st.success(
            f"Your document is ready: {uploaded_file.name}"
        )

        # Update the sidebar with document details
        with document_info.container():

            st.write(uploaded_file.name)

            st.metric("Indexed chunks", chunk_count)

            if page_count is not None:
                st.metric("Pages", page_count)
            else:
                st.caption("Previously indexed document")

    except Exception as error:

        st.error(f"Could not process PDF: {error}")

else:

    if st.session_state.active_pdf is not None:

        st.session_state.active_pdf = None
        st.session_state.messages = []
        st.session_state.pending_question = None

    with document_info.container():
        st.caption("No document uploaded yet.")



if collection is None:

    st.markdown("### How it works")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            "📄 Upload\n\n"
            "Choose your research paper or document."
        )

    with col2:
        st.info(
            "🔎 Explore\n\n"
            "Search relevant passages using AI."
        )

    with col3:
        st.info(
            "💬 Understand\n\n"
            "Get answers supported by PDF sources."
        )



else:

    st.subheader("💬 Research Assistant")

    st.caption(
        "Ask questions about your uploaded document. "
        "Answers are generated from retrieved passages."
    )

    # Show suggested questions before first message
    if not st.session_state.messages:

        st.markdown(
            '<div class="section-label">'
            'Try asking'
            '</div>',
            unsafe_allow_html=True
        )

        suggestions = [
            "Summarize the main ideas of this paper.",
            "What problem does this paper address?",
            "Explain the proposed method step by step.",
            "What are the main limitations?"
        ]

        col1, col2 = st.columns(2)

        for i, suggestion in enumerate(suggestions):

            target_column = col1 if i % 2 == 0 else col2

            with target_column:

                if st.button(
                    suggestion,
                    key=f"suggestion_{i}",
                    use_container_width=True
                ):

                    st.session_state.pending_question = (
                        suggestion
                    )

                    st.rerun()

    # Display previous messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            # Display sources from previous answers
            if message.get("sources"):

                with st.expander("📖 View source passages"):

                    for i, source in enumerate(
                        message["sources"],
                        start=1
                    ):

                        st.markdown(
                            f"Source {i} — Page {source['page']}"
                        )

                        st.write(source["text"])

                        st.caption(
                            f"Relevance score: "
                            f"{source['score']:.3f}"
                        )

                        st.divider()

    # Get question from chat input
    typed_question = st.chat_input(
        "Ask anything about your PDF..."
    )

    # Suggested question takes priority when clicked
    question = st.session_state.pending_question

    if question is not None:

        st.session_state.pending_question = None

    else:

        question = typed_question

    # Process question
    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):

            st.markdown(question)

        with st.chat_message("assistant"):

            try:

                with st.spinner(
                    "Searching your document and preparing an answer..."
                ):

                    top_chunks = retrieve_passages(
                        question,
                        collection
                    )
                   # debug_retrieval(question, collection)
                    #if not top_chunks:
                     #   raise ValueError(
                      #      "No relevant passages were found."
                      #  )

                    answer = generate_answer(
                        question,
                        top_chunks
                    )

                st.markdown(answer)

                # Prepare source information
                sources = []

                for chunk, metadata, score in top_chunks:

                    sources.append({
                        "page": metadata.get(
                            "page",
                            "Unknown"
                        ),
                        "text": chunk,
                        "score": float(score)
                    })

                # Display source passages
                with st.expander("📖 View source passages"):

                    for i, source in enumerate(
                        sources,
                        start=1
                    ):

                        st.markdown(
                            f"Source {i} — Page {source['page']}"
                        )

                        st.write(source["text"])

                        st.caption(
                            f"Relevance score: "
                            f"{source['score']:.3f}"
                        )

                        st.divider()

                # Save answer and its sources
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as error:

                st.error(
                    f"Something went wrong: {error}"
                )


st.markdown(
    """
    <div class="app-footer">
        DocMind AI · Local PDF Research Assistant<br>
        Built with Streamlit, ChromaDB,
        SentenceTransformers and Ollama
    </div>
    """,
    unsafe_allow_html=True
)

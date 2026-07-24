# APP FOR COHERE
import streamlit as st
import numpy as np
import tempfile
import cohere

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ----------------------------------
# Load embedding model
# ----------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")


# ----------------------------------
# Function 1: Load and Chunk Document
# ----------------------------------
def load_and_chunk_document(file_path, chunk_size=300, overlap=50):

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)

    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)

    else:
        raise ValueError("Only PDF and TXT files are supported.")

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )

    chunks = splitter.split_documents(documents)

    return [chunk.page_content for chunk in chunks]


# ----------------------------------
# Function 2: Create Embeddings
# ----------------------------------
def create_embeddings(chunks):

    embeddings = model.encode(chunks)

    return embeddings


# ----------------------------------
# Function 3: Semantic Search
# ----------------------------------
def search_chunks(query, chunks, embeddings, k=3):

    query_embedding = model.encode([query])

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    top_indices = np.argsort(similarities)[::-1][:k]

    return [chunks[i] for i in top_indices]


# ----------------------------------
# Function 4: Generate Answer (Task 5)
# ----------------------------------
def generate_answer(query, context, api_key):

    co = cohere.Client(api_key)

    prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the information given in the context.

Context:
{context}

Question:
{query}

Answer:
"""

    response = co.chat(
        model="command-a-03-2025",
        message=prompt
    )

    return response.text


# ==================================
# Streamlit UI
# ==================================

st.set_page_config(page_title="Chat With PDF", layout="wide")
st.title("Chat With PDF")


# Cohere API Key
api_key = st.text_input(
    "Enter Cohere API Key",
    type="password"
)

uploaded_file = st.file_uploader(
    "Upload a PDF or TXT file",
    type=["pdf", "txt"]
)

if uploaded_file is not None:

    suffix = "." + uploaded_file.name.split(".")[-1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:

        tmp.write(uploaded_file.read())

        temp_path = tmp.name

    chunks = load_and_chunk_document(temp_path)

    if len(chunks) == 0:
        st.error("No text could be extracted from the document.")
        st.stop()

    embeddings = create_embeddings(chunks)

    st.success("✅ Document processed successfully!")

    question = st.text_input("Ask a question")

    if st.button("Ask"):

        if api_key.strip() == "":
            st.warning("Please enter your Cohere API Key.")

        elif question.strip() == "":
            st.warning("Please enter a question.")

        else:

            with st.spinner("Searching document..."):

                results = search_chunks(
                    question,
                    chunks,
                    embeddings
                )

                context = "\n\n".join(results)

            with st.spinner("Generating answer using Cohere..."):

                try:

                    answer = generate_answer(
                        question,
                        context,
                        api_key
                    )

                    st.subheader("🤖 Generated Answer")
                    st.write(answer)

                except Exception as e:

                    st.error(f"Cohere Error: {e}")

            st.subheader("📚 Top Relevant Chunks")

            for i, chunk in enumerate(results):

                st.markdown(f"### Chunk {i+1}")
                st.write(chunk)
                st.markdown("---")

import os
import ollama
from pdfminer.high_level import extract_text
from typing import List
from config import AI_MODEL, TOP_K, CHUNK_SIZE, CHUNK_OVERLAP
from utils.chunking import sliding_window_chunking
from utils.embedding import embed_chunks
from utils.retrieval import initialize_chroma, retrieve_and_rerank, merge_results
from utils.logger_config import logger

collection = initialize_chroma()


def process_pdf(pdf_path):
    """Extracts text from a PDF, chunks it, embeds it, and stores it in ChromaDB."""
    logger.debug(f"Processing PDF: {pdf_path}")
    text = extract_text(pdf_path)
    chunks = sliding_window_chunking(text, CHUNK_SIZE, CHUNK_OVERLAP)
    chunk_embeddings = embed_chunks(chunks)

    doc_id = os.path.basename(pdf_path)

    # Retrieve existing IDs to avoid duplicate insertion
    existing_ids = set(collection.get()["ids"])  # Fetch all existing document IDs

    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_{i}"

        # Only add if the chunk ID does not already exist
        if chunk_id not in existing_ids:
            collection.add(
                ids=[chunk_id],
                embeddings=[chunk_embeddings[i]],
                documents=[chunk],
                metadatas=[{"document": doc_id}],
            )
        else:
            logger.debug(
                f"Skipping existing chunk: {chunk_id}"
            )  # Optional debug message

    logger.info(f"Stored {len(chunks)} new chunks from {doc_id}.")


def process_multiple_pdfs(folder_path: str) -> None:
    """
    Processes all PDFs in a given folder.

    Args:
        folder_path (str): The directory containing PDF files.
    """
    pdf_files = [file for file in os.listdir(folder_path) if file.endswith(".pdf")]

    for file in pdf_files:
        process_pdf(os.path.join(folder_path, file))


def query_rag(query: str) -> str:
    """
    Retrieves relevant chunks from ChromaDB and generates an AI response.

    Args:
        query (str): The user query.

    Returns:
        str: The generated response.
    """
    logger.debug(f"Querying: {query}")

    retrieved_chunks = retrieve_and_rerank(query, collection, top_k=TOP_K)

    if not retrieved_chunks:
        return "No relevant information found."

    # Construct retrieved text with metadata
    retrieved_texts = "\n\n".join(
        f"Document: {chunk[1]}, Article: {chunk[2]}\n{chunk[0]}"
        for chunk in retrieved_chunks
    )

    return generate_response(query, retrieved_texts)


def generate_response(query: str, retrieved_text: str) -> str:
    """
    Generates a response from an AI model based on retrieved text.

    Args:
        query (str): The search query.
        retrieved_text (str): The text retrieved from the vector store.

    Returns:
        str: The AI-generated response.
    """
    prompt = f"""
    You are an AI assistant specializing in legal and administrative documents.

    ### **Question**:
    "{query}"

    ### **Retrieved Information**:
    {retrieved_text}

    ### **Instructions**:
    - Read only the retrieved information.
    - Clearly mention the document name and article number when answering.
    - Ignore page numbers and links.
    - Provide an accurate answer based only on the available information.
    - If the information is insufficient, clearly inform the user.

    ### **Answer**:
    """

    response = ollama.chat(
        model=AI_MODEL, messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


def expand_query(query):
    """Expands the query using AI for better document retrieval."""
    response = ollama.chat(
        model=AI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Expand the following query into multiple related search terms.",
            },
            {"role": "user", "content": query},
        ],
    )
    return response["message"]["content"]

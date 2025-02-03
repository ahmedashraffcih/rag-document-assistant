import chromadb
import numpy as np
from sentence_transformers import CrossEncoder
from utils.embedding import embed_chunks
from config import CHROMA_DB_PATH, COLLECTION_NAME
from typing import List, Tuple, Dict
from rank_bm25 import BM25Okapi
from utils.logger_config import logger

# Load reranker model
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def initialize_chroma():
    """Initializes ChromaDB and retrieves the collection and BM25 index."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # Initialize BM25 Index
    global bm25_index, bm25_corpus, bm25_chunks
    bm25_chunks = []  # Store text chunks for BM25
    bm25_corpus = []  # Store tokenized text chunks

    results = collection.get()  # Fetch stored chunks
    if not results["documents"]:
        logger.warning(
            "Warning: No documents found in ChromaDB. BM25 retrieval will be disabled."
        )
        return collection  # Skip BM25 setup if there's no data

    for doc in results["documents"]:
        tokens = doc.split()  # Tokenize chunk text
        bm25_corpus.append(tokens)
        bm25_chunks.append(doc)

    # Rebuild BM25 if we have documents
    if bm25_corpus:
        logger.debug("Initializing BM25 with newly added documents...")
        bm25_index = BM25Okapi(bm25_corpus)
    else:
        logger.warning("Warning: No valid text chunks found for BM25 indexing.")

    return collection


def retrieve_and_rerank(query, collection, top_k=10):
    """Retrieves relevant chunks using BM25 & Embeddings and applies reranking."""

    # Retrieve ChromaDB (Embedding-based search)
    retrieved_chunks = retrieve_similar_text(query, collection, top_k)

    # Retrieve BM25-based results
    bm25_results = retrieve_bm25(query, top_k)

    # Combine Results from BM25 & Embeddings
    combined_chunks = merge_results(retrieved_chunks, bm25_results)

    # Rerank with Cross-Encoder
    if combined_chunks:
        for chunk in combined_chunks:
            logger.debug(f"{chunk[1]} - Article {chunk[2]} | Score: {chunk[4]:.4f}")
        reranked_chunks = rerank_chunks(query, combined_chunks)
        return reranked_chunks
    else:
        return []


def retrieve_bm25(query, top_k=10):
    """Retrieve text chunks using BM25 keyword search."""
    query_tokens = query.split()  # Tokenize query
    scores = bm25_index.get_scores(query_tokens)  # Get BM25 similarity scores
    top_indices = np.argsort(scores)[-top_k:][::-1]  # Get top matches

    retrieved_chunks = [
        (bm25_chunks[i], "BM25", "N/A", i, scores[i]) for i in top_indices
    ]

    return retrieved_chunks


def merge_results(embedding_results, bm25_results):
    """Merge BM25 and embedding search results, prioritizing higher-ranked items."""

    merged = {}

    for chunk in embedding_results + bm25_results:
        chunk_text, doc_name, article, index, score = chunk
        if chunk_text not in merged:
            merged[chunk_text] = (chunk_text, doc_name, article, index, score)
        else:
            # Take the higher score from both sources
            merged_score = max(merged[chunk_text][-1], score)
            merged[chunk_text] = (chunk_text, doc_name, article, index, merged_score)

    # Sort by combined score (higher is better)
    return sorted(merged.values(), key=lambda x: x[-1], reverse=True)


def retrieve_similar_text(
    query: str, collection, top_k: int = 10
) -> List[Tuple[str, str, str, int, float]]:
    """
    Retrieves the most similar text chunks from ChromaDB.

    Args:
        query (str): The search query.
        collection: The ChromaDB collection.
        top_k (int): Number of top matches to retrieve.

    Returns:
        list of tuples: (chunk_text, document_name, article, chunk_index, score).
    """
    query_embedding = embed_chunks([query])[0]
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()

    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    if not results["documents"]:
        return []

    retrieved_chunks = [
        (
            results["documents"][0][i],
            results["metadatas"][0][i]["document"],
            results["metadatas"][0][i].get("article", "N/A"),
            results["metadatas"][0][i].get("chunk_index", i),
            results["distances"][0][i],
        )
        for i in range(len(results["ids"][0]))
    ]

    # Sort by distance score (higher = more relevant)
    return sorted(retrieved_chunks, key=lambda x: x[4], reverse=True)


def rerank_chunks(
    query: str, retrieved_chunks: List[Tuple[str, str, str, int, float]]
) -> List[Tuple[str, str, str, float]]:
    """
    Re-ranks retrieved chunks using a neural ranking model (Cross-Encoder).

    Args:
        query (str): The search query.
        retrieved_chunks (list): Retrieved chunks with metadata.

    Returns:
        list: Re-ranked text chunks.
    """
    if not retrieved_chunks:
        return []

    # Prepare Query-Chunk Pairs for Reranking
    text_pairs = [(query, chunk[0]) for chunk in retrieved_chunks]

    # Compute Relevance Scores Using Cross-Encoder Model
    rerank_scores = reranker_model.predict(text_pairs)

    # Sort Chunks by Neural Relevance Score
    reranked_chunks = sorted(
        zip(retrieved_chunks, rerank_scores), key=lambda x: x[1], reverse=True
    )

    return [
        (chunk[0], chunk[1], chunk[2], rerank_score)
        for chunk, rerank_score in reranked_chunks
    ]

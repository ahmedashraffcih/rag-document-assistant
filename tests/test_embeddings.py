from utils.embedding import embed_chunks
import numpy as np


def test_embed_chunks():
    texts = ["Hello world", "Test embedding"]
    embeddings = embed_chunks(texts)

    assert len(embeddings) == len(texts)
    assert all(isinstance(emb, np.ndarray) for emb in embeddings)

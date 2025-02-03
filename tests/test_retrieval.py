import pytest
import chromadb
from utils.retrieval import initialize_chroma, retrieve_similar_text


@pytest.fixture
def mock_collection():
    client = chromadb.PersistentClient(path="/tmp/test_chroma_db")
    return client.get_or_create_collection(name="test_collection")


def test_initialize_chroma(mock_collection):
    assert mock_collection is not None


def test_retrieve_similar_text(mock_collection):
    query = "What is AI?"
    retrieved_chunks = retrieve_similar_text(query, mock_collection, top_k=5)

    assert isinstance(retrieved_chunks, list)

import pytest
from rag_pipeline import query_rag


def test_query_rag():
    query = "What is the role of AI in legal documents?"
    response = query_rag(query)

    assert isinstance(response, str)
    assert len(response) > 0

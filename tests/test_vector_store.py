import pytest
from src.database.vector_store import EnvironmentalVectorStore

def test_vector_store_initialization_and_ingestion():
    vs = EnvironmentalVectorStore()
    assert len(vs.documents) > 0
    assert len(vs.metadatas) == len(vs.documents)

def test_similarity_search():
    vs = EnvironmentalVectorStore()
    results = vs.search_similar("agroforestry soil organic carbon rainfall", top_k=2)
    assert len(results) > 0
    assert "content" in results[0]
    assert "metadata" in results[0]
    assert results[0]["score"] > 0

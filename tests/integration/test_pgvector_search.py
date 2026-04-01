from pathlib import Path


def test_similarity_query_uses_pgvector_cosine_distance() -> None:
    content = Path('app/services/similarity.py').read_text(encoding='utf-8')
    assert 'cosine_distance' in content
    assert 'top_k' in content

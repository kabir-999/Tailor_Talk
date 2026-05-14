from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def semantic_scores(query: str, documents: list[str]) -> list[float]:
    if not documents:
        return []
    corpus = [query, *documents]
    try:
        matrix = TfidfVectorizer(stop_words="english", max_features=5000).fit_transform(corpus)
        scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
        return [float(score) for score in scores]
    except ValueError:
        return [0.0 for _ in documents]

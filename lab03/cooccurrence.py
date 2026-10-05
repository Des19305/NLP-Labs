from collections import Counter
import re
from typing import Iterable, List, Sequence, Tuple, Dict

import numpy as np


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    if lowercase:
        text = text.lower()
    return re.findall(r"\b[\w']+\b", text, flags=re.UNICODE)


def build_vocabulary(
    corpus_tokens: Sequence[Sequence[str]],
    min_count: int = 1
) -> Tuple[List[str], Dict[str, int]]:
    if min_count < 1:
        raise ValueError("min_count must be >= 1")

    counts = Counter(token for sentence in corpus_tokens for token in sentence)
    vocab = sorted(
        (word for word, count in counts.items() if count >= min_count),
        key=lambda word: (-counts[word], word),
    )
    word_to_index = {word: index for index, word in enumerate(vocab)}
    return vocab, word_to_index


def build_cooccurrence_matrix(
    corpus_tokens: Sequence[Sequence[str]],
    word_to_index: Dict[str, int],
    window_size: int = 1,
    symmetric: bool = True,
    distance_weighting: bool = False,
) -> np.ndarray:
    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    vocab_size = len(word_to_index)
    matrix = np.zeros((vocab_size, vocab_size), dtype=np.float64)

    for sentence in corpus_tokens:
        for target_pos, target in enumerate(sentence):
            if target not in word_to_index:
                continue

            start = max(0, target_pos - window_size)
            end = min(len(sentence), target_pos + window_size + 1)

            for context_pos in range(start, end):
                if context_pos == target_pos:
                    continue
                if not symmetric and context_pos > target_pos:
                    continue

                context = sentence[context_pos]
                if context not in word_to_index:
                    continue

                distance = abs(context_pos - target_pos)
                weight = 1.0 / distance if distance_weighting else 1.0
                matrix[word_to_index[target], word_to_index[context]] += weight

    return matrix


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.shape != y.shape:
        raise ValueError(f"Vector shapes must match, got {x.shape} and {y.shape}")

    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if denominator == 0:
        return 0.0
    return float(np.dot(x, y) / denominator)


def most_similar(
    word: str,
    matrix: np.ndarray,
    vocabulary: Sequence[str],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if matrix.shape[0] != len(vocabulary):
        raise ValueError("Number of matrix rows must equal vocabulary length")

    try:
        query_index = list(vocabulary).index(word)
    except ValueError as exc:
        raise KeyError(f"Word {word!r} is not in the vocabulary") from exc

    query_vector = matrix[query_index]
    scores = []
    for index, candidate in enumerate(vocabulary):
        if index == query_index:
            continue
        score = cosine_similarity(query_vector, matrix[index])
        scores.append((candidate, score))

    scores.sort(key=lambda item: (-item[1], item[0]))
    return scores[:top_k]


def matrix_statistics(matrix: np.ndarray) -> dict:
    return {
        "rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
        "total_entries": int(matrix.size),
        "nonzero_entries": int(np.count_nonzero(matrix)),
        "density": float(np.count_nonzero(matrix) / matrix.size) if matrix.size else 0.0,
    }


if __name__ == "__main__":
    example_corpus = [
        "the cat eats fish",
        "the dog eats fish",
        "the cat likes milk",
        "the dog likes meat",
    ]
    tokenized = [tokenize(sentence) for sentence in example_corpus]
    vocab, word_to_index = build_vocabulary(tokenized)
    matrix = build_cooccurrence_matrix(tokenized, word_to_index, window_size=1)

    print("Vocabulary order:", vocab)
    print("Co-occurrence matrix (rows/columns follow vocabulary order):")
    print(matrix)
    print("Matrix statistics:", matrix_statistics(matrix))
    if "cat" in word_to_index:
        print("Most similar to cat:", most_similar("cat", matrix, vocab, top_k=5))

"""Deterministic lightweight lexical vectors for cosine retrieval without another API."""
import hashlib
import re

import numpy as np

from config import VECTOR_DIMENSIONS


def feature_vector(text):
    words = re.findall(r"[a-z0-9]+", (text or "").casefold())
    features = words + [f"{left}_{right}" for left, right in zip(words, words[1:])]
    vector = np.zeros(VECTOR_DIMENSIONS, dtype=np.float32)
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        index = value % VECTOR_DIMENSIONS
        sign = 1.0 if (value >> 32) & 1 else -1.0
        vector[index] += sign
    norm = np.linalg.norm(vector)
    if norm:
        vector /= norm
    return vector

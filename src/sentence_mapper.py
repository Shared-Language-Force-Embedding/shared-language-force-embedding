import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Any


class SentenceMapper:
    def __init__(self, vocabulary: List[Tuple[str, Any]]):
        self.vocabulary, self.output = zip(*vocabulary)
        self.sbert = SentenceTransformer('all-mpnet-base-v2')
        self.embeddings = self.sbert.encode(vocabulary)

    def map(self, sentence: str | List[str]) -> str | NDArray:
        if type(sentence) is str:
            embedding = self.sbert.encode([sentence])
            cosine_similarity = np.sum(embedding * self.embeddings, axis=-1)
            nearest_index = np.argmax(cosine_similarity)
            return self.output[nearest_index]

        embeddings = self.sbert.encode(sentence)
        return np.array([self.output[np.argmax(np.sum(embedding * self.embeddings, axis=-1))] for embedding in embeddings])

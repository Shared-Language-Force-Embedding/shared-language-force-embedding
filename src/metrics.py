import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class Metrics:
    def __init__(self):
        self.sbert = SentenceTransformer('all-mpnet-base-v2')

    def score_modifier(self, reference: NDArray, prediction: NDArray) -> float:
        embeddings = self.sbert.encode([reference[0], prediction[0]])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return similarity

    def score_direction(self, reference: NDArray, prediction: NDArray) -> float:
        reference = ' '.join(reference[1:]).strip()
        prediction = ' '.join(prediction[1:]).strip()
        embeddings = self.sbert.encode([reference, prediction])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return similarity

    def score_force_profile(self, reference: NDArray, prediction: NDArray) -> float:
        mse = np.mean((reference - prediction) ** 2)
        return mse

    def score_force_profile_direction(self, reference: NDArray, prediction: NDArray) -> float:
        similarity = cosine_similarity([reference[-1]], [prediction[-1]])[0][0]
        return similarity

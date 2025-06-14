import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
from torch import Tensor
from globals import FORCE_SAMPLE_COUNT, FORCE_CHANNEL_COUNT, DEVICE
from sentence_transformers import SentenceTransformer
from typing import List, Tuple


class Embedder:
    def __init__(self, unique_words: NDArray, phrase_length: int):
        self.unique_words = unique_words
        self.word_to_label = np.vectorize(
            {word: i for i, word in enumerate(unique_words)}.get)
        self.label_to_word = np.vectorize(
            {i: word for i, word in enumerate(unique_words)}.get)
        self.phrase_length = phrase_length
        self.embedding_length = len(self.unique_words)

    def _encode_one_hot(self, words: NDArray) -> NDArray:
        classes = self.word_to_label(words)
        one_hot = np.zeros((words.size, len(self.unique_words)), dtype=int)
        one_hot[np.arange(words.size), classes.ravel()] = 1
        return one_hot.reshape(*words.shape, len(self.unique_words))

    def _decode_one_hot(self, embeddings: NDArray) -> NDArray:
        classes = np.argmax(embeddings, axis=-1)
        return self.label_to_word(classes)

    def _encode_label(self, words: NDArray) -> NDArray:
        return self.word_to_label(words)

    def _decode_label(self, labels: NDArray) -> NDArray:
        return self.label_to_word(labels)

    def embed_force(self, force_data: NDArray) -> NDArray:
        raise NotImplementedError()

    def unembed_force(self, X: NDArray) -> NDArray:
        raise NotImplementedError()

    def embed_phrase(self, phrase_data: NDArray) -> NDArray:
        raise NotImplementedError()

    def unembed_phrase(self, Y: NDArray) -> NDArray:
        raise NotImplementedError()


class SVMKNNEmbedder(Embedder):
    def __init__(self, unique_words: NDArray, phrase_length: int):
        super().__init__(unique_words, phrase_length)

    def embed_force(self, force_data: NDArray) -> NDArray:
        return force_data[:, -1, :]

    def unembed_force(self, X: NDArray) -> NDArray:
        force_curve = np.ones(
            (X.shape[0], FORCE_SAMPLE_COUNT, FORCE_CHANNEL_COUNT))
        force_curve *= np.linspace(0, X, FORCE_SAMPLE_COUNT).swapaxes(0, 1)
        return force_curve

    def embed_phrase(self, phrase_data: NDArray) -> NDArray:
        return self._encode_label(phrase_data)

    def unembed_phrase(self, Y: NDArray) -> NDArray:
        return self._decode_label(Y)


class BinaryEmbedder(Embedder):
    def __init__(self, unique_words: NDArray, phrase_length: int):
        super().__init__(unique_words, phrase_length)

    def embed_force(self, force_data: NDArray) -> Tensor:
        X = force_data.reshape(force_data.shape[0], -1)
        return torch.tensor(X, dtype=torch.float32)

    def unembed_force(self, X: Tensor) -> NDArray:
        X = X.cpu().detach().numpy()
        force_curve = X.reshape(X.shape[0], FORCE_SAMPLE_COUNT, -1)
        return force_curve

    def embed_phrase(self, phrase_data: NDArray) -> Tensor:
        Y = self._encode_one_hot(phrase_data)
        Y = Y.reshape(Y.shape[0], -1)
        return torch.tensor(Y, dtype=torch.float32)

    def unembed_phrase(self, Y: Tensor) -> NDArray:
        Y = Y.reshape(Y.shape[0], self.phrase_length, -
                      1).cpu().detach().numpy()
        return self._decode_one_hot(Y)


class GloveEmbedder(Embedder):
    def __init__(self, embeddings_path: str, phrase_length: int):
        embeddings = torch.load(embeddings_path, weights_only=False)

        self.unique_words = embeddings['vocab']
        self.word_to_label = np.vectorize(
            {word: i for i, word in enumerate(self.unique_words)}.get)
        self.label_to_word = np.vectorize(
            {i: word for i, word in enumerate(self.unique_words)}.get)
        self.phrase_length = phrase_length
        self.embedding_layer = nn.Embedding.from_pretrained(
            embeddings['embedding_layer_state_dict']['weight'], freeze=True)
        self.embedding_length = self.embedding_layer.embedding_dim
        self.word_embeddings = self.embedding_layer(
            torch.arange(len(self.unique_words))).to(DEVICE)

    def embed_force(self, force_data: NDArray) -> Tensor:
        X = force_data.reshape(force_data.shape[0], -1)
        return torch.tensor(X, dtype=torch.float32)

    def unembed_force(self, X: Tensor) -> NDArray:
        X = X.cpu().detach().numpy()
        force_curve = X.reshape(X.shape[0], FORCE_SAMPLE_COUNT, -1)
        return force_curve

    def embed_phrase(self, phrase_data: NDArray) -> Tensor:
        Y = phrase_data.copy()
        Y[Y == ''] = '<no_word>'
        Y = torch.tensor(self.word_to_label(Y), dtype=torch.int64)
        Y = self.embedding_layer(Y)
        Y = Y.reshape(Y.shape[0], -1)
        return Y

    def unembed_phrase(self, Y: Tensor) -> NDArray:
        Y = Y.reshape(Y.shape[0], self.phrase_length, -1)

        distances = torch.cdist(
            Y.view(-1, Y.shape[-1]), self.word_embeddings, p=2)
        indices = distances.argmin(dim=-1)
        Y = indices.view(Y.shape[0], Y.shape[1]).cpu().numpy()

        Y = self.label_to_word(Y)
        Y[Y == '<no_word>'] = ''
        return Y


class SBERTEmbedder(Embedder):
    def __init__(self, vocabulary: List[Tuple[str, NDArray]]):
        self.vocabulary, self.output = zip(*vocabulary)
        self.sbert = SentenceTransformer('all-mpnet-base-v2', device='cpu')
        self.embedding_length = 768
        self.phrase_length = 1
        self.embeddings = self.sbert.encode(self.vocabulary)

    def embed_force(self, force_data: NDArray) -> Tensor:
        X = force_data.reshape(force_data.shape[0], -1)
        return torch.tensor(X, dtype=torch.float32)

    def unembed_force(self, X: Tensor) -> NDArray:
        X = X.cpu().detach().numpy()
        force_curve = X.reshape(X.shape[0], FORCE_SAMPLE_COUNT, -1)
        return force_curve

    def embed_phrase(self, phrase_data: NDArray) -> Tensor:
        embeddings = self.sbert.encode(phrase_data)
        return torch.tensor(embeddings, dtype=torch.float32)

    def unembed_phrase(self, Y: Tensor) -> NDArray:
        Y = Y.cpu().detach().numpy()
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        phrases = np.array([
            self.vocabulary[np.argmax(np.sum(embedding * self.embeddings, axis=-1))] for embedding in Y], dtype='U32')
        return phrases

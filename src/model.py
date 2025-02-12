from numpy.typing import NDArray
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from embedder import Embedder
from globals import FORCE_EMBEDDING_DIM, DEVICE


def phrase_cross_entropy(Y_pred: Tensor, Y: Tensor, phrase_length: int) -> Tensor:
    Y_pred = Y_pred.reshape(Y_pred.shape[0], phrase_length, -1)
    Y_pred = F.softmax(Y_pred, dim=-1)
    Y_pred = Y_pred.reshape(Y_pred.shape[0], -1)
    return -torch.mean(Y * torch.log(Y_pred + 1e-9))


def info_nce_loss(Z_f: Tensor, Z_p: Tensor, temperature: float = 0.1) -> Tensor:
    batch_size = Z_f.size(0)

    Z_f_norm = F.normalize(Z_f, dim=1)
    Z_p_norm = F.normalize(Z_p, dim=1)

    similarity_matrix = torch.matmul(Z_f_norm, Z_p_norm.T)
    similarity_matrix = similarity_matrix / temperature

    labels = torch.arange(batch_size).to(Z_f.device)

    loss_f_to_p = F.cross_entropy(similarity_matrix, labels)
    loss_p_to_f = F.cross_entropy(similarity_matrix.T, labels)

    loss = (loss_f_to_p + loss_p_to_f) / 2
    return loss


class Model:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    def train(self, force_data: NDArray, phrase_data: NDArray, epochs: int = 1000, verbose: bool = False) -> None:
        raise NotImplementedError()

    def force_to_phrase(self, force_data: NDArray) -> NDArray:
        raise NotImplementedError()

    def phrase_to_force(self, phrase_data: NDArray) -> NDArray:
        raise NotImplementedError()


class SVMKNNModel(Model):
    def __init__(self, embedder: Embedder):
        super().__init__(embedder)
        self.classifier = MultiOutputClassifier(SVC(C=32.0))
        self.knn = NearestNeighbors()

    def train(self, force_data: NDArray, phrase_data: NDArray, epochs: int = 1000, verbose: bool = False) -> None:
        X = self.embedder.embed_force(force_data)
        Y = self.embedder.embed_phrase(phrase_data)

        self.classifier.fit(X, Y)

        self.knn.fit(Y, X)
        self.trained_X = X

    def force_to_phrase(self, force_data: NDArray) -> NDArray:
        X = self.embedder.embed_force(force_data)
        Y = self.classifier.predict(X)
        return self.embedder.unembed_phrase(Y)

    def phrase_to_force(self, phrase_data: NDArray) -> NDArray:
        Y = self.embedder.embed_phrase(phrase_data)

        distances, indices = self.knn.kneighbors(Y, n_neighbors=1)
        X = self.trained_X[indices[:, 0]]

        return self.embedder.unembed_force(X)


class MLPModel(Model):
    def __init__(self, embedder: Embedder, phrase_mse_loss: bool = False, hidden_dim: int = 64):
        super().__init__(embedder)
        self.phrase_mse_loss = phrase_mse_loss

        self.force_to_phrase_model = nn.Sequential(
            nn.Linear(FORCE_EMBEDDING_DIM, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, embedder.embedding_length * embedder.phrase_length)).to(DEVICE)

        self.phrase_to_force_model = nn.Sequential(
            nn.Linear(embedder.embedding_length * embedder.phrase_length, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, FORCE_EMBEDDING_DIM)).to(DEVICE)

    def train(self, force_data: NDArray, phrase_data: NDArray, epochs: int = 1000, verbose: bool = False) -> None:
        X = self.embedder.embed_force(force_data).to(DEVICE)
        Y = self.embedder.embed_phrase(phrase_data).to(DEVICE)

        self.force_to_phrase_model.train()
        self.phrase_to_force_model.train()

        optimizer = optim.Adam(list(self.force_to_phrase_model.parameters()) +
                               list(self.phrase_to_force_model.parameters()))

        for epoch in range(epochs):
            optimizer.zero_grad()
            Y_pred = self.force_to_phrase_model(
                X * (0.95 + 0.1 * torch.rand_like(X)))
            X_pred = self.phrase_to_force_model(Y)
            loss = F.mse_loss(X_pred, X) + (F.mse_loss(Y_pred, Y)
                                            if self.phrase_mse_loss else phrase_cross_entropy(Y_pred, Y, self.embedder.phrase_length))
            loss.backward()
            optimizer.step()

            if verbose:
                print(f'Epoch {epoch}: {loss.item():.4f}')

        self.force_to_phrase_model.eval()
        self.phrase_to_force_model.eval()

    def force_to_phrase(self, force_data: NDArray) -> NDArray:
        X = self.embedder.embed_force(force_data).to(DEVICE)
        Y = self.force_to_phrase_model(X)
        return self.embedder.unembed_phrase(Y)

    def phrase_to_force(self, phrase_data: NDArray) -> NDArray:
        Y = self.embedder.embed_phrase(phrase_data).to(DEVICE)
        X = self.phrase_to_force_model(Y)
        return self.embedder.unembed_force(X)


class DualAutoencoderModel(Model):
    def __init__(self, embedder: Embedder, phrase_mse_loss: bool = False, latent_dim: int = 16):
        super().__init__(embedder)

        self.phrase_mse_loss = phrase_mse_loss

        self.force_encoder = nn.Sequential(
            nn.Linear(FORCE_EMBEDDING_DIM, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, latent_dim)).to(DEVICE)

        self.force_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, FORCE_EMBEDDING_DIM)).to(DEVICE)

        self.phrase_encoder = nn.Sequential(
            nn.Linear(embedder.embedding_length * embedder.phrase_length, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, latent_dim)).to(DEVICE)

        self.phrase_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, embedder.embedding_length * embedder.phrase_length)).to(DEVICE)

    def train(self, force_data: NDArray, phrase_data: NDArray, epochs: int = 1000, verbose: bool = False) -> None:
        X = self.embedder.embed_force(force_data).to(DEVICE)
        Y = self.embedder.embed_phrase(phrase_data).to(DEVICE)

        models = [self.force_encoder, self.force_decoder,
                  self.phrase_encoder, self.phrase_decoder]

        parameters = []
        for model in models:
            parameters += model.parameters()
            model.train()

        optimizer = optim.Adam(parameters)

        for epoch in range(epochs):
            optimizer.zero_grad()

            Z_f = self.force_encoder(X * (0.95 + 0.1 * torch.rand_like(X)))
            Z_p = self.phrase_encoder(Y)
            X_pred_f = self.force_decoder(Z_f)
            X_pred_p = self.force_decoder(Z_p)
            Y_pred_f = self.phrase_decoder(Z_f)
            Y_pred_p = self.phrase_decoder(Z_p)

            loss = F.mse_loss(X_pred_f, X) + \
                F.mse_loss(X_pred_p, X) + info_nce_loss(Z_f, Z_p)
            if self.phrase_mse_loss:
                loss += F.mse_loss(Y_pred_f, Y) + F.mse_loss(Y_pred_p, Y)
            else:
                loss += phrase_cross_entropy(Y_pred_f, Y, self.embedder.phrase_length) + \
                    phrase_cross_entropy(
                        Y_pred_p, Y, self.embedder.phrase_length)
            loss.backward()
            optimizer.step()

            if verbose:
                print(f'Epoch {epoch}: {loss.item():.4f}')

        for model in models:
            model.eval()

    def force_to_phrase(self, force_data: NDArray) -> NDArray:
        X = self.embedder.embed_force(force_data).to(DEVICE)
        Y = self.phrase_decoder(self.force_encoder(X))
        return self.embedder.unembed_phrase(Y)

    def phrase_to_force(self, phrase_data: NDArray) -> NDArray:
        Y = self.embedder.embed_phrase(phrase_data).to(DEVICE)
        X = self.force_decoder(self.phrase_encoder(Y))
        return self.embedder.unembed_force(X)

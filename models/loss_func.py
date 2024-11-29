import torch
import torch.nn as nn


def regression_loss(criterion: torch.nn.modules.MSELoss,
                    ys_hat_1: torch.Tensor, ys_hat_2: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # @see: https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html#torch.nn.CosineSimilarity
    cos = nn.CosineSimilarity()
    prediction = cos(ys_hat_1, ys_hat_2)

    return criterion(prediction, target)


def cosine_embeddings_loss(criterion: torch.nn.modules.loss.CosineEmbeddingLoss,
                           ys_hat_1: torch.Tensor, ys_hat_2: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # @see: https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html#torch.nn.CosineEmbeddingLoss
    return criterion(ys_hat_1, ys_hat_2, target)


def cross_entropy_loss(criterion: torch.nn.modules.loss.CrossEntropyLoss, ys: torch.Tensor, target: torch.Tensor) -> \
        torch.Tensor:
    # @ see: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    return criterion(ys, target)

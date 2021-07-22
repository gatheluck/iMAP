from typing import Final

import torch
import torch.nn as nn


class GaussianPositionalEmbedder(nn.Module):
    """Gaussian positional embedding

    Note:
        For detail, please check original paper,
        "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"
        https://arxiv.org/abs/2006.10739

    Attributes:
        dim_input (int): A dimension of input tensor
        dim_embedding (int): A dimension of embedded tensor
        mean (float, optional): A mean of Gaussian
        std (float, optional): A standard deviation of Gaussian
    """

    def __init__(
        self,
        dim_input: int,
        dim_embedding: int,
        mean: float = 0.0,
        std: float = 25.0,
    ) -> None:
        """
        Args:
            dim_input (int): A dimension of input tensor
            dim_embedding (int): A dimension of embedded tensor
            mean (float, optional): A mean of Gaussian
            std (float, optional): A standard deviation of Gaussian
        """
        super(GaussianPositionalEmbedder, self).__init__()
        assert dim_input > 0, "dim_input should be positive"
        assert dim_embedding > 0, "dim_embedding should be positive"
        assert std > 0.0, "std should be positive"

        self.mean: Final[float] = mean
        self.std: Final[float] = std
        self.embedding_matrix = nn.Linear(dim_input, dim_embedding, bias=False)
        nn.init.normal_(self.embedding_matrix.weight, self.mean, self.std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): An input tensor

        Returns:
            torch.Tensor: An embedded tensor
        """
        return torch.sin(self.embedding_matrix(x))

    def dim_input(self) -> int:
        """
        Note: torch.nn.Module does not support @property
            https://github.com/pytorch/pytorch/pull/16823
        """
        with torch.no_grad():
            return self.embedding_matrix.in_features

    def dim_embedding(self) -> int:
        """
        Note: torch.nn.Module does not support @property
            https://github.com/pytorch/pytorch/pull/16823
        """
        with torch.no_grad():
            return self.embedding_matrix.out_features

from typing import Callable, Final

import torch
import torch.nn as nn
import torch.nn.functional as F


class SceneNetwork(nn.Module):
    """Implicit scene neural network

    Note:
        The architecture is similar to NeRF. However, following points are different
        - viewing directions are not considered
    """

    def __init__(
        self,
        embedder: nn.Module,
        dim_feature: int,
        dim_output_color: int = 3,
        dim_output_density: int = 1,
        activation_function: Callable[[torch.Tensor], torch.Tensor] = F.relu,
    ) -> None:
        """
        Args:
            embedder (nn.Module): An instance of embedder class
            dim_feature (int): A dimension of hidden layer feature
            dim_output_color (int, optional): A dimension for RGB color
            dim_output_density (int, optional): A dimension of output for density
            activation_function (Callable[torch.Tensor, torch.Tensor]): An activation function
        """
        super(SceneNetwork, self).__init__()
        self.embedder = embedder
        # mypy raise 'error: "Tensor" not callable' about following line.
        # It seems to have some relation to following pytorch issue: https://github.com/pytorch/pytorch/issues/24807
        dim_embedding: Final[int] = self.embedder.dim_embedding()  # type: ignore

        self.fc1 = torch.nn.Linear(dim_embedding, dim_feature)
        self.fc2 = torch.nn.Linear(dim_feature, dim_feature)
        self.fc3 = torch.nn.Linear(dim_feature + dim_embedding, dim_feature)
        self.fc4 = torch.nn.Linear(dim_feature, dim_feature)

        self.head_color = torch.nn.Linear(dim_feature, dim_output_color, bias=False)
        self.head_density = torch.nn.Linear(dim_feature, dim_output_density, bias=False)

        self.activation_function = activation_function

    def forward(self, position) -> torch.Tensor:
        """
        Args:
            position (torch.Tensor): An batched 3d positions

        Returns:
            torch.Tensor: The tensor whose first three dimension represents RGB color and
                last dimension represents density
        """
        embedded_position = self.embedder(position)

        h1 = self.activation_function(self.fc1(embedded_position))
        h2 = self.activation_function(self.fc2(h1))
        h2_concatenated = torch.cat([h2, embedded_position], dim=-1)
        h3 = self.activation_function(self.fc3(h2_concatenated))
        h4 = self.activation_function(self.fc4(h3))

        color = self.head_color(h4)
        density = self.head_density(h4)

        return torch.cat([color, density], dim=-1)

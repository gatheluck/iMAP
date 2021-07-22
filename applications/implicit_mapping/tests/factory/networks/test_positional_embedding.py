import torch

from imap.factory.networks.positional_embedding import GaussianPositionalEmbedder


class TestGaussianPositionalEmbedder:
    def test_backward(self):
        batch_size = 32
        dim_input = 3
        dim_embedding = 93
        mean = 0.0
        std = 25.0
        x = torch.randn(batch_size, dim_input).requires_grad_(True)

        embedder = GaussianPositionalEmbedder(
            dim_input,
            dim_embedding,
            mean,
            std,
        )

        x_embedded = embedder(x)
        assert x_embedded.size() == torch.Size([batch_size, dim_embedding])

        pseudo_loss = x_embedded.sum()
        assert x.grad is None
        pseudo_loss.backward()
        assert x.grad.size() == torch.Size([batch_size, dim_input])

    def test_property(self):
        dim_input = 3
        dim_embedding = 93
        embedder = GaussianPositionalEmbedder(
            dim_input,
            dim_embedding,
        )
        assert embedder.dim_input() == dim_input
        assert embedder.dim_embedding() == dim_embedding

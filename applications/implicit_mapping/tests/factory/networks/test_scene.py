import torch

from imap.factory.networks.scene import SceneNetwork


class TestSceneNetwork:
    def test_backward(self, gaussian_positional_embedder_factory):
        batch_size = 32
        dim_input = 3
        dim_embedding = 93
        dim_feature = 256
        dim_output_color = 3
        dim_output_density = 1
        mean = 0.0
        std = 25.0
        x = torch.randn(batch_size, dim_input).requires_grad_(True)

        embedder = gaussian_positional_embedder_factory(
            dim_input=dim_input,
            dim_embedding=dim_embedding,
            mean=mean,
            std=std,
        )

        scene_network = SceneNetwork(
            embedder=embedder,
            dim_feature=dim_feature,
        )

        output = scene_network(x)
        assert output.size() == torch.Size(
            [batch_size, dim_output_color + dim_output_density]
        )

        pseudo_loss = output.sum()
        assert x.grad is None
        pseudo_loss.backward()
        assert x.grad.size() == torch.Size([batch_size, dim_input])

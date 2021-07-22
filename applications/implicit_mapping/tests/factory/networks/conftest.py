import pytest

from imap.factory.networks.positional_embedding import GaussianPositionalEmbedder


@pytest.fixture
def gaussian_positional_embedder_factory():
    def f(
        dim_input=3,
        dim_embedding=93,
        mean=0.0,
        std=25.0,
    ):
        return GaussianPositionalEmbedder(
            dim_input=dim_input,
            dim_embedding=dim_embedding,
            mean=mean,
            std=std,
        )

    return f

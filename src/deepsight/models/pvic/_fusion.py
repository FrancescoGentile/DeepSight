##
##
##

from typing import Annotated

import torch
from torch import Tensor, nn


class MultiModalFusion(nn.Module):
    def __init__(self, embed_dim: int, spatial_dim: int, output_dim: int) -> None:
        super().__init__()

        self.embed_mlp = nn.Sequential(
            nn.Linear(embed_dim, output_dim), nn.LayerNorm(output_dim)
        )

        self.spatial_mlp = nn.Sequential(
            nn.Linear(spatial_dim, output_dim), nn.LayerNorm(output_dim)
        )

        self.out_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2 * output_dim, int(output_dim * 1.5)),
            nn.ReLU(),
            nn.Linear(int(output_dim * 1.5), output_dim),
        )

    def forward(
        self,
        embedddings: Annotated[Tensor, "B N E", float],
        spatial: Annotated[Tensor, "B N S", float],
    ) -> Annotated[Tensor, "B N D", float]:
        embedddings = self.embed_mlp(embedddings)
        spatial = self.spatial_mlp(spatial)
        return self.out_mlp(torch.cat([embedddings, spatial], dim=-1))

    def __call__(
        self,
        embedddings: Annotated[Tensor, "B N E", float],
        spatial: Annotated[Tensor, "B N S", float],
    ) -> Annotated[Tensor, "B N D", float]:
        return super().__call__(embedddings, spatial)

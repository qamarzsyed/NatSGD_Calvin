from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from concat_encoders import ConcatEncoders
from vision_network import VisionNetwork
class VisualGoalEncoder(nn.Module):
    def __init__(
        self,
        hidden_size = 2048,
        latent_goal_features = 32,
        in_features = 64,
        l2_normalize_goal_embeddings = False,
        activation_function = "ReLU",
    ):

        super().__init__()
        self.vision_static_encoder = VisionNetwork()
        self.l2_normalize_output = l2_normalize_goal_embeddings
        self.act_fn = getattr(nn, activation_function)()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_size),
            self.act_fn,
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            self.act_fn,
            nn.Linear(in_features=hidden_size, out_features=latent_goal_features),
        )

    def forward(self, start: torch.Tensor) -> torch.Tensor:
        s, h, w, c = start.shape
        imgs_static = start.reshape(-1, c, h, w)  # (batch_size * sequence_length, 3, 200, 200)
        # ------------ Vision Network ------------ #
        encoded_imgs = self.vision_static_encoder(imgs_static)  # (batch*seq_len, 64)
        encoded_imgs = encoded_imgs.reshape(1, s, -1)
        x = self.mlp(encoded_imgs)
        if self.l2_normalize_output:
            x = F.normalize(x, p=2, dim=1)
        x = x.reshape(1, 1, 1, 32)
        return x


class LanguageGoalEncoder(nn.Module):
    def __init__(
        self,
        language_features = 384,
        hidden_size=2048,
        latent_goal_features=32,
        word_dropout_p=0.0,
        l2_normalize_goal_embeddings=False,
        activation_function= "ReLU",
    ):
        super().__init__()
        self.l2_normalize_output = l2_normalize_goal_embeddings
        self.act_fn = getattr(nn, activation_function)()
        self.mlp = nn.Sequential(
            nn.Dropout(word_dropout_p),
            nn.Linear(in_features=language_features, out_features=hidden_size),
            self.act_fn,
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            self.act_fn,
            nn.Linear(in_features=hidden_size, out_features=latent_goal_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        if self.l2_normalize_output:
            x = F.normalize(x, p=2, dim=1)
        return x

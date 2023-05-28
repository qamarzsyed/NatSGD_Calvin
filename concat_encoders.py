from typing import Dict, Optional

import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import cv2
from vision_network import VisionNetwork  

class ConcatEncoders(nn.Module):
    def __init__(
        self,
        # vision_static: DictConfig,
        # proprio: DictConfig,
        # vision_gripper: Optional[DictConfig] = None,
        # depth_static: Optional[DictConfig] = None,
        # depth_gripper: Optional[DictConfig] = None,
        # tactile: Optional[DictConfig] = None,
    ):
        super().__init__()
        self._latent_size = 4*64
        # self._latent_size = vision_static.visual_features
        # if vision_gripper:
        #     self._latent_size += vision_gripper.visual_features
        # if depth_gripper and vision_gripper.num_c < 4:
        #     vision_gripper.num_c += depth_gripper.num_c
        # if depth_static and vision_static.num_c < 4:
        #     vision_static.num_c += depth_static.num_c
        # if tactile:
        #     self._latent_size += tactile.visual_features

        self.vision_static_encoder = VisionNetwork()
        # self.vision_gripper_encoder = hydra.utils.instantiate(vision_gripper) if vision_gripper else None
        # self.proprio_encoder = hydra.utils.instantiate(proprio)
        # self.tactile_encoder = hydra.utils.instantiate(tactile)
        # self._latent_size += self.proprio_encoder.out_features

    @property
    def latent_size(self):
        return self._latent_size

    def forward(
        self, start: torch.Tensor, end: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
        # imgs_static = imgs["start_obs"]
        # imgs_static_2 = imgs["end_obs"]
        # imgs_static_3 = imgs['bbox']
        b = 1
        # b, s, c, h, w = start.shape
        s, w, h, c = start.shape
        imgs_static = start.reshape(-1, c, h, w)  # (batch_size * sequence_length, 3, 200, 200)
        # ------------ Vision Network ------------ #
        encoded_imgs = self.vision_static_encoder(imgs_static)  # (batch*seq_len, 64)
        encoded_imgs = encoded_imgs.reshape(b, s, -1)  # (batch, seq, 64)

        # b, s, c, h, w = end.shape
        s, w, h, c = end.shape
        imgs_static_2 = end.reshape(-1, c, h, w)
        encoded_imgs_2 = self.vision_static_encoder(imgs_static_2)  # (batch*seq_len, 64)
        encoded_imgs_2 = encoded_imgs_2.reshape(b, s, -1)  # (batch, seq, 64)
        
        # b, s, c, h, w = bbox.shape
        s, c, h, w = bbox.shape
        imgs_static_3 = bbox.reshape(-1, c, h, w)
        encoded_imgs_3 = self.vision_static_encoder(imgs_static_3)  # (batch*seq_len, 64)
        encoded_imgs_3 = encoded_imgs_3.reshape(b, s, -1)  # (batch, seq, 64)

        blank_4 = torch.zeros((1, 1, 64))

        #state_obs_out = self.proprio_encoder(state_obs)
        perceptual_emb = torch.cat([encoded_imgs, encoded_imgs_2, encoded_imgs_3, blank_4], dim=-1)
        return perceptual_emb

# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from models.networks_stylegan2 import FullyConnectedLayer
import torch.nn.functional as F
import torch.nn as nn
import torch
import copy
import math

class IDUpsampler(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(IDUpsampler, self).__init__()
        self.up_xy = nn.Sequential(nn.ConvTranspose2d(in_dim, 128, 4, 2, 1),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv2d(128, out_dim, 3, 1, 1))
        self.up_xz = nn.Sequential(nn.ConvTranspose2d(in_dim, 128, 4, 2, 1),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv2d(128, out_dim, 3, 1, 1))
        self.up_zy = nn.Sequential(nn.ConvTranspose2d(in_dim, 128, 4, 2, 1),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv2d(128, out_dim, 3, 1, 1))
        self.skip_xy = nn.Conv2d(in_dim, out_dim, 1, 1, 0)
        self.skip_xz = nn.Conv2d(in_dim, out_dim, 1, 1, 0)
        self.skip_zy = nn.Conv2d(in_dim, out_dim, 1, 1, 0)

    def forward(self, id_planes):
        xy_ = F.interpolate(id_planes[:, 0], size = (256, 256), mode = 'bilinear', align_corners = False)
        xy = self.skip_xy(xy_) + self.up_xy(id_planes[:, 0])
        xz_ = F.interpolate(id_planes[:, 1], size = (256, 256), mode = 'bilinear', align_corners = False)
        xz = self.skip_xz(xz_) + self.up_xy(id_planes[:, 1])
        zy_ = F.interpolate(id_planes[:, 2], size = (256, 256), mode = 'bilinear', align_corners = False)
        zy = self.skip_zy(zy_) + self.up_xy(id_planes[:, 2])
        return torch.stack((xy, xz, zy), dim = 1)

class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )

    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}

class Embed2Plane(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Embed2Plane, self).__init__()
        self.up_layers = nn.Sequential(nn.Conv2d(in_dim, 256, 3, 1, 1), # 128x128
                                       nn.LeakyReLU(0.2),
                                       nn.Conv2d(256, 128, 3, 1, 1), # 128x128
                                       nn.LeakyReLU(0.2),
                                       nn.ConvTranspose2d(128, 128, 4, 2, 1), # 256x256
                                       nn.LeakyReLU(0.2),
                                       nn.Conv2d(128, 128, 3, 1, 1), # 256x256
                                       nn.LeakyReLU(0.2),
                                       nn.Conv2d(128, out_dim, 3, 1, 1))

    def forward(self, x):
        x = self.up_layers(x)
        return x.view(x.shape[0], 3, 32, 256, 256)

# Copyright (c) 2022 Sicheng Xu
# MIT License
# To view a copy of this license, visit
# https://github.com/sicxu/Deep3DFaceRecon_pytorch/blob/master/LICENSE
# ------------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def set_rasterizer(type = 'pytorch3d'):
    if type == 'pytorch3d':
        global Meshes, load_obj, rasterize_meshes
        from pytorch3d.structures import Meshes
        from pytorch3d.io import load_obj
        from pytorch3d.renderer.mesh import rasterize_meshes
    else:
        raise "Only support Pytorch3D!!!"

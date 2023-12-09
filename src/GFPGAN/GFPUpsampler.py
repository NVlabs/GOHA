# ------------------------------------------------------------------------------
# Copyright (C) 2021 THL A29 Limited, a Tencent company.
# Apache License
# To view a copy of this license, visit
# https://github.com/TencentARC/GFPGAN/blob/master/LICENSE
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import sys
sys.path.append('./')
from GFPGAN.gfpganv1_clean_arch import GFPGANv1Clean, GFPGANv2Clean
from torchvision.transforms.functional import normalize
from basicsr.utils import img2tensor, tensor2img

class GFPUpsampler(nn.Module):
    def __init__(self, model_path, upscale=2, channel_multiplier=2):
        super().__init__()
        self.gfpgan = GFPGANv1Clean(
            out_size=512,
            num_style_feat=512,
            channel_multiplier=channel_multiplier,
            decoder_load_path=None,
            fix_decoder=False,
            num_mlp=8,
            input_is_latent=True,
            different_w=True,
            narrow=1,
            sft_half=True)

        loadnet = torch.load(model_path)
        self.gfpgan.load_state_dict(loadnet['params_ema'], strict=True)

        self.upscale = upscale
        return

    def forward(self, img, noise=None):
        """
        Input:
          - img: (b, 3, h, w), range [-1.0, 1.0]
        Output:
          - image: restored image of size (b, 3, 512, 512)
          - out_rgbs: seven intermediate rgb images of sizes (8, 8), (16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512)
        """
        img = F.interpolate(img, size = (512, 512), mode = 'bilinear', align_corners = True)
        image, out_rgbs = self.gfpgan(img, noise=noise)
        return image, out_rgbs

class GFPUpsamplerv2(nn.Module):
    def __init__(self, model_path, upscale=2, channel_multiplier=2):
        super().__init__()
        self.gfpgan = GFPGANv2Clean(
            out_size=512,
            num_style_feat=512,
            channel_multiplier=channel_multiplier,
            decoder_load_path=None,
            fix_decoder=False,
            num_mlp=8,
            input_is_latent=True,
            different_w=True,
            narrow=1,
            sft_half=True)

        loadnet = torch.load(model_path)
        self.gfpgan.load_state_dict(loadnet['params_ema'], strict=True)

        self.upscale = upscale
        return

    def forward(self, img, noise=None):
        """
        Input:
          - img: (b, 3, h, w), range [-1.0, 1.0]
        Output:
          - image: restored image of size (b, 3, 512, 512)
          - out_rgbs: seven intermediate rgb images of sizes (8, 8), (16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512)
        """
        img = F.interpolate(img, size = (512, 512), mode = 'bilinear', align_corners = True)
        image, out_rgbs = self.gfpgan(img, noise=noise)
        return image, out_rgbs

if __name__ == '__main__':
    upsampler = GFPUpsampler(model_path='/home/xli/Documents/FaceStylization/GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth')
    upsampler.cuda()
    img = cv2.imread("/home/xli/Documents/FaceStylization/GFPGAN/inputs/whole_imgs/0.png")
    img = img2tensor(img / 255., bgr2rgb=True, float32=True)
    normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    img = img.unsqueeze(0).cuda()
    img, out_rgbs = upsampler(img)
    import torchvision.utils as vutils
    vutils.save_image((img + 1) / 2.0, 'img.png')
    for i, rgb in enumerate(out_rgbs):
        print(rgb.min(), rgb.max(), rgb.shape)
        vutils.save_image((rgb + 1) / 2.0, '{}.png'.format(i))

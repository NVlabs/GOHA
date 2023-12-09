# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from kornia import morphology as morph
from transformers import SegformerForSemanticSegmentation

from models.BiSeNet import BiSeNet
from models.volumetric_rendering.renderer import ImportanceRenderer
from models.volumetric_rendering.ray_sampler import RaySampler
from models.networks import OSGDecoder, Embed2Plane, IDUpsampler

import numpy as np
from PIL import Image
from losses.threeDMM import ExpLoss
from GFPGAN.GFPUpsampler import GFPUpsampler

class Model(torch.nn.Module):
    def __init__(self, cfg, test = False):
        super(Model, self).__init__()
        self.cfg = cfg

        # canonical branch
        self.template_net = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.template_net.decode_head.batch_norm = nn.Identity()
        self.template_net.decode_head.activation = nn.Identity()
        self.template_net.decode_head.dropout = nn.Identity()
        self.template_net.decode_head.classifier = nn.Identity()
        self.embedding2neutral = Embed2Plane(256, 96)
        self.freeze_cano = False
        if cfg.experiment.freeze_cano == 1:
            self.freeze_cano = True
            for param in self.template_net.parameters():
                param.requires_grad = False
            for param in self.embedding2neutral.parameters():
                param.requires_grad = False

        # apperance branch
        self.id_net = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.id_net.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(6, 32, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
        self.id_net.decode_head.batch_norm = nn.Identity()
        self.id_net.decode_head.activation = nn.Identity()
        self.id_net.decode_head.dropout = nn.Identity()
        self.id_net.decode_head.classifier = nn.Identity()
        self.id_upsampler = IDUpsampler(256, 32)

        # expression branch
        self.expr_net = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.expr_net.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(3, 32, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
        self.expr_net.decode_head.batch_norm = nn.Identity()
        self.expr_net.decode_head.activation = nn.Identity()
        self.expr_net.decode_head.dropout = nn.Identity()
        self.expr_net.decode_head.classifier = nn.Identity()
        self.embedding2expr = Embed2Plane(256, 96)
        self.expr_upsampler = IDUpsampler(96, 32)
        cano_pose = torch.tensor([[ 0.9997,  0.0059,  0.0250, -0.0731,  0.0081, -0.9961, -0.0882,  0.2462,
          0.0244,  0.0884, -0.9958,  2.6878,  0.0000,  0.0000,  0.0000,  1.0000,
          4.2647,  0.0000,  0.5000,  0.0000,  4.2647,  0.5000,  0.0000,  0.0000,
          1.0000]])
        self.register_buffer("cano_pose", cano_pose) # 1, 25

        img = np.array(Image.open('dataset/cano_exemplar.png')).transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        self.register_buffer("cano_exemplar", (img / 255.0) * 2 - 1)

        # decoder
        self.decoder = OSGDecoder(32, {'decoder_lr_mul': 1, 'decoder_output_dim': 32})

        # renderer
        self.ray_sampler = RaySampler()
        self.renderer = ImportanceRenderer()
        self.render_resolution = cfg.models.render_resolution
        self.rendering_kwargs = {'image_resolution': 512, 'disparity_space_sampling': False, 'clamp_mode': 'softplus', 'gpc_reg_prob': 0.5, 'c_scale': 1.0, 'density_reg': 0.25, 'density_reg_p_dist': 0.004, 'reg_type': 'l1', 'decoder_lr_mul': 1.0, 'sr_antialias': True, 'depth_resolution': 48, 'depth_resolution_importance': 48, 'ray_start': 2.25, 'ray_end': 3.3, 'box_warp': 1, 'avg_camera_radius': 2.7, 'avg_camera_pivot': [0, 0, 0.2]}

        # face parsing helper
        net = BiSeNet(n_classes=19)
        net.load_state_dict(torch.load(cfg.experiment.parsing_path, map_location=torch.device('cpu')))
        for param in net.parameters():
            param.requires_grad = False
        self.face_parsing_net = net
        self.face_parsing_net.eval()
        self.face_parsing_normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        self.register_buffer('image_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('image_std', torch.tensor([[0.229, 0.224, 0.225]]).view(1, 3, 1, 1))

        u = torch.linspace(-1, 1, steps = 128)
        v = torch.linspace(-1, 1, steps = 128)
        u, v = torch.meshgrid(u, v, indexing='xy')
        uv = torch.stack((u, v), dim = -1).view(-1, 2)
        self.register_buffer('uv', uv.unsqueeze(0))

        self.cano_render_resolution = 128
        self.exp_loss_fn = ExpLoss(cfg.device, test)

        self.neural_layers = GFPUpsampler(cfg.experiment.GFP_path)
        if hasattr(cfg.experiment, 'freeze_gfp') and cfg.experiment.freeze_gfp == 1:
            for param in self.neural_layers.parameters():
                param.requires_grad = False

    def depth2pcd(self, cam2world, intrinsics, depth_image):
        """
        Construct a point cloud given a depth image.
         - Input:
           - cam2world: camera view (b, 4, 4)
           - intrinsics: camera intrinsics (b, 3, 3)
           - depth_image: depth image rendered from the canonical branch (b, c, 128, 128)
         - Output:
           - pcd: point cloud (b, 128*128, 3)
        """
        batch_size = depth_image.shape[0]
        resolution = depth_image.shape[-1]
        ray_origins, ray_dirs = self.ray_sampler(cam2world, intrinsics, resolution)
        depths_ = depth_image.view(batch_size, -1, 1, 1)
        pcd = (ray_origins.unsqueeze(-2) + depths_ * ray_dirs.unsqueeze(-2)).reshape(batch_size, -1, 3)
        return pcd

    def construct_planes(self, pcd, img_feat):
        """
        Rasterize point cloud to tri-plane features.
         - Input
           - pcd: point cloud (b, 128*128, 3).
           - img_feat: image feature extracted from E_p (b, 256, 128, 128).
         - Output:
           - A feature map (b, 3, 256, 128, 128), which will be upsampled and decoded to the apperance tri-plane.
        """
        b, c, _, _ = img_feat.shape
        img_feat_flatten = img_feat.permute(0, 2, 3, 1).view(b, -1, c)

        xy = pcd[:, :, :2].squeeze()
        xy = xy * 2
        dist = torch.cdist(self.uv.repeat(b, 1, 1), xy)
        nearest_idx = torch.argmin(dist, dim = -1)
        xy_planes = []
        for i in range(b):
            xy_planes.append(img_feat_flatten[i:i+1, nearest_idx[i], :])
        xy_plane = torch.cat(xy_planes).view(b, 128, 128, -1).permute(0, 3, 1, 2)

        xz = torch.cat((pcd[:, :, 0:1], pcd[:, :, 2:]), dim = -1).squeeze()
        xz = xz * 2
        dist = torch.cdist(self.uv.repeat(b, 1, 1), xz)
        nearest_idx = torch.argmin(dist, dim = -1)
        xz_planes = []
        for i in range(b):
            xz_planes.append(img_feat_flatten[i:i+1, nearest_idx[i], :])
        xz_plane = torch.cat(xz_planes).view(b, 128, 128, -1).permute(0, 3, 1, 2)

        zy = torch.cat((pcd[:, :, 2:], pcd[:, :, 1:2]), dim = -1).squeeze()
        zy = zy * 2
        dist = torch.cdist(self.uv.repeat(b, 1, 1), zy)
        nearest_idx = torch.argmin(dist, dim = -1)
        zy_planes = []
        for i in range(b):
            zy_planes.append(img_feat_flatten[i:i+1, nearest_idx[i], :])
        zy_plane = torch.cat(zy_planes).view(b, 128, 128, -1).permute(0, 3, 1, 2)
        return torch.stack((xy_plane, xz_plane, zy_plane), dim = 1)

    def get_parsing_mask(self, img_src):
        src = (img_src.clone() + 1) / 2.0
        src = self.face_parsing_normalize(src)
        out = self.face_parsing_net(src)[0]
        # remove eye glasses category
        out = torch.cat((out[:, :6], out[:, 7:]), dim = 1)
        out = torch.argmax(out[:, :-1], dim = 1).unsqueeze(1)
        mask = torch.zeros_like(out)
        kernel = torch.ones((7, 7)).to(out.device)
        for i in [2, 3, 4, 5, 10, 11, 12]:
            if i == 4 or i == 5:
                # dilate eyes
                dilated_mask = morph.dilation((out == i) * 1, kernel)
                dilated_mask = morph.dilation(dilated_mask, kernel)
                mask += (dilated_mask > 0).long()
            else:
                mask += (out == i) * 1
        mouth_mask = (out == 10) * 1
        return (mask > 0) * 1

    def interpolate(self, feature, size, mode = 'bilinear', align_corners = False):
        return F.interpolate(feature, size = size, mode = mode, align_corners = align_corners)

    def get_expr_plane(self, id_img, tar_expr, exp_transfer = True):
        """
        Predict the expression tri-plane given 3DMM rendering.
         - Input:
           - id_img: source image (b, 3, 512, 512)
           - tar_expr: drive image (b, 3, 512, 512)
           - expr_transfer: whether source and target image has different expressions?
         - Output:
           - expr_planes: the expression tri-plane (b, 3, 32, 256, 256)
           - visualize: visualization of the 3DMM rendering (b, 3, 512, 512) in range [-1, 1]
        """
        sudo = self.exp_loss_fn.get_sudo(id_img, self.cano_exemplar, tar_expr, exp_transfer)
        sudo = F.interpolate(sudo, size = id_img.shape[-2:], mode = 'bilinear', align_corners = False)
        visualize = sudo.clone()

        sudo = ((sudo + 1) / 2.0 - self.image_mean.detach()) / self.image_std.detach()
        inputs = {}
        inputs['pixel_values'] = sudo
        inputs['output_hidden_states'] = True
        expr_feat = self.expr_net(**inputs).logits
        expr_planes = self.embedding2expr(expr_feat)
        return expr_planes, visualize

    def get_canonical_plane(self, img_src, mask):
        """
        Predict the canonical tri-plane from source image.
         - Inputs:
           - img_src: (b, 3, 512, 512)
           - mask: (b, 1, 512, 512), mask out eyes and mouth
         - Output:
           - canonical triplane: (b, 3, 32, 256, 256)
        """
        neutral_sudo, neutral_mask = self.exp_loss_fn.get_neutral_render(img_src)
        neutral_sudo = neutral_sudo * 2 - 1

        im = ((img_src + 1) / 2.0 - self.image_mean.detach()) / self.image_std.detach()
        inputs = {}
        inputs['pixel_values'] = im * mask
        inputs['output_hidden_states'] = True
        template_feat = self.template_net(**inputs).logits
        canonical_plane = self.embedding2neutral(template_feat).contiguous()
        if self.freeze_cano:
            canonical_plane = canonical_plane.detach()
        return canonical_plane

    def get_apperance_plane(self, img, cam2world, intrinsics, base_plane, mask):
        """
        Predict the apperance tri-plane from the source image
         - Inputs:
           - img: (b, 3, 512, 512)
           - cam2world: (b, 4, 4)
           - intrinsics: (b, 3, 3)
           - base_plane: (b, 3, 32, 256, 256), [canonical + expression] triplane
           - mask: (b, 1, 512, 512), mask out eyes and mouth
         - Output:
           - app_planes: (b, 3, 32, 256, 256)
        """
        ray_origins_, ray_directions_ = self.ray_sampler(cam2world, intrinsics, self.cano_render_resolution)
        cano_feature, cano_depth, _ = self.renderer(base_plane, self.decoder, ray_origins_, ray_directions_, self.rendering_kwargs) # channels last
        cano_feature = cano_feature.view(-1, self.cano_render_resolution, self.cano_render_resolution, 32).permute(0, 3, 1, 2)
        cano_depth = cano_depth.permute(0, 2, 1).reshape(-1, 1, self.cano_render_resolution, self.cano_render_resolution)
        cano_depth = F.interpolate(cano_depth, size = (128, 128), mode = 'bilinear', align_corners = False)
        pcd = self.depth2pcd(cam2world, intrinsics, cano_depth)

        cano_render = F.interpolate(cano_feature[:, :3], size = img.shape[-2:], mode = 'bilinear', align_corners = False)

        cano_mask = self.get_parsing_mask(cano_render)
        union_mask = (1 - cano_mask.float()) * mask
        kernel = torch.ones((7, 7)).to(union_mask.device)
        dilated_mask = morph.erosion(union_mask, kernel)

        cano_render = ((cano_render + 1) / 2.0 - self.image_mean.detach()) / self.image_std.detach()
        inputs = {}
        im = ((img + 1) / 2.0 - self.image_mean.detach()) / self.image_std.detach()
        inputs['pixel_values'] = torch.cat((im * dilated_mask + cano_render * (1 - dilated_mask), cano_render), dim = 1)
        inputs['output_hidden_states'] = True
        id_feat = self.id_net(**inputs).logits

        app_planes = self.construct_planes(pcd, id_feat)
        app_planes = self.id_upsampler(app_planes)
        return app_planes

    def render_plane(self, plane, ray_origins, ray_directions):
        """
        Render a tri-plane at given camera view.
        - Inputs:
          - plane: (b, 3, 32, 256, 256)
          - ray_origins: (b, h*w, 3)
          - ray_directions: (b, h*w, 3)
        - Outputs:
          - rgb_low_res: (b, 3, self.render_resolution ,self.render_resolution)
          - depth_image: (b, 1, self.render_resolution ,self.render_resolution)
        """
        feature_samples, depth_samples, _ = self.renderer(plane, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last
        depth_image = depth_samples.permute(0, 2, 1).reshape(-1, 1, self.render_resolution, self.render_resolution)
        input_feat = feature_samples.view(-1, self.render_resolution, self.render_resolution, 32).permute(0, 3, 1, 2)
        rgb_low_res = input_feat[:, :3]
        return rgb_low_res, depth_image

    def forward(self, data, mode = 'full_rec_tar', iteration = 0, exp_transfer = True):
        outputs = {}
        img_tar = data['img_target']
        pose_tar = data['pose_target']
        img_src = data['img_src']
        pose_src = data['pose_src']
        batch_size = img_tar.shape[0]

        # facial mask
        mask_  = self.get_parsing_mask(torch.cat((img_src, img_tar)))
        src_mask = (1 - mask_[:batch_size])
        tar_mask = (1 - mask_[batch_size:])

        src_cam2world = pose_src[:, :16].view(-1, 4, 4)
        src_intrinsics = pose_src[:, 16:25].view(-1, 3, 3)
        tar_cam2world = pose_tar[:, :16].view(-1, 4, 4)
        tar_intrinsics = pose_tar[:, 16:25].view(-1, 3, 3)
        src_ray_origins, src_ray_directions = self.ray_sampler(src_cam2world, src_intrinsics, self.render_resolution)
        tar_ray_origins, tar_ray_directions = self.ray_sampler(tar_cam2world, tar_intrinsics, self.render_resolution)

        # canonical branch
        canonical_plane = self.get_canonical_plane(img_src, src_mask)

        # expression branch
        source_plane = canonical_plane
        target_plane = canonical_plane
        if iteration >= self.cfg.experiment.add_expr_plane:
            expr_plane_src, sudo_src = self.get_expr_plane(img_src, img_src, exp_transfer = False)
            expr_plane_tar, sudo_tar = self.get_expr_plane(img_src, img_tar, exp_transfer = exp_transfer)
            source_plane = source_plane + expr_plane_src
            target_plane = target_plane + expr_plane_tar
            if mode == 'full_rec_tar' or mode == 'rec':
                outputs['sudo_src'] = sudo_src
                outputs['sudo_tar'] = sudo_tar

        if mode == 'coarse_target':
            """
            Coarse reconstruction of canonical + expression tri-plane.
            """
            plane = target_plane
            ray_origins = tar_ray_origins
            ray_directions = tar_ray_directions

            rgb_low_res, depth_image = self.render_plane(plane, ray_origins, ray_directions)
            gt_low_res = self.interpolate(img_tar, rgb_low_res.shape[-2:])

            outputs['pred'] = rgb_low_res
            outputs['gt'] = gt_low_res
            outputs['target_plane'] = plane
            outputs['input'] = self.interpolate(img_src, rgb_low_res.shape[-2:])
            return outputs

        if mode == 'full_rec_src':
            """
            Full reconstruction rendered at source camera view.
            """
            plane = source_plane

            # apperance branch
            if iteration >= self.cfg.experiment.add_id_plane:
                app_planes = self.get_apperance_plane(img_src, src_cam2world, src_intrinsics, plane, src_mask)
                plane = plane + app_planes
                outputs['id_planes'] = app_planes

            rgb_low_res, depth_image = self.render_plane(plane, src_ray_origins, src_ray_directions)

            # super resolution
            neural_rendering = self.neural_layers(rgb_low_res)[0]
            resized_low_res = self.interpolate(rgb_low_res, size = neural_rendering.shape[-2:])
            gt_low_res = self.interpolate(img_src, size = rgb_low_res.shape[-2:])
            resized_gt = self.interpolate(gt_low_res, size = neural_rendering.shape[-2])

            outputs['rec'] = torch.cat((neural_rendering, resized_low_res))
            outputs['gt'] = torch.cat((img_src, resized_gt))

            outputs['mask'] = self.interpolate(src_mask.float(), rgb_low_res.shape[-2:])
            outputs['depth_image'] = depth_image
            outputs['target_plane'] = plane
            return outputs

        if mode == 'full_rec_tar':
            """
            Full reconstruction rendered at target view
            """
            plane = target_plane

            # apperance branch
            if iteration >= self.cfg.experiment.add_id_plane:
                app_planes = self.get_apperance_plane(img_src, src_cam2world, src_intrinsics, plane, src_mask)
                plane = plane + app_planes
                outputs['id_planes'] = app_planes

            rgb_low_res, depth_image = self.render_plane(plane, tar_ray_origins, tar_ray_directions)

            # super resolution
            neural_rendering = self.neural_layers(rgb_low_res)[0]
            resized_low_res = self.interpolate(rgb_low_res, size = neural_rendering.shape[-2:])
            gt_low_res = self.interpolate(img_tar, size = rgb_low_res.shape[-2:])
            resized_gt = self.interpolate(gt_low_res, size = neural_rendering.shape[-2])

            # we compute reconstruction loss for both the rendering and super resolution image
            outputs['rec'] = torch.cat((neural_rendering, resized_low_res))
            outputs['gt'] = torch.cat((img_tar, resized_gt))

            outputs['rgb_low_res'] = rgb_low_res
            outputs['mask'] = self.interpolate(tar_mask.float(), rgb_low_res.shape[-2:])
            outputs['input'] = self.interpolate(img_src, (self.render_resolution, self.render_resolution))
            outputs['depth_image'] = depth_image
            outputs['target_plane'] = plane

            outputs['high_res'] = neural_rendering
            return outputs

    def sample_mixed(self, coordinates, target_plane):
        """
        Used in tri-plane TV loss.
        """
        return self.renderer.run_model(target_plane, self.decoder, coordinates, torch.zeros_like(coordinates), self.rendering_kwargs)

    def sample(self, coordinates, target_plane):
        """
        Used in marching cube.
        """
        return self.renderer.run_model(target_plane, self.decoder, coordinates, torch.zeros_like(coordinates), self.rendering_kwargs)

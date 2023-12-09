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
from torch import nn
import torch.nn.functional as F
from deep3d_modules import create_model
from deep3d_modules.render import MeshRenderer
import numpy as np
from kornia import morphology as morph
from imageio import imsave

class ExpLoss(nn.Module):
    def __init__(self, device, test = False):
        super(ExpLoss, self).__init__()
        print("Loading Deep3D encoder")
        self.deep3D = create_model()
        self.deep3D.setup()
        self.deep3D.device = device
        if not test:
            self.deep3D.parallelize()
        self.deep3D.eval()
        self.deep3D.facemodel.to(device)
        self.deep3D.net_recon.to(device)
        #self.deep3D.facemodel.render.to(device)

        focal = 1015.0
        center = 112.0
        z_near = 5.0
        z_far = 15.0
        fov = 2 * np.arctan(center / focal) * 180 / np.pi
        self.renderer = MeshRenderer(rasterize_fov=fov, znear=z_near, zfar=z_far, rasterize_size=int(2 * center))

        self.bfm_UVs = np.load('/xtli-correspondence/OMG/HRN/assets/3dmm_assets/template_mesh/bfm_uvs2.npy')
        self.bfm_UVs = torch.from_numpy(self.bfm_UVs).to(device).float()

    def preprocess(self, img):
        return F.interpolate((img + 1) / 2.0, size = (224, 224), mode = 'bilinear', align_corners = False)

    def expr_loss(self, img, exp_tar):
        batch_size = img.shape[0]
        im_tensor = self.preprocess(img)
        output_coeff = self.deep3D.net_recon(im_tensor)
        pred_expr = output_coeff[:, 80: 144]
        loss = torch.nn.functional.l1_loss(pred_expr, exp_tar)
        return loss

    def get_neutral_render(self, img):
        im_tensor = self.preprocess(img)
        with torch.no_grad():
            output_coeff = self.deep3D.net_recon(im_tensor)

        output_coeff[:, 80: 144] = torch.zeros_like(output_coeff[:, 80: 144])
        pred_vertex, pred_texture, pred_color, landmark = self.deep3D.facemodel.compute_for_render(output_coeff)

        # sudo supervision
        pred_mask, _, pred_face = self.renderer(pred_vertex, self.deep3D.facemodel.face_buf, feat=pred_color)
        pred_mask = F.interpolate(pred_mask, size = img.shape[-2:], mode = 'bilinear', align_corners = False)
        sudo = F.interpolate(pred_face, size = img.shape[-2:], mode = 'bilinear', align_corners = False)
        return sudo, pred_mask

    def get_neutral_render_random(self, img, pose_img):
        img = self.preprocess(img)
        pose_img = self.preprocess(pose_img)
        im_tensor = torch.cat((img, pose_img))
        with torch.no_grad():
            output_coeff = self.deep3D.net_recon(im_tensor)

        output_coeff[0, 80: 144] = torch.zeros_like(output_coeff[0, 80: 144])
        output_coeff[0, 224: 227] = output_coeff[1, 224: 227]
        output_coeff[0, 254:] = output_coeff[1, 254:]
        pred_vertex, pred_texture, pred_color, landmark = self.deep3D.facemodel.compute_for_render(output_coeff[:1])

        # sudo supervision
        pred_mask, _, pred_face = self.renderer(pred_vertex, self.deep3D.facemodel.face_buf, feat=pred_color)
        pred_mask = F.interpolate(pred_mask, size = img.shape[-2:], mode = 'bilinear', align_corners = False)
        sudo = F.interpolate(pred_face, size = img.shape[-2:], mode = 'bilinear', align_corners = False)
        return sudo, pred_mask

    def neutral_loss(self, img_tar, cano_render):
        batch_size = img_tar.shape[0]
        img_tar_resize = self.preprocess(img_tar)
        cano_render_resize = self.preprocess(cano_render)
        im_tensor = torch.cat((img_tar_resize.detach(), cano_render_resize))
        output_coeff = self.deep3D.net_recon(im_tensor)

        tar_coeff = output_coeff[:batch_size].detach()
        cano_coeff = output_coeff[batch_size:]
        tar_coeff = tar_coeff.clone()
        tar_coeff[:, 80: 144] = torch.zeros_like(tar_coeff[:, 80: 144])
        pred_vertex, pred_texture, pred_color, landmark = self.deep3D.facemodel.compute_for_render(torch.cat((tar_coeff, cano_coeff)))

        weight = np.ones([68])
        weight[17:27] = 200
        weight[36:37] = 50
        weight[-10:] = 200
        weight = np.expand_dims(weight, 0)
        weight = torch.tensor(weight).to(landmark.device)
        recon_lm = landmark[:batch_size]
        face_lm = landmark[batch_size:]
        lm_loss = torch.sum((recon_lm - face_lm)**2, dim=-1) * weight
        lm_loss = torch.sum(lm_loss) / (face_lm.shape[0] * face_lm.shape[1] * 224.0)

        # sudo supervision
        pred_mask, _, pred_face = self.renderer(pred_vertex[:batch_size], self.deep3D.facemodel.face_buf, feat=pred_color[:batch_size])
        pred_mask = F.interpolate(pred_mask, size = cano_render.shape[-2:], mode = 'bilinear', align_corners = False)
        sudo = F.interpolate(pred_face, size = cano_render.shape[-2:], mode = 'bilinear', align_corners = False)
        pred = (cano_render + 1) / 2.0 * pred_mask

        #import torchvision.utils as vutils
        #vutils.save_image(pred, 'pred.png')
        #vutils.save_image(sudo, 'sudo.png')
        #import pdb; pdb.set_trace()
        sudo_l1 = torch.nn.functional.l1_loss(pred, sudo)
        return lm_loss, sudo_l1, sudo

    def neutral_loss_with_mask(self, img_tar, cano_render, tar_mask):
        tar_mask = 1 - tar_mask
        batch_size = img_tar.shape[0]
        img_tar_resize = self.preprocess(img_tar)
        cano_render_resize = self.preprocess(cano_render)
        im_tensor = torch.cat((img_tar_resize.detach(), cano_render_resize))
        output_coeff = self.deep3D.net_recon(im_tensor)

        tar_coeff = output_coeff[:batch_size].detach()
        cano_coeff = output_coeff[batch_size:]
        tar_coeff = tar_coeff.clone()
        tar_coeff[:, 80: 144] = torch.zeros_like(tar_coeff[:, 80: 144])
        pred_vertex, pred_texture, pred_color, landmark = self.deep3D.facemodel.compute_for_render(torch.cat((tar_coeff, cano_coeff)))

        #weight = np.ones([68])
        #weight[17:27] = 200
        #weight[36:37] = 50
        #weight[-10:] = 200
        #weight = np.expand_dims(weight, 0)
        #weight = torch.tensor(weight).to(landmark.device)
        #recon_lm = landmark[:batch_size]
        #face_lm = landmark[batch_size:]
        #lm_loss = torch.sum((recon_lm - face_lm)**2, dim=-1) * weight
        #lm_loss = torch.sum(lm_loss) / (face_lm.shape[0] * face_lm.shape[1] * 224.0)

        # sudo supervision
        pred_mask, _, pred_face = self.renderer(pred_vertex[:batch_size], self.deep3D.facemodel.face_buf, feat=pred_color[:batch_size])
        pred_mask = F.interpolate(pred_mask, size = cano_render.shape[-2:], mode = 'bilinear', align_corners = False)
        sudo = F.interpolate(pred_face, size = cano_render.shape[-2:], mode = 'bilinear', align_corners = False)
        pred = (cano_render + 1) / 2.0 * pred_mask

        # use parsing mask to add more weights on mouth and eyes
        kernel = torch.ones((5, 5)).to(tar_mask.device)
        dilated_tar_mask = morph.dilation(tar_mask, kernel)
        dilated_tar_mask = F.interpolate(dilated_tar_mask, size = pred.shape[-2:], mode = 'bilinear', align_corners = False)
        #import torchvision.utils as vutils
        #vutils.save_image(pred, 'pred.png')
        #vutils.save_image(sudo, 'sudo.png')
        #vutils.save_image(dilated_tar_mask, 'dilated_tar_mask.png')
        #import pdb; pdb.set_trace()
        #sudo_l1 = torch.nn.functional.l1_loss(pred, sudo)
        diff = torch.abs(pred - sudo)
        #import torchvision.utils as vutils
        #vutils.save_image(pred * dilated_tar_mask, 'pred.png')
        #vutils.save_image(sudo * dilated_tar_mask, 'sudo.png')
        #import pdb; pdb.set_trace()
        #sudo_l1 = (diff * dilated_tar_mask * 3 + diff * (1 - dilated_tar_mask)).mean()
        sudo_l1 = (diff * dilated_tar_mask).mean()
        #return lm_loss, sudo_l1, sudo
        return sudo_l1, sudo

    def get_sudo(self, id_img, pose_img, tar_img, exp_transfer = True):
        """
        - Given an image and a desired expression coefficient, return sudo 3DMM rendering
        """
        b, c, h, w = id_img.shape
        id_img = self.preprocess(id_img)
        pose_img = self.preprocess(pose_img)
        tar_img = self.preprocess(tar_img)
        img = torch.cat((id_img, pose_img, tar_img))
        with torch.no_grad():
            output_coeff = self.deep3D.net_recon(img)

        if exp_transfer:
            # replace expression with target expression
            output_coeff[0, 80: 144] = output_coeff[2, 80:144]
        # replace head pose with target head pose
        output_coeff[0, 224: 227] = output_coeff[1, 224: 227]
        output_coeff[0, 254:] = output_coeff[1, 254:]

        with torch.no_grad():
            pred_vertex, pred_texture, pred_color, landmark = self.deep3D.facemodel.compute_for_render(output_coeff[0:1])
            pred_mask, _, pred_face = self.renderer(pred_vertex, self.deep3D.facemodel.face_buf, feat=pred_color)
        return pred_face * 2 - 1

    def lm_loss(self, pred, gt):
        batch_size = pred.shape[0]
        pred = self.preprocess(pred)
        gt = self.preprocess(gt)
        im_tensor = torch.cat((pred, gt))
        output_coeff = self.deep3D.net_recon(im_tensor)
        pred_vertex, pred_texture, pred_color, landmark = self.deep3D.facemodel.compute_for_render(output_coeff)

        weight = np.ones([68])
        weight[17:27] = 200 # eye brow
        weight[36:47] = 50 # eyes
        weight[-10:] = 200 # mouth
        weight = np.expand_dims(weight, 0)
        weight = torch.tensor(weight).to(landmark.device)
        recon_lm = landmark[:batch_size]
        face_lm = landmark[batch_size:]
        lm_loss = torch.sum((recon_lm - face_lm.detach())**2, dim=-1) * weight
        lm_loss = torch.sum(lm_loss) / (face_lm.shape[0] * face_lm.shape[1] * 224.0)
        return lm_loss

    def forward_with_mask(self, img_tar, img_random, transfer_tar, exp_random, transfer_mask):
        batch_size = img_tar.shape[0]
        img_tar = self.preprocess(img_tar)
        img_random = self.preprocess(img_random)
        transfer_tar_resize = self.preprocess(transfer_tar)
        im_tensor = torch.cat((img_tar.detach(), img_random.detach(), transfer_tar_resize))

        output_coeff = self.deep3D.net_recon(im_tensor)
        img_tar_coeff = output_coeff[:batch_size].detach()
        img_random_coeff = output_coeff[batch_size:2*batch_size].detach()
        transfer_tar_coeff = output_coeff[2*batch_size:]

        img_tar_coeff_ = img_tar_coeff.clone()
        img_tar_random_coeff = img_tar_coeff.clone()
        img_tar_random_coeff[:, 80: 144] = exp_random
        #img_tar_random_coeff[:, 224: 227] = img_random_coeff[:, 224: 227]
        #img_tar_random_coeff[:, 254:] = img_random_coeff[:, 254:]

        # synthesize target image
        pred_vertex, pred_texture, pred_color, landmark = self.deep3D.facemodel.compute_for_render(torch.cat((img_tar_random_coeff, transfer_tar_coeff)))

        weight = np.ones([68])
        weight[17:27] = 200 # eye brow
        weight[36:37] = 50
        weight[-10:] = 200
        weight = np.expand_dims(weight, 0)
        weight = torch.tensor(weight).to(landmark.device)
        face_lm = landmark[:batch_size]
        recon_lm = landmark[batch_size:]
        lm_loss = torch.sum((recon_lm - face_lm.detach())**2, dim=-1) * weight
        lm_loss = torch.sum(lm_loss) / (face_lm.shape[0] * face_lm.shape[1] * 224.0)

        # visualize landmarks
        """
        pred_mask, _, pred_face = self.renderer(pred_vertex, self.deep3D.facemodel.face_buf, feat=pred_color)
        transfer_face = pred_face[batch_size:].detach().cpu().squeeze().permute(1, 2, 0).numpy()
        sudo = pred_face[:batch_size].detach().cpu().squeeze().permute(1, 2, 0).numpy()
        transfer_face = np.array(transfer_face * 255, dtype = np.uint8)
        sudo = np.array(sudo * 255, dtype = np.uint8)
        from imageio import imsave
        import cv2
        face_lm[:, :, 1] = 224 - face_lm[:, :, 1]
        recon_lm[:, :, 1] = 224 - recon_lm[:, :, 1]
        for i in range(36, 38):
            sudo_lm = face_lm[0, i].detach().cpu().numpy()
            cv2.circle(sudo, (int(sudo_lm[0]), int(sudo_lm[1])), 2, (255, 0, 0), thickness=-1)
            transfer_lm = recon_lm[0, i].detach().cpu().numpy()
            cv2.circle(transfer_face, (int(transfer_lm[0]), int(transfer_lm[1])), 2, (255, 0, 0), thickness=-1)
        imsave('sudo_lm.png', sudo)
        imsave('transfer_lm.png', transfer_face)
        import pdb; pdb.set_trace()
        """

        # sudo supervision
        pred_mask, _, pred_face = self.renderer(pred_vertex[:batch_size], self.deep3D.facemodel.face_buf, feat=pred_color[:batch_size])
        pred_mask = F.interpolate(pred_mask, size = transfer_tar.shape[-2:], mode = 'bilinear', align_corners = False)
        sudo = F.interpolate(pred_face, size = transfer_tar.shape[-2:], mode = 'bilinear', align_corners = False)
        pred = (transfer_tar + 1) / 2.0 * pred_mask

        kernel = torch.ones((8, 8)).to(transfer_mask.device)
        dilated_transfer_mask = morph.dilation(transfer_mask, kernel)
        dilated_transfer_mask = F.interpolate(dilated_transfer_mask, size = pred.shape[-2:], mode = 'bilinear', align_corners = False)
        #import torchvision.utils as vutils
        #vutils.save_image(pred * dilated_transfer_mask, 'pred.png')
        #vutils.save_image(sudo * dilated_transfer_mask, 'sudo.png')
        #import pdb; pdb.set_trace()
        sudo_l1 = torch.nn.functional.l1_loss(pred * dilated_transfer_mask, sudo * dilated_transfer_mask)
        return lm_loss, sudo_l1 * 5, sudo, landmark

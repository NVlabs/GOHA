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
from eg3d_modules.torch_utils.ops import bias_act
from eg3d_modules.torch_utils.ops import upfirdn2d

import os
import sys
import time
import yaml
import argparse
import numpy as np
from tqdm import tqdm, trange

sys.path.insert(1, './nerf')
os.environ['GPU_DEBUG']='3'
import torch
import lpips
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from nerf import CfgNode, img2mse, mse2psnr

# DDP related
from torch.distributed import Backend
from torch.utils.data import DistributedSampler
from torch.nn.parallel.distributed import DistributedDataParallel
from torch import nn

import importlib
from dataset.EG3DDatasetMultiVideoMatting import ImageFolderDataset as VideoDataset
from dataset.EG3DDatasetMultiRavdessMatting import ImageFolderDataset as RavdessDataset
from dataset.EG3DDatasetMultiSyntheticMatting import ImageFolderDataset as ImageDataset

class Trainer():
    def __init__(self, configargs):
        # Read config file.
        cfg = None
        with open(configargs.config, "r") as f:
            cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
            cfg = CfgNode(cfg_dict)

        # Seed experiment for repeatability
        seed = cfg.experiment.randomseed
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.world_size = configargs.world_size
        self.rank = configargs.rank
        cfg.device = f'cuda:{configargs.rank}'
        self.cfg = cfg
        self.start_iter = 0

        # Setup logging only for the first GPU
        if self.rank == 0:
            logdir = os.path.join(cfg.experiment.logdir, cfg.experiment.id)
            self.writer = SummaryWriter(logdir)
            self.logdir = logdir

        self.losses = {}

    def load_dataset(self):
        cfg = self.cfg
        self.image_dataset = ImageDataset(path = cfg.dataset.image_path, resolution = 512)
        image_sampler = DistributedSampler(self.image_dataset,
                                           num_replicas = self.world_size,  # Number of GPUs
                                           rank         = self.rank,  # GPU where process is running
                                           shuffle      = True,  # Shuffling is done by Sampler
                                           seed         = 42)
        self.image_loader_ = torch.utils.data.DataLoader(dataset     = self.image_dataset,
                                                         batch_size  = self.cfg.experiment.batch_size,
                                                         num_workers = 2,
                                                         drop_last   = True,
                                                         sampler     = image_sampler,
                                                         pin_memory  = False)
        self.image_loader = iter(self.image_loader_)

        self.video_dataset = VideoDataset(path = cfg.dataset.video_path, resolution = 512)
        video_sampler = DistributedSampler(self.video_dataset,
                                           num_replicas = self.world_size,  # Number of GPUs
                                           rank         = self.rank,  # GPU where process is running
                                           shuffle      = True,  # Shuffling is done by Sampler
                                           seed         = 42)
        self.video_loader_ = torch.utils.data.DataLoader(dataset     = self.video_dataset,
                                                         batch_size  = self.cfg.experiment.batch_size,
                                                         num_workers = 4,
                                                         drop_last   = True,
                                                         sampler     = video_sampler,
                                                         pin_memory  = False)
        self.video_loader = iter(self.video_loader_)

        self.ravdess_dataset = RavdessDataset(path = cfg.dataset.ravdess_path, resolution = 512)
        ravdess_sampler = DistributedSampler(self.ravdess_dataset,
                                             num_replicas = self.world_size,  # Number of GPUs
                                             rank         = self.rank,  # GPU where process is running
                                             shuffle      = True,  # Shuffling is done by Sampler
                                             seed         = 42)
        self.ravdess_loader_ = torch.utils.data.DataLoader(dataset     = self.ravdess_dataset,
                                                           batch_size  = self.cfg.experiment.batch_size,
                                                           num_workers = 4,
                                                           drop_last   = True,
                                                           sampler     = ravdess_sampler,
                                                           pin_memory  = False)
        self.ravdess_loader = iter(self.ravdess_loader_)
        return

    def define_model(self, checkpoint_path = None):
        torch.cuda.set_device(self.rank)
        torch.cuda.empty_cache()
        cfg = self.cfg
        mod = importlib.import_module('models.{}'.format(cfg.models.model_file))
        model = mod.Model(cfg)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            self.start_iter = checkpoint['iter']
            saved_state_dict = checkpoint['model_state_dict']
            new_params = model.state_dict().copy()
            for name, param in new_params.items():
                if name in saved_state_dict and param.size() == saved_state_dict[name].size():
                    new_params[name].copy_(saved_state_dict[name])
                else:
                    print("Missing state:", name)
            model.load_state_dict(new_params)
            print("Load model from {}.".format(checkpoint_path))
        model.to(cfg.device)

        self.model = DistributedDataParallel(model,
                                             device_ids=[self.rank],
                                             output_device=self.rank,
                                             find_unused_parameters=True)
        self.lpips_fn = DistributedDataParallel(lpips.LPIPS(net = 'alex').to(cfg.device))
        return

    def define_optimizer(self):
        my_list = ['module.expr_bases']
        params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in my_list, self.model.named_parameters()))))
        base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in my_list, self.model.named_parameters()))))
        optimizerG = torch.optim.Adam([{"params": params, "lr": 0.01},
                                       {"params": base_params, "lr": self.cfg.optimizer.glr}],
                                      lr = self.cfg.optimizer.glr,
                                      betas = (0, 0.99))
        self.optimizerG = optimizerG
        return

    def load_one_batch_of_syn(self):
        data = {}
        device = self.cfg.device
        try:
            img_target, pose_target, exp_target, img_src, pose_src, exp_src, img_random, pose_random, exp_random, name = self.image_loader.next()
        except StopIteration:
            self.image_loader = iter(self.image_loader_)
            img_target, pose_target, exp_target, img_src, pose_src, exp_src, img_random, pose_random, exp_random, name = self.image_loader.next()

        data['img_target'] = (img_target.to(device).to(torch.float32) / 127.5 - 1)
        data['pose_target'] = pose_target.to(device)
        data['exp_target'] = exp_target.to(device).to(torch.float32)
        data['img_src'] = (img_src.to(device).to(torch.float32) / 127.5 - 1)
        data['pose_src'] = pose_src.to(device)
        data['exp_src'] = exp_src.to(device).to(torch.float32)
        data['exp_random'] = exp_random.to(device).to(torch.float32)
        data['pose_random'] = pose_random.to(device)
        data['img_random'] = (img_random.to(device).to(torch.float32) / 127.5 - 1)
        data['name'] = name
        return data

    def load_one_batch_of_video(self):
        data = {}
        device = self.cfg.device
        try:
            img_target, pose_target, img_src, pose_src, pose_random, exp_target, exp_src, name, img_random, exp_random = self.video_loader.next()
        except StopIteration:
            self.video_loader = iter(self.video_loader_)
            img_target, pose_target, img_src, pose_src, pose_random, exp_target, exp_src, name, img_random, exp_random = self.video_loader.next()

        data['img_target'] = (img_target.to(device).to(torch.float32) / 127.5 - 1)
        data['pose_target'] = pose_target.to(device)
        data['pose_random'] = pose_random.to(device)
        data['img_src'] = (img_src.to(device).to(torch.float32) / 127.5 - 1)
        data['pose_src'] = pose_src.to(device)
        data['exp_target'] = exp_target.to(device).to(torch.float32)
        data['exp_src'] = exp_src.to(device).to(torch.float32)
        data['img_random'] = (img_random.to(device).to(torch.float32) / 127.5 - 1)
        data['exp_random'] = exp_random.to(device).to(torch.float32)
        data['name'] = name
        return data

    def load_one_batch_of_ravdess(self):
        data = {}
        device = self.cfg.device
        try:
            img_target, pose_target, img_src, pose_src, pose_random, exp_target, exp_src, name, img_random, exp_random = self.ravdess_loader.next()
        except StopIteration:
            self.ravdess_loader = iter(self.ravdess_loader_)
            img_target, pose_target, img_src, pose_src, pose_random, exp_target, exp_src, name, img_random, exp_random = self.ravdess_loader.next()

        data['img_target'] = (img_target.to(device).to(torch.float32) / 127.5 - 1)
        data['pose_target'] = pose_target.to(device)
        data['pose_random'] = pose_random.to(device)
        data['img_src'] = (img_src.to(device).to(torch.float32) / 127.5 - 1)
        data['pose_src'] = pose_src.to(device)
        data['exp_target'] = exp_target.to(device).to(torch.float32)
        data['exp_src'] = exp_src.to(device).to(torch.float32)
        data['img_random'] = (img_random.to(device).to(torch.float32) / 127.5 - 1)
        data['exp_random'] = exp_random.to(device).to(torch.float32)
        data['name'] = name
        return data

    def save_checkpoint(self, i):
        checkpoint_dict = {
            "iter": i,
            "model_state_dict": self.model.module.state_dict()
        }
        torch.save(
            checkpoint_dict,
            os.path.join(self.logdir, "checkpoint" + str(i).zfill(5) + ".ckpt"),
        )
        tqdm.write("================== Saved Checkpoint =================")
        return

    def logging(self, i, losses):
        out_str = "[TRAIN] Iter: " + str(i)
        losses['lr'] = self.optimizerG.param_groups[1]['lr']
        for k,v in losses.items():
            out_str = out_str + " " + k + ": {:.4f}".format(v)
            self.writer.add_scalar("train/{}".format(k), v, i)
        tqdm.write(out_str)
        return

    def validate_step(self, i):
        self.model.eval()

        start = time.time()
        with torch.no_grad():

            if self.rank == 0:
                if hasattr(self, 'rgb_low_res'):
                    self.writer.add_image(
                        "target_view_rec/img_target_low_res",
                        ((self.rgb_low_res[0] + 1) / 2.0).clamp_(0.0, 1.0).squeeze(),
                        i,
                    )
                if hasattr(self, 'src_gt'):
                    self.writer.add_image(
                        "source_view_rec/src_gt",
                        ((self.src_gt[0] + 1) / 2.0).squeeze(),
                        i,
                    )
                    self.writer.add_image(
                        "source_view_rec/src_rec",
                        ((self.src_rec[0] + 1) / 2.0).clamp_(0.0, 1.0).squeeze(),
                        i,
                    )
                if hasattr(self, 'gt'):
                    loss = img2mse((self.rgb_low_res + 1) / 2.0, (self.gt + 1) / 2.0)
                    psnr = mse2psnr(loss.item())
                    self.writer.add_scalar("validation/loss", loss.item(), i)
                    self.writer.add_scalar("validation/psnr", psnr, i)
                    self.writer.add_image(
                        "target_view_rec/img_target_gt",
                        ((self.gt[0] + 1) / 2.0).squeeze(),
                        i,
                    )
                    self.writer.add_image(
                        "target_view_rec/img_target_input",
                        ((self.input[0] + 1) / 2.0).squeeze(),
                        i,
                    )
                    tqdm.write(
                        "Validation loss: "
                        + str(loss.item())
                        + " Validation PSNR: "
                        + str(psnr)
                        + " Time: "
                        + str(time.time() - start)
                    )
                if hasattr(self, 'depth'):
                    depth = -self.depth[0]
                    depth = (depth - depth.min()) / ((depth.max() - depth.min()))
                    depth = depth.clamp(0.0, 1.0).repeat(3, 1, 1)
                    self.writer.add_image(
                        "target_view_rec/depth_target",
                        (depth).squeeze(),
                        i,
                    )
                if hasattr(self, 'neutral_sudo'):
                    self.writer.add_image(
                        "canonical/neutral_sudo",
                        ((self.neutral_sudo[0] + 1) / 2.0).clamp_(0.0, 1.0).squeeze(),
                        i,
                    )
                if hasattr(self, 'coarse_target'):
                    self.writer.add_image(
                        "canonical/coarse_target",
                        ((self.coarse_target[0] + 1) / 2.0).squeeze(),
                        i,
                    )
                    self.writer.add_image(
                        "canonical/coarse_gt",
                        ((self.coarse_gt[0] + 1) / 2.0).squeeze(),
                        i,
                    )
                    self.writer.add_image(
                        "canonical/coarse_input",
                        ((self.coarse_input[0] + 1) / 2.0).squeeze(),
                        i,
                    )
                if hasattr(self, 'neutral_render'):
                    self.writer.add_image(
                        "canonical/neutral_render",
                        ((self.neutral_render[0] + 1) / 2.0).squeeze(),
                        i,
                    )
                    self.writer.add_image(
                        "canonical/cano_input",
                        ((self.cano_input[0] + 1) / 2.0).squeeze(),
                        i,
                    )
                    self.writer.add_image(
                        "canonical/cano_target",
                        ((self.cano_target[0] + 1) / 2.0).squeeze(),
                        i,
                    )
                if hasattr(self, 'sudo_src'):
                    self.writer.add_image(
                        "target_view_rec/sudo_src",
                        (self.sudo_src[0] + 1) / 2.0,
                        i,
                    )
                    self.writer.add_image(
                        "target_view_rec/sudo_tar",
                        (self.sudo_tar[0] + 1) / 2.0,
                        i,
                    )

        return

    def tv_loss(self, data, outputs):
        batch_size = data['img_target'].shape[0]
        device = data['img_target'].device
        initial_coordinates = torch.rand((batch_size, 1000, 3), device=device) * 2 - 1
        perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * 0.004
        all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
        sigma = self.model.module.sample_mixed(all_coordinates, outputs['target_plane'])['sigma']
        sigma_initial = sigma[:, :sigma.shape[1]//2]
        sigma_perturbed = sigma[:, sigma.shape[1]//2:]

        TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.cfg.weights.TV_weight
        return TVloss

    def generator_match_step(self, data, outputs, iter_num):
        pred = outputs['rec']
        gt = outputs['gt']
        l1_loss = F.l1_loss(pred, gt)
        lpips_loss = self.lpips_fn(pred, gt)

        # regularization
        TVloss = self.tv_loss(data, outputs)

        if 'id_planes' in outputs:
            id_planes_loss = torch.norm(outputs['id_planes'])
        else:
            id_planes_loss = torch.zeros_like(l1_loss)
        return l1_loss, lpips_loss.mean(), id_planes_loss, TVloss

    def load_one_batch_of_data(self):
        seed = torch.rand(1)
        if seed < 0.33:
            data = self.load_one_batch_of_ravdess()
            exp_transfer = True
            type = 'ravdess'
        elif seed >= 0.33 and seed < 0.66:
            data = self.load_one_batch_of_video()
            exp_transfer = True
            type = 'celebvhq'
        else:
            data = self.load_one_batch_of_syn()
            exp_transfer = False
            type = 'synthetic'
        return data, exp_transfer, type

    def train_step(self, iter_num):
        losses = {}
        self.model.train()
        self.model.module.face_parsing_net.eval()
        if hasattr(self.model.module, 'exp_loss_fn') and hasattr(self.model.module.exp_loss_fn, 'recog_net'):
            self.model.module.exp_loss_fn.recog_net.eval()
        if hasattr(self.model.module, 'exp_loss_fn') and hasattr(self.model.module.exp_loss_fn, 'deep3D'):
            self.model.module.exp_loss_fn.deep3D.eval()

        data, exp_transfer, type = self.load_one_batch_of_data()

        self.model.zero_grad()
        if iter_num > self.cfg.experiment.add_expr_plane and iter_num % 6 == 0:
            outputs = self.model(data, mode = 'full_rec_src', iteration = iter_num, exp_transfer = exp_transfer)
            l1_loss, lpips_loss, id_planes_loss, TVloss = self.generator_match_step(data, outputs, iter_num)
            g_loss = (l1_loss + lpips_loss) * self.cfg.weights.rec_loss_weight + TVloss
            self.src_gt = outputs['gt'].detach().cpu()
            self.src_rec = outputs['rec'].detach().cpu()
            if 'lm_loss' in outputs:
                g_loss = g_loss + outputs['lm_loss'] * self.cfg.weights.lm_loss_weight
        elif iter_num > self.cfg.experiment.add_expr_plane and iter_num % 6 == 1:
            outputs = self.model(data, mode = 'full_rec_tar', iteration = iter_num, exp_transfer = exp_transfer)
            l1_loss, lpips_loss, id_planes_loss, TVloss = self.generator_match_step(data, outputs, iter_num)
            g_loss = (l1_loss + lpips_loss) * self.cfg.weights.rec_loss_weight + TVloss

            if 'sudo_src' in outputs:
                self.sudo_src = F.interpolate(outputs['sudo_src'], size = outputs['rec'].shape[-2:], mode = 'bilinear', align_corners = False)
                self.sudo_tar = F.interpolate(outputs['sudo_tar'], size = outputs['rec'].shape[-2:], mode = 'bilinear', align_corners = False)

            self.losses['id_planes_loss'] = id_planes_loss.item()
            self.losses['full_l1_loss'] = l1_loss.item()
            self.losses['full_lpips_loss'] = lpips_loss.item()
            self.losses['TVloss'] = TVloss.item()
            self.gt = outputs['gt'].detach().cpu()
            self.input = outputs['input'].detach().cpu()
            self.rgb_low_res = outputs['rec'].detach().cpu()
            self.depth = outputs['depth_image'].detach().cpu()

            if 'lm_loss' in outputs:
                g_loss = g_loss + outputs['lm_loss'] * self.cfg.weights.lm_loss_weight
                self.losses['lm_loss'] = outputs['lm_loss'].detach().cpu()
        elif iter_num > self.cfg.experiment.add_expr_plane and (iter_num % 6 == 2 or iter_num % 6 == 3):
            outputs = self.model(data, mode = 'coarse_target', iteration = iter_num, exp_transfer = exp_transfer)
            pred = outputs['pred']; gt = outputs['gt']
            l1_loss = F.l1_loss(pred, gt)
            lpips_loss = self.lpips_fn(pred, gt).mean()
            g_loss = l1_loss + lpips_loss
            if iter_num % 4 == 0:
                g_loss += self.tv_loss(data, outputs)
            self.coarse_target = pred.detach().cpu()
            self.coarse_input = outputs['input'].detach().cpu()
            self.coarse_gt = outputs['gt'].detach().cpu()

            if 'lm_loss' in outputs:
                g_loss = g_loss + outputs['lm_loss'] * self.cfg.weights.lm_loss_weight
        else:
            outputs = self.model(data, mode = 'cano', iteration = iter_num, exp_transfer = exp_transfer)
            pred = outputs['pred']; gt = outputs['gt']
            l1_loss = F.l1_loss(pred, gt)
            lpips_loss_ = self.lpips_fn(pred, gt)
            bs = self.cfg.experiment.batch_size
            lpips_loss = (lpips_loss_[:bs] * self.cfg.weights.neutral_weight + lpips_loss_[bs:]).mean()
            g_loss = l1_loss + lpips_loss
            if iter_num % 6 == 4:
                g_loss += self.tv_loss(data, outputs)
            if 'neutral_sudo' in outputs:
                self.neutral_sudo = outputs['neutral_sudo']
                self.neutral_render = outputs['neutral_render'].detach().cpu()
            self.cano_input = outputs['cano_input'].detach().cpu()
            self.cano_target = outputs['cano_target'].detach().cpu()
            self.losses['coarse_l1_loss'] = l1_loss.item()
            self.losses['coarse_lpips_loss'] = lpips_loss.item()

        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.experiment.grad_clip_norm)
        self.optimizerG.step()
        return losses

    def adjust_lr(self, i, optimizer, init_lr):
        num_decay_steps = self.cfg.scheduler.lr_decay * 10000
        lr_new = init_lr * (
            self.cfg.scheduler.lr_decay_factor ** (i / num_decay_steps)
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_new
        return

    def train(self):
        cfg = self.cfg
        print("Training started")
        for i in trange(self.start_iter, cfg.experiment.train_iters):
            self.train_step(i)

            # Learning rate updates
            #self.adjust_lr(i, self.optimizerG, self.cfg.optimizer.glr)
            #self.adjust_lr(i, self.optimizerD, self.cfg.optimizer.dlr_init)

            if (i % cfg.experiment.print_every == 0 or i == cfg.experiment.train_iters - 1) and self.rank == 0:
                self.logging(i, self.losses)

            if i % cfg.experiment.validate_every == 0:
                self.validate_step(i)

            if (i % cfg.experiment.save_every == 0 or i == cfg.experiment.train_iters - 1) and self.rank == 0:
                self.save_checkpoint(i)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default="",
        help="Path to load saved checkpoint from.",
    )
    parser.add_argument("--local_rank", type=int)
    configargs = parser.parse_args()
    configargs.rank = configargs.local_rank
    configargs.world_size = torch.cuda.device_count()

    torch.cuda.set_device(configargs.rank)
    torch.distributed.init_process_group(backend=Backend.NCCL, init_method='env://')
    trainer = Trainer(configargs)
    trainer.load_dataset()
    trainer.define_model(configargs.load_checkpoint)
    trainer.define_optimizer()
    trainer.train()


if __name__ == "__main__":
    main()

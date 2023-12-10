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

import os
import sys
import yaml
import imageio
import argparse
import numpy as np
from tqdm import tqdm

sys.path.insert(1, './nerf')
os.environ['GPU_DEBUG']='3'
import torchvision.utils as vutils
import torch.nn.functional as F

from utils.cfgnode import CfgNode

import importlib
from dataset.Dataset import ImageFolderDataset as ImageDataset

import PIL.Image

class Tester():
    def __init__(self, configargs):
        # Read config file.
        cfg = None
        with open(configargs.config, "r") as f:
            cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
            cfg = CfgNode(cfg_dict)

        cfg.device = f'cuda:0'
        self.cfg = cfg

        # create output folders
        os.makedirs(os.path.join(configargs.savedir, 'source'), exist_ok = True)
        os.makedirs(os.path.join(configargs.savedir, 'target'), exist_ok = True)
        os.makedirs(os.path.join(configargs.savedir, 'low_res'), exist_ok = True)
        self.savedir = configargs.savedir

        self.transfer_pose = (configargs.transfer_pose == 1)

    def load_dataset(self):
        cfg = self.cfg
        device = cfg.device
        self.image_dataset = ImageDataset(path = '/raid/celeba', resolution = 512)
        self.image_loader_ = torch.utils.data.DataLoader(dataset     = self.image_dataset,
                                                         batch_size  = 1,
                                                         num_workers = 4,
                                                         drop_last   = True,
                                                         pin_memory  = False,
                                                         shuffle     = False)
        self.image_loader = iter(self.image_loader_)
        return

    def define_model(self, checkpoint_path):
        cfg = self.cfg
        mod = importlib.import_module('models.{}'.format(cfg.models.model_file))
        model = mod.Model(cfg, test = True)
        assert os.path.exists(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location = torch.device('cpu'))
        self.iter_num = checkpoint['iter']
        saved_state_dict = checkpoint['model_state_dict']
        new_params = model.state_dict().copy()
        for name, param in new_params.items():
            if name in saved_state_dict and param.size() == saved_state_dict[name].size():
                new_params[name].copy_(saved_state_dict[name])
            else:
                print("=====> Missing state:", name)
        model.load_state_dict(new_params)
        self.model = model.to(cfg.device)
        self.model.eval()
        print("=====> Loaded checkpoint from {} at iteration {}.".format(checkpoint_path, self.iter_num))
        return

    def load_one_batch_of_img(self):
        data = {}
        device = self.cfg.device
        try:
            img_target, pose_target, img_src, pose_src, pose_random, exp_target, exp_src, name, img_random, exp_random, matting_mask = self.image_loader.next()
        except StopIteration:
            return None

        data['img_target'] = (img_target.to(device).to(torch.float32) / 127.5 - 1)
        data['img_random'] = (img_random.to(device).to(torch.float32) / 127.5 - 1)
        data['pose_target'] = pose_target.to(device)
        data['pose_random'] = pose_random.to(device)
        data['img_src'] = (img_src.to(device).to(torch.float32) / 127.5 - 1)
        data['pose_src'] = pose_src.to(device)
        data['exp_target'] = exp_target.to(device).to(torch.float32)
        data['exp_src'] = exp_src.to(device).to(torch.float32)
        data['exp_random'] = exp_random.to(device).to(torch.float32)
        data['img_name'] = name
        data['type'] = 'test'

        # load matting mask for foreground point cloud computation
        matting_path = name[0].replace('images', 'matting')
        matting_mask = PIL.Image.open(matting_path)
        matting_mask = np.asarray(matting_mask) / 255.0
        data['matting'] = torch.from_numpy(matting_mask).view(1, 1, matting_mask.shape[-2], matting_mask.shape[-1])
        return data

    def test_step(self, iteration):
        # prepare data
        data = self.load_one_batch_of_img()
        data['img_src'] = data['img_target']
        data['pose_src'] = data['pose_target']

        tar_data = self.load_one_batch_of_img()
        data['img_target'] = tar_data['img_target']
        data['exp_target'] = tar_data['exp_target']
        if self.transfer_pose:
            data['pose_target'] = tar_data['pose_target']
        src_img_name = data['img_name'][0].split('/')[-1].split('.')[0]
        tar_img_name = tar_data['img_name'][0].split('/')[-1].split('.')[0]

        # forward
        with torch.no_grad():
            outputs = self.model(data, mode = 'full_rec_tar', iteration = self.iter_num)
            if 'high_res' in outputs:
                low_res = outputs['high_res']
            elif 'high_res_target_rec' in outputs:
                low_res = outputs['high_res_target_rec']
            else:
                if 'rgb_low_res' in outputs:
                    low_res = outputs['rgb_low_res']
                elif 'low_res_target_rec' in outputs:
                    low_res = outputs['low_res_target_rec']
                else:
                    low_res = outputs['target_rec']
            low_res = (low_res * 127.5 + 128).clamp(0, 255)

        out_path = os.path.join(self.savedir, "target/{}.png".format(src_img_name))
        vutils.save_image((data['img_target'] + 1) / 2.0, out_path)
        out_path = os.path.join(self.savedir, "source/{}.png".format(tar_img_name))
        vutils.save_image((data['img_src'] + 1) / 2.0, out_path)
        out_path = os.path.join(self.savedir, "low_res/{}_{}.png".format(src_img_name, tar_img_name))
        vutils.save_image(low_res[:1]/255.0, out_path)

    def test(self):
        print("===> Testing started.")
        for iteration in tqdm(range(self.test_sample_number // 2)):
            self.test_step(iteration)
        print("===> Testing done.")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to load saved checkpoint from.",
    )
    parser.add_argument(
        "--savedir", type=str, default='./renders/', help="Save images to this directory, if specified."
    )
    parser.add_argument(
        "--transfer_pose", type=int, default=1, help="Do we want to transfer pose from target to source?"
    )
    parser.add_argument(
        "--test_sample_number", type=int, default=None, help="How many images to test on"
    )
    configargs = parser.parse_args()

    tester = Tester(configargs)
    tester.load_dataset()
    tester.define_model(configargs.checkpoint)

    if configargs.test_sample_number is None:
        tester.test_sample_number = len(tester.image_dataset)
    else:
        tester.test_sample_number = configargs.test_sample_number
    tester.test()

if __name__ == "__main__":
    main()

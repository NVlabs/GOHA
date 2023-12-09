# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import cv2
import yaml
import torch
import argparse
import importlib
import numpy as np
from tqdm import tqdm

from dataset.Dataset import ImageFolderDataset as ImageDataset
from dataset.Dataset import ImageFolderDataset as VideoDataset

from utils.cfgnode import CfgNode
from utils.camera_utils import LookAtPoseSampler, FOV_to_intrinsics

class Tester():
    def __init__(self, configargs):
        """
        Tester initialization:
         - read config file
         - create output folders
        """
        cfg = None
        with open(configargs.config, "r") as f:
            cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
            cfg = CfgNode(cfg_dict)
        cfg.device = f'cuda:0'
        self.cfg = cfg

        os.makedirs(configargs.savedir, exist_ok = True)
        self.savedir = configargs.savedir

        self.source_path = configargs.source_path
        self.target_path = configargs.target_path
        self.frame_limit = configargs.frame_limit
        self.image_limit = configargs.image_limit

    def load_dataset(self):
        """
        Create source images and target video dataset:
         - self.image_dataset: source image dataset
         - self.tgt_video_dataset: drive video dataset
        """
        self.image_dataset = ImageDataset(path = self.source_path, resolution = 512)
        self.image_loader_ = torch.utils.data.DataLoader(dataset     = self.image_dataset,
                                                         batch_size  = 1,
                                                         num_workers = 4,
                                                         drop_last   = True,
                                                         pin_memory  = False)
        self.image_loader = iter(self.image_loader_)

        self.tgt_video_dataset = VideoDataset(path = self.target_path, resolution = 512)
        self.tgt_video_loader_ = torch.utils.data.DataLoader(dataset     = self.tgt_video_dataset,
                                                             batch_size  = 1,
                                                             num_workers = 4,
                                                             drop_last   = True,
                                                             pin_memory  = False)
        self.tgt_video_loader = iter(self.tgt_video_loader_)
        return

    def define_model(self, checkpoint_path):
        """
        Define and load pre-trained model.
        """
        cfg = self.cfg
        mod = importlib.import_module('models.{}'.format(cfg.models.model_file))
        model = mod.Model(cfg, test = True)
        assert os.path.exists(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location = torch.device('cpu'))
        self.iter_num = checkpoint['iter']
        saved_state_dict = checkpoint['model_state_dict']
        model.load_state_dict(saved_state_dict)
        self.model = model.to(cfg.device)
        self.model.eval()
        print("===> Loaded checkpoint from {} at iteration {}.".format(checkpoint_path, self.iter_num))
        return

    def load_one_batch_of_video(self):
        """
        Load a batch of drive video frame.
        """
        data = {}
        device = self.cfg.device
        try:
            img_target, pose_target, img_src, pose_src, pose_random, exp_target, exp_src, name, img_random, exp_random, _ = self.tgt_video_loader.next()
        except StopIteration:
            return None

        data['img_target'] = (img_target.to(device).to(torch.float32) / 127.5 - 1)
        data['pose_target'] = pose_target.to(device)
        data['pose_random'] = pose_random.to(device)
        data['img_src'] = (img_src.to(device).to(torch.float32) / 127.5 - 1)
        data['img_random'] = (img_src.to(device).to(torch.float32) / 127.5 - 1)
        data['pose_src'] = pose_src.to(device)
        data['exp_target'] = exp_target.to(device).to(torch.float32)
        data['exp_src'] = exp_src.to(device).to(torch.float32)
        data['exp_random'] = exp_target.to(device).to(torch.float32)
        return data

    def load_one_batch_of_tgt(self):
        """
        Load a batch of source image.
        """
        data = {}
        device = self.cfg.device
        try:
            img_target, pose_target, img_src, pose_src, pose_random, exp_target, exp_src, name, img_random, exp_random, _ = self.image_loader.next()
        except StopIteration:
            return None

        data['img_target'] = (img_target.to(device).to(torch.float32) / 127.5 - 1)
        data['pose_target'] = pose_target.to(device)
        data['pose_random'] = pose_random.to(device)
        data['img_src'] = (img_src.to(device).to(torch.float32) / 127.5 - 1)
        data['pose_src'] = pose_src.to(device)
        data['exp_target'] = exp_target.to(device).to(torch.float32)
        data['exp_src'] = exp_src.to(device).to(torch.float32)
        data['exp_random'] = exp_random.to(device).to(torch.float32)
        data['img_random'] = (img_random.to(device).to(torch.float32) / 127.5 - 1)
        data['name'] = name
        return data
    
    def torch_to_np(self, img):
        """
        Convert a torch tensor image to a numpy image.
         - Input: torch.Tensor of size (1, c, h, w) in range [-1, 1].
         - Output: numpy array of size (h, w, c) in range [0, 255].
        """
        img = (img + 1) / 2.0
        img = img * 255
        img = img.clamp_(0, 255)
        img = img.detach().cpu().squeeze().permute(1, 2, 0).numpy()
        img = np.array(img, dtype = np.uint8)
        return img
    
    def sample_random_camera(self, iteration, device):
        """
        Sample a rendering camera view based on iteration.
        """
        camera_lookat_point = torch.tensor([0, 0, 0.2], device=device)
        fov_deg = 18.837
        intrinsics = FOV_to_intrinsics(fov_deg, device=device)
        pitch_range = 0.25
        yaw_range = 0.35
        w_frames = 4 * 60
        frame_idx = iteration % w_frames
        cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * frame_idx / w_frames), 3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (w_frames)), camera_lookat_point, radius=2.7, device=device)
        c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        return c

    def test_step(self, iteration, tgt_data):
        """
        Run testing on a drive frame and source image.
        """
        # prepare data
        data = self.load_one_batch_of_video()
        tgt_data['img_target'] = data['img_target']
        if data is None:
            return

        # render at target view
        tgt_data['pose_target'] = data['pose_target']
        with torch.no_grad():
            outputs = self.model(tgt_data, iteration = self.iter_num, mode = 'full_rec_tar')
        pred_tgt_view = self.torch_to_np(outputs['high_res'])

        # render at random view
        c = self.sample_random_camera(iteration, self.cfg.device)
        tgt_data['pose_target'] = c.repeat(data['img_target'].shape[0], 1)

        with torch.no_grad():
            outputs = self.model(tgt_data, iteration = self.iter_num, mode = 'full_rec_tar')
        pred_random_view = self.torch_to_np(outputs['high_res'])

        drive_frame = self.torch_to_np(tgt_data['img_target'])
        src_img = self.torch_to_np(tgt_data['img_src'])
        out_frame = np.concatenate([src_img, drive_frame, pred_tgt_view, pred_random_view], axis = 1)
        out_frame = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
        return out_frame

    def test(self):
        print("===> Testing started.")
        for i in range(min(len(self.image_dataset), self.image_limit)):
            tgt_data = self.load_one_batch_of_tgt()
            name = tgt_data['name'][0].split('/')[-1].split('.')[0]
            tgt_data['img_target'] = tgt_data['img_target'][:, :3]
            tgt_data['img_src'] = tgt_data['img_src'][:, :3]

            out_path = os.path.join(self.savedir, f'{name}.mp4')
            cap_out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (512*4, 512))

            self.tgt_video_loader = iter(self.tgt_video_loader_)
            for iteration in tqdm(range(min(len(self.tgt_video_dataset), self.frame_limit))):
                out_frame = self.test_step(iteration, tgt_data)
                cap_out.write(out_frame)
            cap_out.release()
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
        "--savedir", type=str, default='./renders/', help="Save animation videos to this directory."
    )
    parser.add_argument(
        "--source_path", type=str, default='/raid/celeba', help="Souce image path."
    )
    parser.add_argument(
        "--target_path", type=str, default='/raid/person_2_test', help="Drive video path."
    )
    parser.add_argument(
        "--frame_limit", type=int, default=1000, help="frame number limitation"
    )
    parser.add_argument(
        "--image_limit", type=int, default=1, help="image number limitation"
    )
    configargs = parser.parse_args()

    tester = Tester(configargs)
    tester.load_dataset()
    tester.define_model(configargs.checkpoint)
    tester.test()

if __name__ == "__main__":
    main()

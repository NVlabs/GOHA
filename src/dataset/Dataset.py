# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import cv2
import numpy as np
import zipfile
import PIL.Image
import json
import torch
from glob import glob
import natsort

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
    ):
        self._path = path

        self._all_videos = glob(os.path.join(self._path, '*'))
        self._video_num = 0
        self._frame_num = 0

        self.meta = {}
        self.inverse_meta = {}
        self._all_fnames = []
        self._all_poses = []
        PIL.Image.init()
        for i, video_path in enumerate(self._all_videos):
            img_list = glob(os.path.join(video_path, 'images', '*.png'))
            json_path = os.path.join(video_path, 'dataset.json')
            with open(json_path, 'rb') as f:
                labels = json.load(f)['labels']
            label_dict = {}
            for label in labels:
                label_dict[label[0]] = [label[1]]
            fnames = []
            poses = []
            exps = []
            for j, (img_name, [pose]) in enumerate(natsort.natsorted(label_dict.items())):
                # EG3D pre-processing code automatically creates mirrored image, which we do not use here, so skip
                if '_mirror' in img_name:
                    continue
                fnames.append(os.path.join(video_path, 'images', img_name.replace(".jpg", ".png")))
                poses.append(np.array(pose))
                self.meta[self._frame_num] = "{}, {}".format(self._video_num, j) # record frame in video i and frame j
                self.inverse_meta["{},{}".format(self._video_num, j)] = self._frame_num
                self._frame_num += 1
            self._video_num += 1
            self._all_poses.append(poses)
            self._all_fnames.append(fnames)

        print("Found {} videos with {} frames in total.".format(self._video_num, self._frame_num))
        self._resolution = resolution

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _load_raw_image(self, fname):
        with open(fname, 'rb') as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_parsing_image(self, parsing_path):
        mask = PIL.Image.open(parsing_path)
        mask = np.asarray(mask) / 255.0
        return mask

    def __getitem__(self, idx):
        meta = self.meta[idx]
        ids = meta.split(",")
        video_id = int(ids[0])
        frame_id = int(ids[1])
        fname = self._all_fnames[video_id][frame_id]
        image = self._load_raw_image(fname)
        parsing_path = fname.replace("images", "matting")
        mask = self._load_parsing_image(parsing_path)
        image_masked = image * mask.reshape(1, mask.shape[0], mask.shape[1])

        assert isinstance(image, np.ndarray)
        assert image.dtype == np.uint8
        pose_target = torch.from_numpy(self._all_poses[video_id][frame_id]).float()

        # randomly sample a pose
        random_video_id = int(np.random.choice(self._video_num, size = 1))
        while len(self._all_fnames[random_video_id]) == 0:
            random_video_id = int(np.random.choice(self._video_num, size = 1))
        random_frame_id = int(np.random.choice(len(self._all_fnames[random_video_id]), size = 1))
        pose_random = torch.from_numpy(self._all_poses[random_video_id][random_frame_id]).float()
        pose_random_image = self._load_raw_image(self._all_fnames[random_video_id][int(random_frame_id)])
        pose_random_parsing_path = self._all_fnames[random_video_id][int(random_frame_id)].replace("images", "matting")
        pose_random_mask = self._load_parsing_image(pose_random_parsing_path)
        pose_random_image_masked = pose_random_image * pose_random_mask.reshape(1, pose_random_mask.shape[0], pose_random_mask.shape[1])

        # TODO: legacy code to keep return values neat, should be thrown away across all related files.
        exp = pose_random
        exp_random = pose_random

        return image_masked, pose_target, image_masked, pose_target, pose_random, exp, exp, fname, pose_random_image_masked, exp_random, mask

    def __len__(self):
        return self._frame_num

#----------------------------------------------------------------------------

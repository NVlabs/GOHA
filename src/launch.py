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
import argparse
import yaml
from nerf import CfgNode
from datetime import datetime

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--train_file", type=str, required=True, help="Path to (.py) train file."
    )
    parser.add_argument(
        "-c", "--config_file", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "-l", "--load-checkpoint", type=str, default="", help="Path to load saved checkpoint from."
    )
    parser.add_argument("-g", "--gpu_num", type=int, default=1)
    args = parser.parse_args()

    print("=====> Save configurations")
    with open(args.config_file, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)
    logdir = os.path.join(cfg.experiment.logdir, cfg.experiment.id)
    os.makedirs(logdir, exist_ok=True)
    current_time = datetime.now()
    code_dir = os.path.join(logdir, 'code-{}-{}-{}'.format(current_time.year, current_time.month, current_time.day))
    os.makedirs(code_dir, exist_ok=True)
    os.makedirs(os.path.join(code_dir, 'models'), exist_ok=True)

    with open(os.path.join(logdir, "config.yml"), "w") as f:
        f.write(cfg.dump())
        f.close()

    # copy code
    print("=====> Save code")
    cmd = 'cp training_files/{} {}'.format(args.train_file, code_dir)
    os.system(cmd)
    cmd = 'cp models/{}.py {}'.format(cfg.models.model_file, os.path.join(code_dir, 'models'))
    os.system(cmd)
    cmd = 'cp -r ./dataset {}'.format(code_dir)
    os.system(cmd)
    cmd = 'cp models/networks.py {}'.format(os.path.join(code_dir, 'models'))
    os.system(cmd)
    cmd = 'cp -r ./models/volumetric_rendering {}'.format(os.path.join(code_dir, 'models'))
    os.system(cmd)
    cmd = 'cp -r ./decalib {}'.format(code_dir)
    os.system(cmd)
    cmd = 'cp -r ./losses {}'.format(code_dir)
    os.system(cmd)

    # grab training file to the current folder
    cmd = 'cp training_files/{} ./'.format(args.train_file)
    os.system(cmd)

    cmd = 'python -m torch.distributed.launch --nproc_per_node={} {} --config {}'.format(args.gpu_num, args.train_file, args.config_file)
    if os.path.isfile(args.load_checkpoint):
        cmd += " --load-checkpoint {}".format(args.load_checkpoint)
    f = open(os.path.join(code_dir, "cmd-{}-{}-{}.txt".format(current_time.year, current_time.month, current_time.day)), 'w')
    f.write(cmd)
    f.close()
    print("=====> Training: {}".format(cmd))
    os.system(cmd)
    print("=====> Done!")

if __name__ == "__main__":
    main()

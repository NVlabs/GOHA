# Generalizable One-shot Neural Head Avatar

[[project page](https://research.nvidia.com/labs/lpr/one-shot-avatar)] [[paper](https://arxiv.org/pdf/2306.08768.pdf)]

This repository is the official PyTorch implementation of the following paper:

**Generalizable One-shot Neural Head Avatar**, *[Xueting Li](https://research.nvidia.com/labs/lpr/author/xueting-li/), [Shalini De Mello](https://research.nvidia.com/labs/lpr/author/shalini-de-mello/), [Sifei Liu](https://research.nvidia.com/labs/lpr/author/sifei-liu/), [Koki Nagano](https://luminohope.org), [Umar Iqbal](https://research.nvidia.com/labs/lpr/author/umar-iqbal/), [Jan Kautz](https://research.nvidia.com/labs/lpr/author/jan-kautz/)*.

![](figs/teaser.gif)

## Citation
If you find our work useful in your research, please cite:
```
@article{li2023goha,
  title={Generalizable One-shot Neural Head Avatar},
  author={Li, Xueting and De Mello, Shalini and Liu, Sifei and Nagano, Koki and Iqbal, Umar and Kautz, Jan},
  journal={NeurIPS},
  year={2023}
}
```

## Environment Setup
Our training is carried out on 8 V100 32GB GPUs, while testing can be run on a single V100 16GB GPU. We develop our code on Ubuntu 18.04.5, with GPU driver version 535.54.03 and CUDA 11.3.

<details>
<summary> Package Installation </summary>

Install all packages by `sh install.sh`.
</details>

<details>
<summary> Dependency Modules </summary>

Please see [here]() for instructions.
</details>

## Demo
<details>
<summary> Demo data download </summary>

We provide pre-processed demo data including a single-view portrait image from [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and a drive video from [NeRFace](https://gafniguy.github.io/4D-Facial-Avatars/). It can be downloaded [here](https://drive.google.com/file/d/18WknUovVO4v-Z9_hNl63LzQkAHEhd6oL/view?usp=sharing). The `celeba` folder includes the source portrait image while the `person_2_test` folder contains the drive video. To test on your own images, please preprocess the data following dataset preprocessing instructions in the [Training]() section.
</details>

<details>
<summary> Pre-trained model download </summary>

Download the pre-trained model from [google drive](https://drive.google.com/file/d/1Fiz_AddgbAinh2ZsRn3MwAR0Qcl4o60V/view?usp=share_link) and put the folder in `src/logs/`. The pre-trained model is subject to the [Creative Commons — Attribution-NonCommercial-ShareAlike 4.0 International — CC BY-NC-SA 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) terms. 
</details>

<details>
<summary> Run the animation demo </summary>

Run the one-shot animation demo by:
```
cd src
python demo.py --config configs/s2.yml --checkpoint logs/s3/checkpoint825000.ckpt --savedir output --source_path ../goha_demo_data/celeba/ --target_path ../goha_demo_data/person_2_test/
```
The `--source_path` points to the source image while the `--target_path` points to the drive video path. Animation results are saved in `--savedir`.
</details>

## Testing on the CelebA dataset
<details>
<summary> CelebA pre-processing </summary>

Follow these [instructions] to process the [CelebA]() dataset. The processed dataset has the structure below, where `images` include cropped portrait image, `matting` include foreground masks predicted by MODNet and `dataset.json` includes camera views for each portrait.
  ```
  - celeba
    - celeba
      - images
      - matting
      - dataset.json
  ```
</details>

<details>
<summary> Testing on CelebA </summary>

To carry out cross-identity animation, run
  ```
  python test_celeba_cross.py --config configs/s2.yml logs/s3/checkpoint825000.ckpt --savedir /raid/results/celeba_cross
  ```
  `test_sample_number` indicates testing image number, the defualt number will run on all image pairs in the CelebA dataset.
</details>

<details>
<summary> Metrics computation </summary>

We use [torch-fidelty](https://github.com/toshas/torch-fidelity) for FID score computation and [this script from NeRFace](https://github.com/gafniguy/4D-Facial-Avatars/blob/main/nerface_code/nerf-pytorch/nerf/metrics.py) for LPIPS, PSNR, SSIM and L1 metrics. We use [ArcFace](https://github.com/ronghuaiyang/arcface-pytorch) to evaluate CSIM, [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch) for AED, APD, and AKD.
</details>

## Training
<details>
<summary> Training dataset preparation </summary>
Please see [here]().
</details>

<details>
<summary> Model training </summary>

To train the model, run:
```
sh train.sh
```
Training logs can be found in `logs/s1`, `logs/s2` or `logs/s3` depending on the training stage and visualized by [Tensorboard](https://www.tensorflow.org/tensorboard). The overall training takes about 1 week on 8 V100 32GB GPUs.
</details>

## Acknowledgement
This work is built on top of [EG3D](https://github.com/NVlabs/eg3d), [GFPGAN](https://github.com/TencentARC/GFPGAN), [Deep3DRecon](https://github.com/sicxu/Deep3DFaceRecon_pytorch) and [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch). We also use [MODNET](https://github.com/ZHKKKe/MODNet) to remove portrait backgrounds.

## Contact
If you have any questions or comments, please feel free to contact xuetingl@nvidia.com
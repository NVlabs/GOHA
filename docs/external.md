# External Modules Preparation
## BFM
  - Follow instructions in [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch#inference-with-a-pre-trained-model) to get the following files and put them into folder named `BFM` under `GOHA/external`:
    ```
    - external
      - BFM
        - 01_MorphableModel.mat
        - BFM_exp_idx.mat
        - BFM_front_idx.mat
        - BFM_model_front.mat
        - Exp_Pca
        - facemodel_info.mat
        - select_vertex_id.mat
        - similarity_Lm3D_all.mat
        - std_exp
    ```
## Deep3DFaceRecon_pytorch
  - Follow instructions in [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch#inference-with-a-pre-trained-model) to get the following files and put them into folder named `Deep3DFaceRecon`  under `GOHA/external`. Note that we used the [new version](https://github.com/sicxu/Deep3DFaceRecon_pytorch#04252023-update) released at 04/25/2023, which can model eye blinking motion better.
  ```
  - external 
    - Deep3DFaceRecon
      - checkpoints
        - face_recon_feat0.2_augment
          - epoch_20.pth
  ```
## face-parsing.PyTorch
  - Follow instructions in [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch#training) to get the following files and put them into folder named `face-parsing` under `GOHA/external`:
  ```
  - external
    - face-parsing
      - 79999_iter.pth
  ```
## GFPGAN (only needed for training)
  - Follow instructions in [GFPGAN](https://github.com/TencentARC/GFPGAN) to get the following files and put them into folder named `GFPGAN` under `GOHA/external`. We use the weights from `v1.3` version.
  ```
  - external
    - GFPGAN
      - GFPGANv1.3.pth
      - GFPGANv1_net_d.pth
  ```
  
  
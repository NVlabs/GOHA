# Dataset Pre-processing
## Testing Data Pre-processing
- We use the same [dataset preprocessing](https://github.com/NVlabs/eg3d#preparing-datasets) as in [EG3D](https://github.com/NVlabs/eg3d) without additional offset. Below we show an example of this process for images in the `/raid/test` folder:
  - Prepare code:
    ```
    git clone https://github.com/NVlabs/eg3d
    cd eg3d/dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch/
    git submodule update --init --recursive
    ```
  - Install packages and download all files for `BFM` and `checkpoints` folder following instructions from [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch/tree/6ba3d22f84bf508f0dde002da8fff277196fef21#installation).
  - Commend out line 48 and 49 in [3dface2idr_mat.py](https://github.com/NVlabs/eg3d/blob/main/dataset_preprocessing/ffhq/3dface2idr_mat.py#L48) to remove additional offset.
  - Run `python preprocess_in_the_wild.py --indir /raid/test/` under `eg3d/dataset_preprocessing/ffhq`.
  - Collect cropped images and predicted camera views by
    ```
    mv /raid/test/crop /raid/test/images
    mv eg3d/dataset_preprocessing/ffhq/test/dataset.json /raid/test/
    ```
  - Predict matting masks using MODNet:
    ```
    git clone https://github.com/ZHKKKe/MODNet
    cd MODNet/
    # download modnet_photographic_portrait_matting.ckpt following MODNet's instruction
    mkdir /raid/test/matting
    rm /raid/test/images/cameras.json
    python -m demo.image_matting.colab.inference --input-path /raid/test/images/ --output-path /raid/test/matting --ckpt-path ./pretrained/modnet_photographic_portrait_matting.ckpt
    ```
  - Now the `/raid/test` folder should include all files needed for test like this:
    ```
    - test
      data
        - images
        - matting
        - dataset.json
    ```
    Note there is a subfolder `data` under `test`.
  - You can run our demo on the processed image as:
    ```
    cd goha/src
    python demo.py --config configs/s2.yml --checkpoint logs/s3/checkpoint825000.ckpt --savedir output --source_path /raid/test --target_path ../goha_demo_data/person_2_test/
    ```

## Training Data Pre-processing
- As described in the paper, we train our model using the [RAVDESS](https://zenodo.org/records/1188976#.YFZuJ0j7SL8), [CelebV-HQ](https://celebv-hq.github.io), and [FFHQ](https://github.com/NVlabs/ffhq-dataset) dataset. You can use similar process discussed above to pre-process these datasets.

- Besides the datasets above, we also use [EG3D](https://github.com/NVlabs/eg3d) to synthesize pairs of images. Each pair includes two views of a single person. The views are randomly sampled from the [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset).

- The pre-processing is similar as the testing data described above except for folder arrangement. After pre-processing, the structure for each dataset looks like below. The `images`, `matting` folder include the cropped images and foreground matting masks predicted by MODNet. The `dataset.json` records camera parameters of each portrait image.
  ```
  - FFHQ
    - 00000
      - images
      - matting
      - dataset.json
    - 000001
    ...
  - RAVDESS
    - Actor_0
      - images
      - matting
      - dataset.json
    - Actor_1
    ...
  - CelebV-HQ
    - -1eKufUP5XQ_3
      - images
      - matting
      - dataset.json
    - -1eKufUP5XQ_4
    ...
  - Synthesized data by EG3D
    - synth
      - images
      - matting
      - dataset.json
  ```
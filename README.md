# ConfFormer

ConfFormer model implementations for MixSTE, MotionBERT, KTPFormer and TCPFormer.

## Installation

```bash
pip install -r requirements.txt
```

## Data preparation

### Official datasets

Download the datasets from their official providers and follow the corresponding backbone project's preprocessing instructions:

| Dataset | Official download page |
| --- | --- |
| Human3.6M | https://vision.imar.ro/human3.6m/description.php |
| HumanEva-I | https://is.mpg.de/en/code/humaneva-datasets-i-and-ii |
| MPI-INF-3DHP | https://vcai.mpi-inf.mpg.de/3dhp-dataset/ |
| 3DPW | https://virtualhumans.mpi-inf.mpg.de/3DPW/ |

### Official 2D detectors

| Detector | Official project | Use in our experiments |
| --- | --- | --- |
| CPN | https://github.com/chenyilun95/tf-cpn | Human3.6M with MixSTE and KTPFormer |
| Stacked Hourglass | https://github.com/princeton-vl/pose-hg-train | Human3.6M with MotionBERT and TCPFormer |
| HRNet | https://github.com/HRNet/HRNet-Human-Pose-Estimation | MPI-INF-3DHP and HumanEva |
| OpenPose | https://github.com/CMU-Perceptual-Computing-Lab/openpose | Official 3DPW `poses2d` detections |

Use each backbone's original detection/preprocessing pipeline and retain the detector's per-joint confidence. The links identify the detector implementations; their raw outputs must be converted to the selected backbone's data format. The models expect `(B, T, J, 3)` tensors containing normalized `(x, y, confidence)`, with the joint order and frame alignment used by that backbone. Only `(x, y)` enters the coordinate embedding; confidence is clipped to `[0,1]` inside the models and used to construct the reliability affinities. Keep confidence during loading and augmentation rather than dropping the third channel. Coordinate normalization belongs to data preprocessing and should not be applied twice.

For 3DPW, use the official sequence pickle files directly. Their `poses2d` field already includes OpenPose coordinates and confidence; rerunning OpenPose is unnecessary. The provided loader converts COCO-18 detections to H36M-17 and prepares root-relative camera-space 3D targets.

### Files for the provided MixSTE training entry point

Set `--root_path` to the prepared data directory. Human3.6M uses:

```text
dataset/
  data_3d_h36m.npz
  data_2d_h36m_cpn_ft_h36m_dbb_with_confidence.npz
  data_2d_h36m_gt.npz
```

The 3D archive contains `positions_3d`. The 2D archives contain a `positions_2d` dictionary indexed by subject and action, with a list of camera arrays, and `metadata` containing `keypoints_symmetry`. Detector arrays must retain `(x, y, confidence)` for each frame and joint. Save 2D coordinates in image pixels: the provided loader performs coordinate normalization. The current loader also reads the GT 2D archive during ordinary detector-confidence runs; its presence does not replace the detector coordinates or scores supplied to the model.

For HumanEva-I, use `--dataset humaneva15` and prepare `data_3d_humaneva15.npz`, `data_2d_humaneva15_gt.npz`, and the HRNet detection archive in the same dictionary format. Select the detection archive with `--keypoints <detector_tag>`: the loader reads `data_2d_humaneva15_<detector_tag>.npz` (for example, `--keypoints detections_with_confidence`). Preserve the `Train/S1`–`Train/S3` and `Validate/S1`–`Validate/S3` subject keys and the joint layout expected by the loader.

For MPI-INF-3DHP, use `--dataset 3dhp` and prepare `data_train_3dhp_hrnet.npz` and `data_test_3dhp_hrnet.npz`. Each archive contains a `data` dictionary with per-sequence/per-camera `data_2d` and `data_3d` arrays; retain the HRNet confidence in the third 2D channel. The loader expects 3D coordinates in millimeters and converts them internally. Use the input resolution matching the detections (`--res_w_3dhp` and `--res_h_3dhp`; defaults are 2048). The test loader handles TS5/TS6 at 1920×1080.

For 3DPW, use `--dataset 3dpw` and point `--root_path` to a directory containing:

```text
sequenceFiles/
  train/*.pkl
  validation/*.pkl
  test/*.pkl
```

The loader also accepts the extracted `3DPW/sequenceFiles/` nesting. For the other three backbones, use their official dataset layouts and preprocessing while retaining the confidence channel.

## MixSTE: training and evaluation

The implementation is `model/mixste_confformer.py`, exposing `Model(args)`. This is the default model of `main_confformer.py`.

Training:

```bash
python main_confformer.py --model mixste_confformer \
  --dataset h36m --root_path /path/to/dataset \
  --batch_size 8 --frames 243 --stride 243 --nepoch 120 \
  --channel 512 --d_hid 1024 --lr 0.00008 --workers 8
```

Evaluation:

```bash
python main_confformer.py --model mixste_confformer --test \
  --dataset h36m --root_path /path/to/dataset \
  --previous_dir /path/to/weights --batch_size 4 --frames 243 --stride 243 \
  --channel 512 --d_hid 1024 --workers 8
```

Original backbone: [MixSTE](https://github.com/JinluZhang1126/MixSTE).

## Using the other backbones in their official projects

Copy the corresponding model file into its official project and change the model import in the existing builder. Use that project's training/evaluation scripts, data pipeline and experimental configurations. Keep its architecture dimensions, optimizer, learning-rate schedule and evaluation protocol; the input-interface adjustments below retain confidence for ConfFormer. The experiments train the lifting backbones from scratch without loading pretrained backbone weights.

### MotionBERT

Official project: https://github.com/Walter0807/MotionBERT

1. Copy `model/motionbert_confformer.py` to `lib/model/motionbert_confformer.py` in MotionBERT.
2. In `lib/utils/learning.py`, replace the original DSTformer import with:

   ```python
   from lib.model.motionbert_confformer import DSTformer
   ```

3. Keep the official model-construction arguments and experiment configuration. The model receives `dim_in=3`; ensure `no_conf=False` so the loader retains `(x, y, confidence)`. The coordinate embedding internally uses only xy.
4. Run training and evaluation using the commands and configuration files provided in the official repository, with pretrained-weight loading disabled for training from scratch.

### KTPFormer

Official project: https://github.com/JihuaPeng/KTPFormer

1. Copy `model/ktpformer_confformer.py` to `common/ktpformer_confformer.py` in KTPFormer.
2. In `run_ktpformer.py`, replace the original model import with:

   ```python
   from common.ktpformer_confformer import KTPFormer
   ```

3. Retain the official adjacency construction, model arguments and experiment settings. Keep `in_chans=2` for the xy embedding, while supplying a three-channel `(x, y, confidence)` input tensor.
4. Use the official training/evaluation commands and settings for the selected dataset and temporal window.

### TCPFormer

Official project: https://github.com/AsukaCamellia/TCPFormer

1. Copy `model/tcpformer_confformer.py` to `model/tcpformer_confformer.py` in TCPFormer. This file includes the full backbone and its required model components.
2. In `utils/learning.py`, replace the original model import with:

   ```python
   from model.tcpformer_confformer import MemoryInducedTransformer
   ```

3. Use the official experimental configuration (for example, `configs/h36m/TCPFormer_h36m_243.yaml`), setting `dim_in: 2` for the xy embedding. The actual input remains `(B, T, J, 3)`: the model separates confidence internally. Keep the remaining architecture and training settings from the official configuration.
4. Use the official `train.py` or `train_3dhp.py` training/evaluation commands with the selected configuration and ConfFormer-trained checkpoint.

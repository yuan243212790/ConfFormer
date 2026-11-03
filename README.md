# ConfFormer

ConfFormer model implementations for MixSTE, MotionBERT, KTPFormer and TCPFormer.

## Installation

```bash
pip install -r requirements.txt
```

## Data preparation

### Official datasets

Download the datasets from their official providers and follow the corresponding backbone project's preprocessing instructions:

| Dataset | Official data page |
| --- | --- |
| Human3.6M | http://vision.imar.ro/human3.6m/description.php |
| HumanEva-I | http://humaneva.is.tue.mpg.de/datasets_human_1 |
| MPI-INF-3DHP | https://vcai.mpi-inf.mpg.de/3dhp-dataset/ |
| 3DPW | https://virtualhumans.mpi-inf.mpg.de/3DPW/evaluation.html |

### Official 2D detectors

| Detector | Official project | Use in our experiments |
| --- | --- | --- |
| CPN | https://github.com/chenyilun95/tf-cpn | Human3.6M with MixSTE and KTPFormer |
| Stacked Hourglass | https://github.com/princeton-vl/pose-hg-train | Human3.6M with MotionBERT and TCPFormer |
| HRNet | https://github.com/HRNet/HRNet-Human-Pose-Estimation | MPI-INF-3DHP and HumanEva |
| OpenPose | https://github.com/CMU-Perceptual-Computing-Lab/openpose | Official 3DPW `poses2d` detections |

## MixSTE: training and evaluation

The implementation is `model/mixste_confformer.py`, exposing `Model(args)`. This is the default model of `main_confformer.py`.

Training:

```bash
python main_confformer.py --model mixste_confformer \
  --dataset h36m --root_path /path/to/dataset \
  --batch_size 4 --frames 243 --stride 243 --nepoch 160 \
  --channel 512 --d_hid 1024 --lr 0.00004 --data_augmentation 1 --workers 8
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

Copy the corresponding model file into its official project and change the model import in the existing builder. Use its training/evaluation scripts and data pipeline, with the following settings from Table 1 of the paper. Train all lifting backbones from scratch.

| Setting | MixSTE | MotionBERT | KTPFormer | TCPFormer |
| --- | --- | --- | --- | --- |
| Transformer blocks | 8 | 5 | 7 | 16 |
| Feature dimension | 512 | 512 | 512 | 128 |
| Training epochs | 160 | 120 | 200 | 90 |
| Batch size | 4 | 4 | 4 | 4 |
| Initial learning rate | 4e-5 | 5e-4 | 4e-5 | 5e-4 |
| Optimizer | AdamW | AdamW | AdamW | AdamW |
| Augmentation | Flip | Flip | Flip | Flip |

### MotionBERT

Official project: https://github.com/Walter0807/MotionBERT

1. Copy `model/motionbert_confformer.py` to `lib/model/motionbert_confformer.py` in MotionBERT.
2. In `lib/utils/learning.py`, replace the original DSTformer import with:

   ```python
   from lib.model.motionbert_confformer import DSTformer
   ```

3. Set the model to 5 blocks and 512 feature dimensions; train for 120 epochs with batch size 4 and initial learning rate `5e-4`. The model receives `dim_in=3`; ensure `no_conf=False` so the loader retains `(x, y, confidence)`. The coordinate embedding internally uses only xy.
4. Run training and evaluation using the commands and configuration files provided in the official repository, with the settings above and pretrained-weight loading disabled.

### KTPFormer

Official project: https://github.com/JihuaPeng/KTPFormer

1. Copy `model/ktpformer_confformer.py` to `common/ktpformer_confformer.py` in KTPFormer.
2. In `run_ktpformer.py`, replace the original model import with:

   ```python
   from common.ktpformer_confformer import KTPFormer
   ```

3. Retain the official adjacency construction. Set 7 blocks and 512 feature dimensions; train for 200 epochs with batch size 4 and initial learning rate `4e-5`. Keep `in_chans=2` for the xy embedding, while supplying a three-channel `(x, y, confidence)` input tensor.
4. Use the official training/evaluation commands with the settings above for the selected dataset and temporal window.

### TCPFormer

Official project: https://github.com/AsukaCamellia/TCPFormer

1. Copy `model/tcpformer_confformer.py` to `model/tcpformer_confformer.py` in TCPFormer. This file includes the full backbone and its required model components.
2. In `utils/learning.py`, replace the original model import with:

   ```python
   from model.tcpformer_confformer import MemoryInducedTransformer
   ```

3. Use the official experimental configuration (for example, `configs/h36m/TCPFormer_h36m_243.yaml`), setting `dim_in: 2` for the xy embedding. The actual input remains `(B, T, J, 3)`: the model separates confidence internally. Set 16 blocks and 128 feature dimensions; train for 90 epochs with batch size 4 and initial learning rate `5e-4`.
4. Use the official `train.py` or `train_3dhp.py` training/evaluation commands with the settings above and a ConfFormer-trained checkpoint for evaluation.

## References

Please also cite the corresponding backbone papers when using their ConfFormer variants:

- Zhang et al. [MixSTE: Seq2seq Mixed Spatio-Temporal Encoder for 3D Human Pose Estimation in Video](https://github.com/JinluZhang1126/MixSTE), CVPR 2022.
- Zhu et al. [MotionBERT: A Unified Perspective on Learning Human Motion Representations](https://github.com/Walter0807/MotionBERT), ICCV 2023.
- Peng et al. [KTPFormer: Kinematics and Trajectory Prior Knowledge-Enhanced Transformer for 3D Human Pose Estimation](https://github.com/JihuaPeng/KTPFormer), CVPR 2024.
- Liu et al. [TCPFormer: Learning Temporal Correlation with Implicit Pose Proxy for 3D Human Pose Estimation](https://github.com/AsukaCamellia/TCPFormer), AAAI 2025.

## Acknowledgements

We thank the authors of [MixSTE](https://github.com/JinluZhang1126/MixSTE), [MotionBERT](https://github.com/Walter0807/MotionBERT), [KTPFormer](https://github.com/JihuaPeng/KTPFormer), and [TCPFormer](https://github.com/AsukaCamellia/TCPFormer) for releasing their backbone implementations. Our model variants build on these projects. We also acknowledge [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) and [3D Pose Baseline](https://github.com/una-dinosauria/3d-pose-baseline) for foundational code used in the data-processing and evaluation pipeline.

We thank the dataset providers and the authors of CPN, Stacked Hourglass, HRNet, and OpenPose, linked above, for making their data and detector implementations available. Original copyright notices and license terms apply to the respective upstream components.

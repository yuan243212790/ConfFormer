import glob
import os
import pickle

import numpy as np

from common.camera import normalize_screen_coordinates
from common.mocap_dataset import MocapDataset
from common.skeleton import Skeleton


pw3d_skeleton = Skeleton(
    parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
    joints_left=[4, 5, 6, 11, 12, 13],
    joints_right=[1, 2, 3, 14, 15, 16],
)


class PW3DDataset(MocapDataset):
    """3DPW/PW3D loader compatible with the project's Fusion dataloader.

    The raw 3DPW files provide SMPL-24 3D joints in world coordinates, OpenPose
    COCO-18 2D detections, per-frame camera extrinsics and intrinsics. This
    class follows the common pose-lifting preprocessing style: transform 3D
    joints to camera coordinates, convert both 2D and 3D to a H36M-like 17-joint
    layout, keep 2D confidence, and split tracks on invalid camera-pose frames.
    """

    def __init__(self, root_path, opt=None, splits=('train', 'validation', 'test')):
        super().__init__(fps=30, skeleton=pw3d_skeleton)

        self.root_path = root_path
        self.opt = opt
        self._data = {}
        self._keypoints_2d = {}
        self._cameras = {}
        self._valid_frames = {}

        sequence_root = self._resolve_sequence_root(root_path)
        self.sequence_root = sequence_root

        loaded = 0
        for split in splits:
            split_dir = os.path.join(sequence_root, split)
            if not os.path.isdir(split_dir):
                continue
            loaded += self._load_split(split, split_dir)

        if loaded == 0:
            raise FileNotFoundError(
                'No 3DPW pickle files found under {}. Expected a directory such '
                'as 3DPW/sequenceFiles/sequenceFiles/{{train,validation,test}}.'
                .format(sequence_root)
            )

        self._num_tracks = loaded
        print('Loaded {} 3DPW person tracks from {}'.format(loaded, sequence_root))

    @staticmethod
    def _resolve_sequence_root(root_path):
        candidates = [
            os.path.join(root_path, '3DPW', 'sequenceFiles', 'sequenceFiles'),
            os.path.join(root_path, '3DPW', 'sequenceFiles'),
            os.path.join(root_path, 'sequenceFiles', 'sequenceFiles'),
            os.path.join(root_path, 'sequenceFiles'),
            root_path,
        ]
        for candidate in candidates:
            if os.path.isdir(candidate) and (
                os.path.isdir(os.path.join(candidate, 'train')) or
                os.path.isdir(os.path.join(candidate, 'test'))
            ):
                return candidate
        return candidates[0]

    def _load_split(self, split, split_dir):
        self._data.setdefault(split, {})
        self._keypoints_2d.setdefault(split, {})
        self._valid_frames.setdefault(split, {})
        self._cameras.setdefault(split, {})

        loaded = 0
        for path in sorted(glob.glob(os.path.join(split_dir, '*.pkl'))):
            with open(path, 'rb') as f:
                seq = pickle.load(f, encoding='latin1')
            loaded += self._process_sequence(split, path, seq)
        return loaded

    def _process_sequence(self, split, path, seq):
        seq_name = str(np.asarray(seq.get('sequence', os.path.splitext(os.path.basename(path))[0])).item())
        intrinsics = np.asarray(seq['cam_intrinsics'], dtype='float32')
        cam_poses = np.asarray(seq['cam_poses'], dtype='float32')
        res_w, res_h = self._resolution_from_intrinsics(intrinsics)

        num_people = len(seq['jointPositions'])
        loaded = 0
        for person_idx in range(num_people):
            joints24_world = np.asarray(seq['jointPositions'][person_idx], dtype='float32').reshape(-1, 24, 3)
            poses2d_coco = np.asarray(seq['poses2d'][person_idx], dtype='float32').transpose(0, 2, 1)
            valid = np.asarray(seq['campose_valid'][person_idx]).astype(bool)

            n_frames = min(joints24_world.shape[0], poses2d_coco.shape[0], cam_poses.shape[0], valid.shape[0])
            joints24_world = joints24_world[:n_frames]
            poses2d_coco = poses2d_coco[:n_frames]
            cam_poses_i = cam_poses[:n_frames]
            valid = valid[:n_frames]

            for seg_idx, frame_idx in enumerate(self._valid_segments(valid)):
                if frame_idx.size == 0:
                    continue

                joints_cam = self._world_to_camera(joints24_world[frame_idx], cam_poses_i[frame_idx])
                joints17 = self._smpl24_to_h36m17(joints_cam).astype('float32')
                keypoints17 = self._coco18_to_h36m17(poses2d_coco[frame_idx]).astype('float32')

                action = seq_name
                cam = self._camera_params(intrinsics, res_w, res_h, seq_name, person_idx, seg_idx)

                if action not in self._data[split]:
                    self._data[split][action] = {
                        'positions': [],
                        'cameras': [],
                    }
                    self._keypoints_2d[split][action] = []
                    self._valid_frames[split][action] = []
                    self._cameras[split][action] = []

                self._data[split][action]['positions'].append(joints17)
                self._data[split][action]['cameras'].append(cam)
                self._keypoints_2d[split][action].append(keypoints17)
                self._valid_frames[split][action].append(np.ones(joints17.shape[0], dtype=bool))
                self._cameras[split][action].append(cam)
                loaded += 1

        return loaded

    @staticmethod
    def _resolution_from_intrinsics(intrinsics):
        cx = float(intrinsics[0, 2])
        cy = float(intrinsics[1, 2])
        res_w = int(round(cx * 2)) if cx > 0 else 1920
        res_h = int(round(cy * 2)) if cy > 0 else 1080
        return res_w, res_h

    @staticmethod
    def _valid_segments(valid):
        valid_idx = np.where(valid)[0]
        if valid_idx.size == 0:
            return []

        breaks = np.where(np.diff(valid_idx) > 1)[0] + 1
        return [seg for seg in np.split(valid_idx, breaks) if seg.size > 0]

    @staticmethod
    def _world_to_camera(joints_world, cam_poses):
        rotation = cam_poses[:, :3, :3]
        translation = cam_poses[:, :3, 3]
        return np.einsum('tij,tkj->tki', rotation, joints_world) + translation[:, None, :]

    @staticmethod
    def _smpl24_to_h36m17(joints):
        out = np.zeros((joints.shape[0], 17, 3), dtype=joints.dtype)
        out[:, 0] = joints[:, 0]                  # pelvis
        out[:, 1] = joints[:, 2]                  # right hip
        out[:, 2] = joints[:, 5]                  # right knee
        out[:, 3] = joints[:, 8]                  # right ankle
        out[:, 4] = joints[:, 1]                  # left hip
        out[:, 5] = joints[:, 4]                  # left knee
        out[:, 6] = joints[:, 7]                  # left ankle
        out[:, 7] = (joints[:, 0] + joints[:, 12]) * 0.5
        out[:, 8] = joints[:, 12]                 # neck
        out[:, 9] = joints[:, 15]                 # SMPL has no nose joint
        out[:, 10] = joints[:, 15]                # head
        out[:, 11] = joints[:, 16]                # left shoulder
        out[:, 12] = joints[:, 18]                # left elbow
        out[:, 13] = joints[:, 20]                # left wrist
        out[:, 14] = joints[:, 17]                # right shoulder
        out[:, 15] = joints[:, 19]                # right elbow
        out[:, 16] = joints[:, 21]                # right wrist
        return out

    @staticmethod
    def _coco18_to_h36m17(keypoints):
        out = np.zeros((keypoints.shape[0], 17, 3), dtype=keypoints.dtype)

        out[:, 0] = PW3DDataset._combine_2d(keypoints, [8, 11])      # pelvis
        out[:, 1] = keypoints[:, 8]                                  # right hip
        out[:, 2] = keypoints[:, 9]                                  # right knee
        out[:, 3] = keypoints[:, 10]                                 # right ankle
        out[:, 4] = keypoints[:, 11]                                 # left hip
        out[:, 5] = keypoints[:, 12]                                 # left knee
        out[:, 6] = keypoints[:, 13]                                 # left ankle
        out[:, 8] = keypoints[:, 1]                                  # neck
        out[:, 7] = PW3DDataset._combine_2d(out, [0, 8])             # spine
        out[:, 9] = keypoints[:, 0]                                  # nose
        out[:, 10] = PW3DDataset._combine_2d(keypoints, [14, 15, 16, 17, 0])
        out[:, 11] = keypoints[:, 5]                                 # left shoulder
        out[:, 12] = keypoints[:, 6]                                 # left elbow
        out[:, 13] = keypoints[:, 7]                                 # left wrist
        out[:, 14] = keypoints[:, 2]                                 # right shoulder
        out[:, 15] = keypoints[:, 3]                                 # right elbow
        out[:, 16] = keypoints[:, 4]                                 # right wrist
        return out

    @staticmethod
    def _combine_2d(keypoints, indices):
        coords = keypoints[:, indices, :2]
        conf = keypoints[:, indices, 2:3]
        weights = np.where(conf > 0, conf, 0)
        denom = weights.sum(axis=1)

        plain_mean = coords.mean(axis=1)
        weighted = (coords * weights).sum(axis=1)
        combined_xy = np.divide(
            weighted,
            np.maximum(denom, 1e-6),
            out=plain_mean.copy(),
            where=denom > 1e-6,
        )

        positive = conf[..., 0] > 0
        combined_conf = np.zeros((keypoints.shape[0], 1), dtype=keypoints.dtype)
        for frame in range(keypoints.shape[0]):
            vals = conf[frame, positive[frame], 0]
            if vals.size > 0:
                combined_conf[frame, 0] = vals.min()

        return np.concatenate((combined_xy, combined_conf), axis=1)

    @staticmethod
    def _camera_params(intrinsics, res_w, res_h, seq_name, person_idx, segment_idx):
        center = intrinsics[:2, 2].astype('float32')
        focal = np.array([intrinsics[0, 0], intrinsics[1, 1]], dtype='float32')
        center_norm = normalize_screen_coordinates(center, w=res_w, h=res_h).astype('float32')
        focal_norm = focal / float(res_w) * 2.0

        radial = np.zeros(3, dtype='float32')
        tangential = np.zeros(2, dtype='float32')
        intrinsic = np.concatenate((focal_norm, center_norm, radial, tangential)).astype('float32')

        return {
            'id': '{}_p{}_s{}'.format(seq_name, person_idx, segment_idx),
            'res_w': res_w,
            'res_h': res_h,
            'intrinsic': intrinsic,
        }

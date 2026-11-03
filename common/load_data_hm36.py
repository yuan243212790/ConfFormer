import sys
import torch.utils.data as data

from common.camera import *
from common.utils import deterministic_random


class Fusion(data.Dataset):
    def __init__(self, opt, dataset, root_path, train=True):
        if opt.stride == 1:
            from common.generator import ChunkedGenerator
        else:
            from common.generator_stride import ChunkedGenerator

        self.opt = opt
        self.data_type = opt.dataset
        self.train = train
        self.keypoints_name = opt.keypoints
        self.root_path = root_path

        self.train_list = opt.subjects_train.split(',')
        self.test_list = opt.subjects_test.split(',')
        self.action_filter = None if opt.actions == '*' else opt.actions.split(',')
        self.downsample = opt.downsample
        self.subset = opt.subset
        self.stride = opt.stride
        self.crop_uv = opt.crop_uv
        self.test_aug = opt.test_augmentation
        self.pad = opt.pad

        if self.data_type in ['3dhp', '3dpw', 'pw3d']:
            test_stride = 1
        elif self.stride != opt.frames and self.stride != 1:
            test_stride = opt.frames
        else:
            test_stride = self.stride

        if self.train:
            self.keypoints, self.keypoints_GT = self.prepare_data(dataset, self.train_list)
            self.cameras_train, self.poses_train, self.poses_train_2d, self.poses_train_2d_GT, _ = \
                    self.fetch(dataset, self.train_list, subset=self.subset)
            
            if self.data_type == '3dhp':
                from common.generator_3dhp import ChunkedGenerator as ChunkedGenerator3DHP
                self.generator = ChunkedGenerator3DHP(opt.batch_size, self.cameras_train, self.poses_train,
                                                  self.poses_train_2d, None,
                                                  self.stride, pad=self.pad,
                                                  augment=opt.data_augmentation,
                                                  reverse_aug=opt.reverse_augmentation,
                                                  kps_left=self.kps_left, kps_right=self.kps_right,
                                                  joints_left=self.joints_left,
                                                  joints_right=self.joints_right, out_all=opt.out_all, train=True)
            else:
                self.generator = ChunkedGenerator(opt.batch_size, self.cameras_train, self.poses_train,
                                                  self.poses_train_2d, self.poses_train_2d_GT,
                                                  self.stride, pad=self.pad,
                                                  augment=opt.data_augmentation,
                                                  reverse_aug=opt.reverse_augmentation,
                                                  kps_left=self.kps_left, kps_right=self.kps_right,
                                                  joints_left=self.joints_left,
                                                  joints_right=self.joints_right, out_all=opt.out_all)
            print('Training on {} samples ({} valid frames)'.format(
                len(self.generator.pairs), self._num_pose_frames(self.poses_train)))
        else:
            self.keypoints, self.keypoints_GT = self.prepare_data(dataset, self.test_list)
            self.cameras_test, self.poses_test, self.poses_test_2d, self.poses_test_2d_GT, self.valid_frame_test = \
                                        self.fetch(dataset, self.test_list, subset=self.subset)
            
            if self.data_type in ['3dhp', '3dpw', 'pw3d']:
                from common.generator_3dhp import ChunkedGenerator as ChunkedGenerator3DHP
                self.generator = ChunkedGenerator3DHP(opt.batch_size, self.cameras_test, self.poses_test,
                                                  self.poses_test_2d, self.valid_frame_test, test_stride,
                                                  pad=self.pad, augment=False, kps_left=self.kps_left,
                                                  kps_right=self.kps_right, joints_left=self.joints_left,
                                                  joints_right=self.joints_right, out_all=opt.out_all, train=False,
                                                  eval_stride=getattr(opt, 'eval_stride', 1))
            else:
                self.generator = ChunkedGenerator(opt.batch_size, self.cameras_test, self.poses_test,
                                                  self.poses_test_2d, self.poses_test_2d_GT, test_stride,
                                                  pad=self.pad, augment=False, kps_left=self.kps_left,
                                                  kps_right=self.kps_right, joints_left=self.joints_left,
                                                  joints_right=self.joints_right, out_all=opt.out_all)
            self.key_index = self.generator.saved_index
            print('Testing on {} samples ({} valid frames)'.format(
                len(self.generator.pairs), self._num_pose_frames(self.poses_test)))

    @staticmethod
    def _num_pose_frames(poses):
        if poses is None:
            return 0
        return sum(pose.shape[0] for pose in poses.values())

    def prepare_data(self, dataset, folder_list):
        # MPI-INF-3DHP / 3DPW: 2D/3D are integrated in dataset._keypoints_2d.
        if self.data_type in ['3dhp', '3dpw', 'pw3d']:
            keypoints = {}
            keypoints_GT = {}
            for subject in folder_list:
                keypoints[subject] = {}
                keypoints_GT[subject] = {}
                for action, cams in dataset._keypoints_2d.get(subject, {}).items():
                    keypoints[subject][action] = [cam_kps.copy() for cam_kps in cams]
                    keypoints_GT[subject][action] = [cam_kps.copy() for cam_kps in cams]

            # Use skeleton symmetry for both keypoints and joints
            self.kps_left, self.kps_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
            self.joints_left, self.joints_right = self.kps_left, self.kps_right

            for subject in folder_list:
                for action in dataset._data.get(subject, {}).keys():
                    anim = dataset._data[subject][action]

                    # Build positions_3d list and convert to root-relative.
                    positions_3d = []
                    for cam_idx, pos_3d in enumerate(anim['positions']):
                        pos_3d = pos_3d.copy()
                        root = pos_3d[:, self.opt.root_joint:self.opt.root_joint + 1, :]
                        if self.data_type == '3dhp':
                            pos_3d[:, :self.opt.root_joint] -= root
                            pos_3d[:, self.opt.root_joint + 1:] -= root
                        else:
                            pos_3d -= root
                        positions_3d.append(pos_3d)
                    anim['positions_3d'] = positions_3d

                    for cam_idx in range(len(keypoints[subject][action])):
                        kps = keypoints[subject][action][cam_idx]
                        if self.data_type in ['3dpw', 'pw3d'] and self.crop_uv == 0:
                            cam = anim['cameras'][cam_idx]
                            kps = kps.copy()
                            kps[..., :2] = normalize_screen_coordinates(
                                kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                        keypoints[subject][action][cam_idx] = kps
                        keypoints_GT[subject][action][cam_idx] = kps

                    # Align lengths
                    for cam_idx in range(len(keypoints[subject][action])):
                        mocap_length = anim['positions_3d'][cam_idx].shape[0]
                        if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                            keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]
                            keypoints_GT[subject][action][cam_idx] = keypoints_GT[subject][action][cam_idx][:mocap_length]

            return keypoints, keypoints_GT

        # H36M / HumanEva: 3D in positions, 2D in separate npz
        for subject in folder_list:
            for action in dataset[subject].keys():
                anim = dataset[subject][action]

                positions_3d = []
                for cam in anim['cameras']:
                    pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                    pos_3d[:, 1:] -= pos_3d[:, :1]  

                    if self.keypoints_name.startswith('sh') or self.keypoints_name.startswith('hr'):
                        pos_3d = np.delete(pos_3d,obj=9,axis=1)

                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d

        if self.keypoints_name.startswith(self.data_type):
            kps_file = 'data_2d_' + self.keypoints_name + '.npz'
        else:
            kps_file = 'data_2d_' + self.data_type + '_' + self.keypoints_name + '.npz'
            
        kps_gt_file = 'data_2d_' + self.data_type + '_gt.npz'

        keypoints = np.load(self.root_path + kps_file, allow_pickle=True)
        keypoints_GT = np.load(self.root_path + kps_gt_file, allow_pickle=True)
        
        keypoints_symmetry = keypoints['metadata'].item()['keypoints_symmetry']

        self.kps_left, self.kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
        self.joints_left, self.joints_right = list(dataset.skeleton().joints_left()), list(
            dataset.skeleton().joints_right())

        keypoints = keypoints['positions_2d'].item()
        keypoints_GT = keypoints_GT['positions_2d'].item()

        for subject in folder_list:
            assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
            for action in dataset[subject].keys():
                assert action in keypoints[
                    subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action,
                                                                                                         subject)
                for cam_idx in range(len(keypoints[subject][action])):
                    mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                    assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

                    if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                        keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]
                        keypoints_GT[subject][action][cam_idx] = keypoints_GT[subject][action][cam_idx][:mocap_length]

        for subject in folder_list:
            for action in keypoints_GT[subject]:
                for cam_idx, item in enumerate(keypoints_GT[subject][action]):
                    kps = keypoints[subject][action][cam_idx]
                    kps_GT = keypoints_GT[subject][action][cam_idx]

                    cam = dataset.cameras()[subject][cam_idx]

                    if self.crop_uv == 0:
                            kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                            kps_GT[..., :2] = normalize_screen_coordinates(kps_GT[..., :2], w=cam['res_w'], h=cam['res_h'])
                      
                    keypoints[subject][action][cam_idx] = kps
                    keypoints_GT[subject][action][cam_idx] = kps_GT

        return keypoints, keypoints_GT

    def fetch(self, dataset, subjects, subset=1, parse_3d_poses=True):
        out_poses_3d = {}
        out_poses_2d = {}
        out_poses_2d_GT = {}
        out_camera_params = {}
        out_valid_frames = {}

        for subject in subjects:
            for action in self.keypoints[subject].keys():
                if self.action_filter is not None:
                    found = False
                    for a in self.action_filter:
                        if action.startswith(a):
                            found = True
                            break
                    if not found:
                        continue

                poses_2d = self.keypoints[subject][action]
                poses_2d_GT = self.keypoints_GT[subject][action]

                for i in range(len(poses_2d)):
                    out_poses_2d[(subject, action, i)] = poses_2d[i]
                    out_poses_2d_GT[(subject, action, i)] = poses_2d_GT[i]

                # Extract valid_frames if available (MotionAGFormer alignment)
                if hasattr(dataset, '_valid_frames') and subject in dataset._valid_frames and action in dataset._valid_frames[subject]:
                    vfs = dataset._valid_frames[subject][action]
                    if vfs is not None:
                        for i in range(len(vfs)):
                            out_valid_frames[(subject, action, i)] = vfs[i]

                # Try to get cameras from action item first (more reliable for 3DHP)
                anim_item = dataset[subject][action]
                if 'cameras' in anim_item and anim_item['cameras'] is not None:
                    cams = anim_item['cameras']
                    for i, cam in enumerate(cams):
                        if 'intrinsic' in cam:
                            out_camera_params[(subject, action, i)] = cam['intrinsic']
                else:
                    cams_by_subject = dataset.cameras() if dataset.cameras() is not None else None
                    if cams_by_subject is not None and subject in cams_by_subject:
                        cams = cams_by_subject[subject]
                        assert len(cams) == len(poses_2d), 'Camera count mismatch'
                        for i, cam in enumerate(cams):
                            if 'intrinsic' in cam:
                                out_camera_params[(subject, action, i)] = cam['intrinsic']

                if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                    poses_3d = dataset[subject][action]['positions_3d']
                    assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                    for i in range(len(poses_3d)):
                        out_poses_3d[(subject, action, i)] = poses_3d[i]


        if len(out_camera_params) == 0:
            out_camera_params = None
        if len(out_poses_3d) == 0:
            out_poses_3d = None
        if len(out_valid_frames) == 0:
            out_valid_frames = None

        stride = self.downsample
        if subset < 1:
            for key in out_poses_2d.keys():
                n_frames = int(round(len(out_poses_2d[key]) // stride * subset) * stride)
                start = deterministic_random(0, len(out_poses_2d[key]) - n_frames + 1, str(len(out_poses_2d[key])))
                out_poses_2d[key] = out_poses_2d[key][start:start + n_frames:stride]
                out_poses_2d_GT[key] = out_poses_2d_GT[key][start:start + n_frames:stride]

                if out_poses_3d is not None:
                    out_poses_3d[key] = out_poses_3d[key][start:start + n_frames:stride]
                
                if out_valid_frames is not None and key in out_valid_frames:
                    out_valid_frames[key] = out_valid_frames[key][start:start + n_frames:stride]
        elif stride > 1:
            for key in out_poses_2d.keys():
                out_poses_2d[key] = out_poses_2d[key][::stride]
                out_poses_2d_GT[key] = out_poses_2d_GT[key][::stride]

                if out_poses_3d is not None:
                    out_poses_3d[key] = out_poses_3d[key][::stride]
                
                if out_valid_frames is not None and key in out_valid_frames:
                    out_valid_frames[key] = out_valid_frames[key][::stride]

        return out_camera_params, out_poses_3d, out_poses_2d, out_poses_2d_GT, out_valid_frames

    def __len__(self):
        return len(self.generator.pairs)

    def __getitem__(self, index):
        seq_name, start_3d, end_3d, flip, reverse = \
            self.generator.pairs[index], self.generator.bounds_1[index], self.generator.bounds_2[index], \
            self.generator.augment_vectors[index], self.generator.reverse_augment_vectors[index]

        cam, gt_3D, input_2D, input_2D_GT, action, subject, cam_ind = self.generator.get_batch(seq_name, start_3d, end_3d, flip, reverse)
        
        if self.train == False and self.test_aug:
            _, _, input_2D_aug, input_2D_aug_GT, _, _,_ = self.generator.get_batch(seq_name, start_3d, end_3d, flip=True, reverse=reverse)
            input_2D = np.concatenate((np.expand_dims(input_2D,axis=0),np.expand_dims(input_2D_aug,axis=0)),0)
            input_2D_GT = np.concatenate((np.expand_dims(input_2D_GT,axis=0),np.expand_dims(input_2D_aug_GT,axis=0)),0)
            
        input_2D_update = input_2D
        input_2D_update_GT = input_2D_GT

        return cam, gt_3D, input_2D_update, input_2D_update_GT, action, subject, cam_ind

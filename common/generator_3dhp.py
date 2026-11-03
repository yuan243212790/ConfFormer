import numpy as np


class ChunkedGenerator:
    """Generator for MPI-INF-3DHP aligned with MotionAGFormer.

    Key differences vs. the default generator:
    - In test mode, it uses valid_frame[seq_name] to select valid bounds.
    - It supports the train key format (subject, seq, cam_index) and test key format (seq_name,).
    """

    def __init__(
        self,
        batch_size,
        cameras,
        poses_3d,
        poses_2d,
        valid_frame,
        chunk_length=1,
        pad=0,
        causal_shift=0,
        shuffle=False,
        random_seed=1234,
        augment=False,
        reverse_aug=False,
        kps_left=None,
        kps_right=None,
        joints_left=None,
        joints_right=None,
        endless=False,
        out_all=False,
        train=True,
        eval_stride=1,
    ):
        assert poses_3d is None or len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d))
        assert cameras is None or len(cameras) == len(poses_2d)

        self.pairs = []
        self.bounds_1 = []
        self.bounds_2 = []
        self.augment_vectors = []
        self.reverse_augment_vectors = []
        
        self.saved_index = {}
        start_index = 0
        self.pad = pad
        self.causal_shift = causal_shift

        # Ensure key is always a tuple for hashing
        def to_tuple(k):
            if isinstance(k, (list, np.ndarray)):
                return tuple(k)
            return k

        if train:
            for key in poses_2d.keys():
                assert poses_3d is None or poses_2d[key].shape[0] == poses_3d[key].shape[0]
                n_chunks = (poses_2d[key].shape[0] + chunk_length - 1) // chunk_length
                offset = (n_chunks * chunk_length - poses_2d[key].shape[0]) // 2
                bounds = np.arange(n_chunks + 1) * chunk_length - offset
                
                # Base pairs
                tuple_key = to_tuple(key)
                for i in range(len(bounds) - 1):
                    self.pairs.append(tuple_key)
                    self.bounds_1.append(bounds[i])
                    self.bounds_2.append(bounds[i+1])
                    self.augment_vectors.append(False)
                    self.reverse_augment_vectors.append(False)

                    if reverse_aug:
                        self.pairs.append(tuple_key)
                        self.bounds_1.append(bounds[i])
                        self.bounds_2.append(bounds[i+1])
                        self.augment_vectors.append(False)
                        self.reverse_augment_vectors.append(True)
                    
                    if augment:
                        self.pairs.append(tuple_key)
                        self.bounds_1.append(bounds[i])
                        self.bounds_2.append(bounds[i+1])
                        self.augment_vectors.append(True)
                        self.reverse_augment_vectors.append(False)
                        if reverse_aug:
                            self.pairs.append(tuple_key)
                            self.bounds_1.append(bounds[i])
                            self.bounds_2.append(bounds[i+1])
                            self.augment_vectors.append(True)
                            self.reverse_augment_vectors.append(True)

                end_index = start_index + poses_2d[key].shape[0]
                self.saved_index[key] = [start_index, end_index]
                start_index = end_index
        else:
            for key in poses_2d.keys():
                assert poses_3d is None or poses_2d[key].shape[0] == poses_3d[key].shape[0]
                n_chunks = (poses_2d[key].shape[0] + chunk_length - 1) // chunk_length
                offset = (n_chunks * chunk_length - poses_2d[key].shape[0]) // 2

                bounds = np.arange(n_chunks) * chunk_length - offset
                
                # TCPFormer style: Direct masking with valid_frame
                # In test mode, chunk_length is 1, so bounds is just [0, 1, ..., F-1]
                full_vf = valid_frame[key]
                if isinstance(full_vf, list):
                    full_vf = np.array(full_vf)
                full_vf = full_vf.flatten().astype(bool)
                
                # Ensure full_vf and bounds match length
                if full_vf.shape[0] > bounds.shape[0]:
                    full_vf = full_vf[:bounds.shape[0]]
                elif bounds.shape[0] > full_vf.shape[0]:
                    bounds = bounds[:full_vf.shape[0]]
                
                bounds_filtered = bounds[full_vf]
                eval_stride = max(1, int(eval_stride))
                if eval_stride > 1:
                    bounds_filtered = bounds_filtered[::eval_stride]
                tuple_key = to_tuple(key)

                for start_f in bounds_filtered:
                    end_f = start_f + 1
                    self.pairs.append(tuple_key)
                    self.bounds_1.append(start_f)
                    self.bounds_2.append(end_f)
                    self.augment_vectors.append(False)
                    self.reverse_augment_vectors.append(False)

                    if reverse_aug:
                        self.pairs.append(tuple_key)
                        self.bounds_1.append(start_f)
                        self.bounds_2.append(end_f)
                        self.augment_vectors.append(False)
                        self.reverse_augment_vectors.append(True)
                    
                    if augment:
                        self.pairs.append(tuple_key)
                        self.bounds_1.append(start_f)
                        self.bounds_2.append(end_f)
                        self.augment_vectors.append(True)
                        self.reverse_augment_vectors.append(False)
                        if reverse_aug:
                            self.pairs.append(tuple_key)
                            self.bounds_1.append(start_f)
                            self.bounds_2.append(end_f)
                            self.augment_vectors.append(True)
                            self.reverse_augment_vectors.append(True)

                end_index = start_index + poses_2d[key].shape[0]
                self.saved_index[key] = [start_index, end_index]
                start_index = end_index

        self.num_batches = (len(self.pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None

        self.cameras = cameras
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d

        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        self.out_all = out_all

        self.valid_frame = valid_frame
        self.train = train

    def num_frames(self):
        return self.num_batches * self.batch_size

    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                pairs_idx = self.random.permutation(len(self.pairs))
            else:
                pairs_idx = np.arange(len(self.pairs))
            return 0, pairs_idx
        else:
            return self.state

    def get_batch(self, seq_i, start_3d, end_3d, flip, reverse):
        # seq_i is already a tuple key from self.pairs
        seq_name = seq_i

        start_2d = start_3d - self.pad - self.causal_shift
        end_2d = end_3d + self.pad - self.causal_shift

        seq_2d = self.poses_2d[seq_name].copy()
        low_2d = max(start_2d, 0)
        high_2d = min(end_2d, seq_2d.shape[0])
        pad_left_2d = low_2d - start_2d
        pad_right_2d = end_2d - high_2d
        if pad_left_2d != 0 or pad_right_2d != 0:
            batch_2d = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)), 'edge')
        else:
            batch_2d = seq_2d[low_2d:high_2d]

        if flip:
            batch_2d[:, :, 0] *= -1
            batch_2d[:, self.kps_left + self.kps_right] = batch_2d[:, self.kps_right + self.kps_left]
        if reverse:
            batch_2d = batch_2d[::-1].copy()

        # Center-crop padded 2D window to target window size (2*pad + 1) to match ConfFormer temporal embedding
        target_len = 2 * self.pad + 1
        if batch_2d.shape[0] > target_len:
            crop_pad = (batch_2d.shape[0] - target_len) // 2
            batch_2d = batch_2d[crop_pad:crop_pad + target_len]
        
        batch_3d = None
        if self.poses_3d is not None:
            seq_3d = self.poses_3d[seq_name].copy()
            if self.out_all:
                low_3d = low_2d
                high_3d = high_2d
                pad_left_3d = pad_left_2d
                pad_right_3d = pad_right_2d
            else:
                low_3d = max(start_3d, 0)
                high_3d = min(end_3d, seq_3d.shape[0])
                pad_left_3d = low_3d - start_3d
                pad_right_3d = end_3d - high_3d

            if pad_left_3d != 0 or pad_right_3d != 0:
                batch_3d = np.pad(seq_3d[low_3d:high_3d], ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
            else:
                batch_3d = seq_3d[low_3d:high_3d]

            if flip:
                batch_3d[:, :, 0] *= -1
                batch_3d[:, self.joints_left + self.joints_right] = batch_3d[:, self.joints_right + self.joints_left]
            if reverse:
                batch_3d = batch_3d[::-1].copy()

            # Center-crop padded 3D window to target window size
            if batch_3d.shape[0] > target_len:
                crop_pad = (batch_3d.shape[0] - target_len) // 2
                batch_3d = batch_3d[crop_pad:crop_pad + target_len]

        batch_cam = None
        if self.cameras is not None:
            batch_cam = self.cameras[seq_name].copy()
            if flip:
                batch_cam[2] *= -1
                batch_cam[7] *= -1

        if self.train:
            subject, action, cam_index = seq_name
            if batch_3d is None and batch_cam is None:
                return None, None, batch_2d, batch_2d.copy(), action, subject, int(cam_index)
            elif batch_3d is not None and batch_cam is None:
                return np.zeros(9), batch_3d, batch_2d, batch_2d.copy(), action, subject, int(cam_index)
            elif batch_3d is None:
                return batch_cam, None, batch_2d, batch_2d.copy(), action, subject, int(cam_index)
            else:
                return batch_cam, batch_3d, batch_2d, batch_2d.copy(), action, subject, int(cam_index)
        else:
            # Test mode: return consistent number of values (7)
            # seq_name is (subject, action, cam_idx)
            subject, action, cam_index = seq_name
            # cam, gt_3D, input_2D, input_2D_GT, action, subject, cam_ind
            return np.zeros(9), batch_3d, batch_2d, batch_2d.copy(), action, subject, int(cam_index)

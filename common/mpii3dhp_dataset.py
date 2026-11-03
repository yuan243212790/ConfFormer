import numpy as np
from common.mocap_dataset import MocapDataset
from common.skeleton import Skeleton

# 保持你当前成功的骨架定义（训练集已验证通过）
mpii3dhp_skeleton = Skeleton(
    parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
    joints_left=[4, 5, 6, 11, 12, 13],
    joints_right=[1, 2, 3, 14, 15, 16],
)

class MPII3DHPDataset(MocapDataset):
    def __init__(self, path, res_w=2048, res_h=2048):
        super().__init__(fps=25, skeleton=mpii3dhp_skeleton)
        self._data = {}
        self._keypoints_2d = {}
        self._cameras = {}
        self._valid_frames = {}
        self.res_w = int(res_w)
        self.res_h = int(res_h)
        print(f'Loading MPII3DHP dataset from {path}...')
        raw_data = np.load(path, allow_pickle=True)['data'].item()
        self._load_train_data(raw_data)
        print(f'Loaded {sum(len(v) for v in self._data.values())} actions from {len(self._data)} subjects')

    def _load_train_data(self, raw_data):
        for full_key, seq_data in raw_data.items():
            parts = full_key.split(' ')
            subject = parts[0]
            action = parts[1] if len(parts) > 1 else "Default"
            if subject not in self._data:
                self._data[subject] = {}
                self._keypoints_2d[subject] = {}
                self._valid_frames[subject] = {}
                self._cameras[subject] = {}

            # ── 只在这里增加对测试集格式的判断和处理 ──
            if isinstance(seq_data, dict) and 'data_3d' in seq_data and 'data_2d' in seq_data:
                # 测试集格式：直接是 {data_2d, data_3d, [valid], ...}
                self._process_test_sequence(subject, action, seq_data)
                continue

            # 原有训练集格式处理
            if isinstance(seq_data, (list, tuple)) and len(seq_data) >= 1:
                cam_dict = seq_data[0] if isinstance(seq_data[0], dict) else {}
                seq_id = seq_data[1] if len(seq_data) > 1 else None
            elif isinstance(seq_data, dict):
                cam_dict = seq_data
                seq_id = None
            else:
                print(f"Warning: Unknown format for {full_key}: {type(seq_data)}")
                continue

            self._process_action(subject, action, cam_dict, seq_id)

    def _process_test_sequence(self, subject, action, ts_dict):
        """新增：处理测试集单视图格式（仅参考旧版正确部分）"""
        if action not in self._data[subject]:
            self._data[subject][action] = {'positions': [], 'cameras': []}
            self._keypoints_2d[subject][action] = []
            self._valid_frames[subject][action] = []

        pos_3d = ts_dict['data_3d'].astype('float32')
        # 如果是毫米，转换（根据你的训练集是否已转换决定是否开启）
        pos_3d = pos_3d / 1000.0
        pos_3d = pos_3d - pos_3d[:, 0:1, :]  # 根对齐，与训练一致

        pos_2d = ts_dict['data_2d'].astype('float32')

        # 分辨率特殊处理（TS5/TS6 常见不同）
        cam_res_w = 1920 if subject in ['TS5', 'TS6'] else self.res_w
        cam_res_h = 1080 if subject in ['TS5', 'TS6'] else self.res_h

        # 2D 归一化到 [-1, 1]，与训练集一致
        if pos_2d.shape[-1] >= 2:
            pos_2d_norm = pos_2d.copy()
            pos_2d_norm[..., 0] = pos_2d[..., 0] / cam_res_w * 2 - 1
            pos_2d_norm[..., 1] = pos_2d[..., 1] / cam_res_h * 2 - 1
            if pos_2d.shape[-1] == 2:
                conf = np.ones((*pos_2d.shape[:2], 1), dtype='float32')
                pos_2d_norm = np.concatenate([pos_2d_norm[..., :2], conf], axis=-1)
            pos_2d = pos_2d_norm

        # 作为单相机添加
        self._data[subject][action]['positions'].append(pos_3d)
        self._keypoints_2d[subject][action].append(pos_2d)

        cam = {
            'id': 0,
            'res_w': cam_res_w,
            'res_h': cam_res_h,
            'seq_id': None
        }
        self._data[subject][action]['cameras'].append(cam)

        # valid frames（测试集常有）
        valid = ts_dict.get('valid')
        if valid is not None:
            valid = valid.astype(bool)
        else:
            valid = np.ones((pos_3d.shape[0],), dtype=bool)
        self._valid_frames[subject][action].append(valid)

        # 兼容旧接口
        self._cameras[subject][0] = cam

    # 原 _process_action 函数保持完全不变
    def _process_action(self, subject, action, cam_dict, seq_id=None):
        if action not in self._data[subject]:
            self._data[subject][action] = {'positions': [], 'cameras': []}
            self._keypoints_2d[subject][action] = []
            self._valid_frames[subject][action] = []

        view_ids = sorted([str(k) for k in cam_dict.keys()], key=lambda x: int(x) if x.isdigit() else x)
        for v_id in view_ids:
            view_data = cam_dict.get(v_id)
            if view_data is None:
                view_data = cam_dict.get(int(v_id) if v_id.isdigit() else v_id)
            if view_data is None or not isinstance(view_data, dict):
                continue
            if 'data_2d' not in view_data or 'data_3d' not in view_data:
                continue

            pos_3d = view_data['data_3d'].astype('float32')
            pos_3d = pos_3d / 1000.0   # 如需要则打开，与训练集一致
            pos_3d = pos_3d - pos_3d[:, 0:1, :]

            pos_2d = view_data['data_2d'].astype('float32')
            cam_res_w = view_data.get('res_w', self.res_w)
            cam_res_h = view_data.get('res_h', self.res_h)

            if pos_2d.shape[-1] >= 2:
                pos_2d_norm = pos_2d.copy()
                pos_2d_norm[..., 0] = pos_2d[..., 0] / cam_res_w * 2 - 1
                pos_2d_norm[..., 1] = pos_2d[..., 1] / cam_res_h * 2 - 1
                if pos_2d.shape[-1] == 2:
                    conf = np.ones((*pos_2d.shape[:2], 1), dtype='float32')
                    pos_2d_norm = np.concatenate([pos_2d_norm[..., :2], conf], axis=-1)
                pos_2d = pos_2d_norm

            self._data[subject][action]['positions'].append(pos_3d)
            self._keypoints_2d[subject][action].append(pos_2d)

            cam = {
                'id': int(v_id) if v_id.isdigit() else v_id,
                'res_w': cam_res_w,
                'res_h': cam_res_h,
                'seq_id': seq_id
            }
            self._data[subject][action]['cameras'].append(cam)

            valid = np.ones((pos_3d.shape[0],), dtype=bool)
            self._valid_frames[subject][action].append(valid)

            self._cameras[subject][int(v_id) if v_id.isdigit() else v_id] = cam
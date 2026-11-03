import os
import sys
import time
import glob
import time
import torch
import random
import socket
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.utils.data
from common.utils import *
import torch.optim as optim
from common.camera import *
import common.loss as eval_loss
from common.arguments import parse_args
from common.load_data_hm36 import Fusion
from common.h36m_dataset import Human36mDataset
from common.humaneva_dataset import HumanEvaDataset
args = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

exec('from model.' + args.model + ' import Model')


def train(dataloader, model, optimizer, epoch):
    model.train()
    loss_all = {'loss': AccumLoss()}

    for i, data in enumerate(tqdm(dataloader, disable=False, dynamic_ncols=False)):
        batch_cam, gt_3D, input_2D, input_2D_GT, action, subject, cam_ind = data
        input_2D, input_2D_GT, gt_3D, batch_cam = input_2D.cuda(), input_2D_GT.cuda(), gt_3D.cuda(), batch_cam.cuda()

        output_3D = model(input_2D)

        out_target = gt_3D.clone()
        out_target[:, :, args.root_joint] = 0

        if args.dataset == 'h36m':
            w_mpjpe = torch.tensor([1, 1, 2.5, 2.5, 1, 2.5, 2.5, 1, 1, 1, 1.5, 1.5, 4, 4, 1.5, 4, 4]).cuda()
        else:
            w_mpjpe = torch.ones(args.out_joints, device=gt_3D.device, dtype=gt_3D.dtype)

        loss_w_mpjpe = eval_loss.weighted_mpjpe(output_3D, out_target, w_mpjpe)
        loss_temporal = eval_loss.temporal_consistency(output_3D, out_target, w_mpjpe)
        loss_mean_velocity = eval_loss.mean_velocity(output_3D, out_target, axis=1)

        loss = loss_w_mpjpe + 0.5 * loss_temporal + 2.0 * loss_mean_velocity

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        N = input_2D.shape[0]
        loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)
        # # ---- 打印 ----
        # print(
        #       f"MPJPE: {loss_w_mpjpe.item():.6f}  "
        #       f"Temp: {loss_temporal.item():.6f}(w={0.5})  "
        #       f"Vel:  {loss_mean_velocity.item():.6f}(w={2})  "
        #       f"IB:   {ibloss.item():.6f} (w={0.01})  ")
    return loss_all['loss'].avg


def test(actions, dataloader, model):
    model.eval()

    action_error = define_error_list(actions)

    joints_left = list(getattr(getattr(dataloader, 'dataset', None), 'joints_left', []))
    joints_right = list(getattr(getattr(dataloader, 'dataset', None), 'joints_right', []))

    if len(joints_left) == 0 or len(joints_right) == 0:
        if args.dataset == 'h36m':
            joints_left = [4, 5, 6, 11, 12, 13]
            joints_right = [1, 2, 3, 14, 15, 16]
        elif args.dataset == 'humaneva15':
            joints_left = [2, 3, 4, 8, 9, 10]
            joints_right = [5, 6, 7, 11, 12, 13]
        else:
            raise ValueError('No joints_left/right available for dataset {}'.format(args.dataset))

    # 创建一个标志位，确保只运行一次可视化
    vis_done = False

    for i, data in enumerate(tqdm(dataloader, disable=False, dynamic_ncols=False)):
        batch_cam, gt_3D, input_2D, input_2D_GT, action, subject, cam_ind = data
        input_2D, input_2D_GT, gt_3D, batch_cam = input_2D.cuda(), input_2D_GT.cuda(), gt_3D.cuda(), batch_cam.cuda()

        # # ================= 新增代码开始 =================
        # # 只在第一个 batch (i==0) 且尚未执行可视化时运行
        # if i == 10 and not vis_done:
        #     print("\n>>> Generating Attention Maps for All Layers (Layer 0-7)...")
        #
        #     # 1. 准备输入数据
        #     # 你的代码使用了 TTA (input_2D[:, 0] 是原图，input_2D[:, 1] 是翻转图)
        #     # 我们可视化原始输入 (index 0) 的注意力
        #     vis_input = input_2D[:, 0]
        #
        #     # 2. 运行模型获取注意力
        #     # return_attention=True: 返回注意力矩阵
        #     # target_layer=None: 捕获所有层
        #     with torch.no_grad():
        #         _, attn_data = model(vis_input, return_attention=True, target_layer=None)
        #
        #     # 3. 保存所有层的图片到 ./vis_attn_test_results 文件夹
        #     # 注意：这里我们稍微变通一下，直接调用一个简化版的保存逻辑，
        #     # 或者复用之前的 visualize_all_layers 但需要它支持传入 attn_data。
        #     # 为了最稳妥，我们在这里直接内联调用之前的逻辑（假设你已将 visualize_all_layers 改为支持传入数据，
        #     # 或者我们直接在这里调用它，让它内部再跑一次模型——对于 i=0 来说开销可以忽略不计且代码最简单）。
        #
        #     # 【方案 A：最简单，让函数内部再跑一次模型】
        #     # 既然已经在 eval 模式且 no_grad 下，再跑一次 vis_input 开销很小，代码改动最小
        #     visualize_all_layers(model, vis_input, output_dir="./vis_attn_test_results")
        #
        #     vis_done = True
        #     print(">>> Attention maps saved to ./vis_attn_test_results\n")
        # ================= 新增代码结束 =================

        if input_2D.dim() == 5:
            output_3D_non_flip = model(input_2D[:, 0])
            output_3D_flip = model(input_2D[:, 1])

            output_3D_flip[:, :, :, 0] *= -1
            output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]

            output_3D = (output_3D_non_flip + output_3D_flip) / 2
        elif input_2D.dim() == 4:
            output_3D = model(input_2D)
        else:
            raise ValueError('Unexpected input_2D shape: {}'.format(tuple(input_2D.shape)))

        out_target = gt_3D.clone()
        if args.stride == 1:
            out_target = out_target[:, args.pad].unsqueeze(1)
            output_3D = output_3D[:, args.pad].unsqueeze(1)

        output_3D[:, :, args.root_joint] = 0
        out_target[:, :, args.root_joint] = 0

        action_error = test_calculation(output_3D, out_target, action, action_error, args.dataset, subject)

    p1, p2 = print_error(args.dataset, action_error, 1)

    return p1, p2


def build_test_loader(args, dataset):
    test_data = Fusion(args, dataset, args.root_path, train=False)
    return torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                                       shuffle=False, num_workers=int(args.workers), pin_memory=True)


def run_final_full_test(args, actions, test_dataset, model):
    if not args.train or not args.final_full_test:
        return

    print('Running final full test with eval_stride=1 and test_augmentation=True...')
    if args.previous_name != '' and os.path.exists(args.previous_name):
        state_dict = torch.load(args.previous_name)
        model.load_state_dict(state_dict)

    old_eval_stride = args.eval_stride
    old_test_augmentation = args.test_augmentation
    args.eval_stride = 1
    args.test_augmentation = True

    full_test_dataloader = build_test_loader(args, test_dataset)
    with torch.no_grad():
        p1, p2 = test(actions, full_test_dataloader, model)
    print('final full test p1: %.2f, p2: %.2f' % (p1, p2))
    logging.info('final full test p1: %.2f, p2: %.2f' % (p1, p2))

    args.eval_stride = old_eval_stride
    args.test_augmentation = old_test_augmentation


if __name__ == '__main__':
    seed = 1126

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if args.dataset == 'h36m':
        dataset_path = args.root_path + 'data_3d_' + args.dataset + '.npz'
        dataset = Human36mDataset(dataset_path, args)
        train_dataset = dataset
        test_dataset = dataset
    elif args.dataset == 'humaneva15':
        dataset_path = args.root_path + 'data_3d_' + args.dataset + '.npz'
        dataset = HumanEvaDataset(dataset_path)
        train_dataset = dataset
        test_dataset = dataset
    elif args.dataset == '3dhp':
        from common.mpii3dhp_dataset import MPII3DHPDataset
        train_dataset = MPII3DHPDataset(os.path.join(args.root_path, 'data_train_3dhp_hrnet.npz'), res_w=args.res_w_3dhp, res_h=args.res_h_3dhp)
        test_dataset = MPII3DHPDataset(os.path.join(args.root_path, 'data_test_3dhp_hrnet.npz'), res_w=args.res_w_3dhp, res_h=args.res_h_3dhp)
        dataset = train_dataset
    elif args.dataset in ['3dpw', 'pw3d']:
        from common.pw3d_dataset import PW3DDataset
        train_dataset = PW3DDataset(args.root_path, args, splits=('train', 'validation'))
        test_dataset = PW3DDataset(args.root_path, args, splits=('test',))
        dataset = train_dataset
    else:
        raise ValueError('Unsupported dataset: {}'.format(args.dataset))
    
    if args.dataset == 'h36m':
        actions = define_actions(args.actions, args.dataset)
    else:
        # 按照 VideoPose3D 逻辑：从测试数据集中动态提取所有 action 名称
        # 3DHP 必须基于 test_dataset（TS1..TS6），否则会抽取为空
        source_dataset = test_dataset if args.dataset in ['3dhp', '3dpw', 'pw3d'] else dataset

        all_actions = []
        subjects = args.subjects_test.split(',')
        for s in subjects:
            if hasattr(source_dataset, '_data') and s in source_dataset._data:
                for a in source_dataset._data[s].keys():
                    all_actions.append(a)

        unique_actions = sorted(list(set(all_actions)))

        if args.actions != '*' and args.actions != 'All':
            specified = args.actions.split(',')
            actions = [a for a in unique_actions if any(sp in a for sp in specified)]
            if len(actions) == 0:
                actions = specified
        else:
            actions = unique_actions

        # 兜底：避免 actions 为空导致评测统计表初始化异常
        if len(actions) == 0:
            actions = ['Default']

        print(f"INFO: Dynamic actions for {args.dataset}: {actions}")


    if args.train:
        train_data = Fusion(args, train_dataset, args.root_path, train=True)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                                       shuffle=True, num_workers=int(args.workers), pin_memory=True)
    test_dataloader = build_test_loader(args, test_dataset)

    model = Model(args).cuda()
    model = nn.DataParallel(model)

    if args.previous_dir != '':
        Load_model(args, model)

    lr = args.lr
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    # # 分离参数
    # scorer_params = []
    # other_params = []
    #
    # for name, param in model.named_parameters():
    #     if 'token_scorer.score_weights' in name:
    #         scorer_params.append(param)
    #         print(f"[INFO] Score weights param: {name} (requires_grad: {param.requires_grad})")
    #     else:
    #         other_params.append(param)
    #
    # # 创建优化器：为score_weights设置wd=0和更高lr
    # optimizer = optim.AdamW([
    #     {'params': other_params, 'weight_decay': 0.1, 'lr': lr},
    #     {'params': scorer_params, 'weight_decay': 0.0, 'lr': lr * 100},  # 关键1: wd=0
    # ], lr=lr)  # 关键2: score_weights的学习率放大100倍

    best_epoch = 0
    loss_epochs = []
    mpjpes = []
    # 指标名称映射
    metric_names = ['Temporal', 'Diversity', 'Represent', 'Centrality', 'DPC']

    for epoch in range(1, args.nepoch + 1):
        if args.train:
            loss = train(train_dataloader, model, optimizer, epoch)
            loss_epochs.append(loss * 1000)

        with torch.no_grad():
            p1, p2 = test(actions, test_dataloader, model)
            mpjpes.append(p1)

        if args.train and p1 < args.previous_best:
            best_epoch = epoch
            args.previous_name = save_model(args, epoch, p1, model, 'model')
            args.previous_best = p1

        if args.train:

            logging.info('epoch: %d, lr: %.6f, l: %.4f, p1: %.2f, p2: %.2f, %d: %.2f' % (
            epoch, lr, loss, p1, p2, best_epoch, args.previous_best))
            print('%d, lr: %.6f, l: %.4f, p1: %.2f, p2: %.2f, %d: %.2f' % (
            epoch, lr, loss, p1, p2, best_epoch, args.previous_best))

            if epoch % args.lr_decay_epoch == 0:
                lr *= args.lr_decay_large
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args.lr_decay_large
            else:
                lr *= args.lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args.lr_decay
        else:
            print('p1: %.2f, p2: %.2f' % (p1, p2))
            break

    run_final_full_test(args, actions, test_dataset, model)

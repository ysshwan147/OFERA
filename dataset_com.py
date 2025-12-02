import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import csv
import random

import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R


def load_data(
    subject_ids, 
    flame_num=50, jaw_pose_num=6, eyes_pose_num=12, 
    dataset_path="/root/Dataset", 
    dataset="NeRSemble-dataset", 
    max_bound_frames=1500
):
    all_arkit = []
    all_flame_param = []

    base_path = os.path.join(dataset_path, dataset)

    for subject_id in subject_ids:
        subject_path = os.path.join(base_path, subject_id)
        
        arkit_path = None
        flame_root = None

        if dataset == "ava-256-dataset":
            arkit_path = os.path.join(subject_path, "decoder", "arkit", "video_bs_param_scores.npy")
            flame_root = os.path.join(subject_path, "decoder", "metrical-tracker", "output", subject_id[-6:], "checkpoint.npz")
        else:
            arkit_path = os.path.join(subject_path, "ysh", "arkit", "video_bs_param_scores.npy")
            flame_root = os.path.join(subject_path, "ysh", "met", "output", subject_id, "checkpoint.npz")
        
        if not os.path.exists(arkit_path):
            print(f"❌ ARKit data missing for {subject_id}")
            continue
        arkit = np.load(arkit_path)

        if arkit.shape[1] == 52:
            print(f"🧹 Removing _neutral column from ARKit: {arkit_path}")
            arkit = arkit[:, 1:]

        arkit = arkit[1:, :]

        data = np.load(flame_root)

        num_frames = data["exp"].shape[0]

        expr = data["exp"]  # (T, 100)
        jaw_pose = data["jaw"]  # (T, 6)
        eyes_pose = data["eyes"]  # (T, 12)

        expr = expr[:, :flame_num]         # (T, Expr)
        jaw_pose = jaw_pose[:, :jaw_pose_num]     # (T, Jaw)
        eyes_pose = eyes_pose[:, :eyes_pose_num]    # (T, Eyes)

        flame_param = np.concatenate([expr, jaw_pose, eyes_pose], axis=1)

        if len(arkit) > max_bound_frames:
            indices = np.random.choice(len(arkit), size=max_bound_frames, replace=False)
            arkit = arkit[indices]
            flame_param = flame_param[indices]

        assert len(arkit) == len(flame_param)

        all_arkit.append(arkit)
        all_flame_param.append(flame_param)

    for sid, a, f in zip(subject_ids, all_arkit, all_flame_param):
        print(f"{sid}: {len(a)} ARKit, {len(f)} FLAME")

    all_arkit = np.concatenate([*all_arkit], axis=0)  # (T, 51)
    all_flame_param = np.concatenate([*all_flame_param], axis=0)  # (T, 68)

    return all_arkit, all_flame_param


class ARKitToFLAMEDataset(Dataset):
    def __init__(self, subject_ids_dict, base_path="/root/Dataset", normalize=True):
        self.input_data = []
        self.output_data = []
        self.in_dim = 51
        
        self.flame_num = 50
        self.jaw_pose_num = 6
        self.eyes_pose_num = 12
        self.out_dim = self.flame_num + self.jaw_pose_num + self.eyes_pose_num
        
        dataset_list = ['NeRSemble-dataset', 'insta', 'ava-256-dataset']
        # dataset_list = ['NeRSemble-dataset', 'insta']
        # dataset_list = ['NeRSemble-dataset', 'ava-256-dataset']
        # dataset_list = ['insta', 'ava-256-dataset']
        # dataset_list = ['NeRSemble-dataset']
        # dataset_list = ['insta']
        # dataset_list = ['ava-256-dataset']
        
        min_num_frames = 10000
        for dataset in dataset_list:
            dataset_path = os.path.join(base_path, dataset)
            
            for subject_id in subject_ids_dict[dataset]:
                subject_path = os.path.join(dataset_path, subject_id)
                
                if dataset == "ava-256-dataset":
                    arkit_path = os.path.join(subject_path, "decoder", "arkit", "video_bs_param_scores.npy")
                else:
                    arkit_path = os.path.join(subject_path, "ysh", "arkit", "video_bs_param_scores.npy")

                if not os.path.exists(arkit_path):
                    print(f"❌ ARKit data missing for {subject_id}")
                    continue
                arkit = np.load(arkit_path)
                arkit = arkit[1:, :]
                num_frames = arkit.shape[0]
                
                if min_num_frames > num_frames:
                    min_num_frames = num_frames
        
        input_data_list = []
        output_data_list = []
        
        for dataset in dataset_list:
            subject_ids = subject_ids_dict[dataset]
            
            input_data, output_data = load_data(subject_ids, dataset=dataset, max_bound_frames=min_num_frames)
            
            input_data_list.append(input_data)
            output_data_list.append(output_data)

        self.input_data = np.concatenate(input_data_list, axis=0)
        self.output_data = np.concatenate(output_data_list, axis=0)
        
        assert self.input_data.shape[0] == self.output_data.shape[0], "Input and output data must have the same number of frames"
        assert self.input_data.shape[1] == self.in_dim, f"Input data must have {self.in_dim} dimensions, but got {self.input_data.shape[1]}"
        assert self.output_data.shape[1] == self.out_dim, f"Output data must have {self.out_dim} dimensions, but got {self.output_data.shape[1]}"
        
        print(f"✅ Total frames: {len(self.input_data)}")

        self.expr_data = self.output_data[:, :self.flame_num]
        self.jaw_data = self.output_data[:, self.flame_num:self.flame_num + self.jaw_pose_num]
        self.eyes_data = self.output_data[:, self.flame_num + self.jaw_pose_num:self.out_dim]

        if normalize:
            self.input_mean = self.input_data.mean(axis=0)
            self.input_std = self.input_data.std(axis=0)

            self.input_std[self.input_std < 1e-4] = 1.0

            self.input_data = (self.input_data - self.input_mean) / self.input_std

            self.expr_mean = self.expr_data.mean(axis=0)
            self.expr_std = self.expr_data.std(axis=0)
            self.jaw_mean = self.jaw_data.mean(axis=0)
            self.jaw_std = self.jaw_data.std(axis=0)
            self.eyes_mean = self.eyes_data.mean(axis=0)
            self.eyes_std = self.eyes_data.std(axis=0)

            self.expr_std[self.expr_std < 1e-4] = 1.0
            self.jaw_std[self.jaw_std < 1e-4] = 1.0
            self.eyes_std[self.eyes_std < 1e-4] = 1.0

            self.expr_data = (self.expr_data - self.expr_mean) / self.expr_std
            self.jaw_data = (self.jaw_data - self.jaw_mean) / self.jaw_std
            self.eyes_data = (self.eyes_data - self.eyes_mean) / self.eyes_std

            self.output_data = np.concatenate([self.expr_data, self.jaw_data, self.eyes_data], axis=1)
        else:
            self.input_mean = self.expr_mean = self.jaw_mean = self.eyes_mean = 0
            self.input_std = self.expr_std = self.jaw_std = self.eyes_std = 1
            

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        x = torch.tensor(self.input_data[idx], dtype=torch.float32)
        y = torch.tensor(self.output_data[idx], dtype=torch.float32)
        return x, y

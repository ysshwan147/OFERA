
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
import random

from dataset_com import ARKitToFLAMEDataset
from model import ARKitToFLAMENet

nersemble_subject_ids = ["036","037","057","074","100","145","165","178","210","251"]
insta_subject_ids = ["bala","biden","justin","malte_1","marcel","nf_01","nf_03","obama","person_0004","wojtek_1"]
ava256_subject_ids = ["20230814--0822--GTA798","20230814--1700--LVD531","20230901--1429--CPP930","20230906--1059--QLL112","20230918--1040--NTA876","20230918--1402--UIQ957","20230919--1056--JEN167","20230919--1626--RHU956","20230920--1404--INQ807","20230925--1356--PKH444"]

def build_splits(split_ratio=0.8):
    random.shuffle(nersemble_subject_ids)
    random.shuffle(insta_subject_ids)
    random.shuffle(ava256_subject_ids)

    train_subject_ids_dict = {}
    val_subject_ids_dict = {}
    test_subject_ids_dict = {}

    train_size = round(len(nersemble_subject_ids) * split_ratio)
    val_size = round(len(nersemble_subject_ids) * (1 - split_ratio)) // 2

    train_subject_ids_dict["NeRSemble-dataset"] = nersemble_subject_ids[:train_size]
    train_subject_ids_dict["insta"] = insta_subject_ids[:train_size]
    train_subject_ids_dict["ava-256-dataset"] = ava256_subject_ids[:train_size]

    val_subject_ids_dict["NeRSemble-dataset"] = nersemble_subject_ids[train_size:train_size + val_size]
    val_subject_ids_dict["insta"] = insta_subject_ids[train_size:train_size + val_size]
    val_subject_ids_dict["ava-256-dataset"] = ava256_subject_ids[train_size:train_size + val_size]

    test_subject_ids_dict["NeRSemble-dataset"] = nersemble_subject_ids[train_size + val_size:]
    test_subject_ids_dict["insta"] = insta_subject_ids[train_size + val_size:]
    test_subject_ids_dict["ava-256-dataset"] = ava256_subject_ids[train_size + val_size:]

    return train_subject_ids_dict, val_subject_ids_dict, test_subject_ids_dict


def train(args):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(args.output_path, now)
    os.makedirs(output_path, exist_ok=True)

    # Dataset
    train_ids, val_ids, test_ids = build_splits(split_ratio=0.8)
    train_dataset = ARKitToFLAMEDataset(train_ids, base_path=args.input_path, normalize=True)
    val_dataset   = ARKitToFLAMEDataset(val_ids,   base_path=args.input_path, normalize=False)
    test_dataset  = ARKitToFLAMEDataset(test_ids,  base_path=args.input_path, normalize=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=64)
    test_loader  = DataLoader(test_dataset,  batch_size=64)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ARKitToFLAMENet(in_dim=train_dataset.in_dim,
                            hidden_dim=args.hidden_dim,
                            out_dim=train_dataset.out_dim,
                            num_blocks=args.num_blocks).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Loss weights (expr/jaw/eyes)
    w_expr = w_jaw = w_eyes = 1.0
    w_sum = w_expr + w_jaw + w_eyes

    expr_num = train_dataset.flame_num
    jaw_num  = train_dataset.jaw_pose_num
    eyes_num = train_dataset.eyes_pose_num

    # Stats
    input_mean = torch.tensor(train_dataset.input_mean, dtype=torch.float32, device=device)
    input_std  = torch.tensor(train_dataset.input_std,  dtype=torch.float32, device=device)
    expr_mean  = torch.tensor(train_dataset.expr_mean,  dtype=torch.float32, device=device)
    expr_std   = torch.tensor(train_dataset.expr_std,   dtype=torch.float32, device=device)
    jaw_mean   = torch.tensor(train_dataset.jaw_mean,   dtype=torch.float32, device=device)
    jaw_std    = torch.tensor(train_dataset.jaw_std,    dtype=torch.float32, device=device)
    eyes_mean  = torch.tensor(train_dataset.eyes_mean,  dtype=torch.float32, device=device)
    eyes_std   = torch.tensor(train_dataset.eyes_std,   dtype=torch.float32, device=device)

    best_val = float('inf')

    for epoch in range(1, args.num_epoch + 1):
        # Train
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)

            expr_gt = yb[:, :expr_num]
            jaw_gt  = yb[:, expr_num:expr_num + jaw_num]
            eyes_gt = yb[:, expr_num + jaw_num:expr_num + jaw_num + eyes_num]

            expr_pred = pred[:, :expr_num]
            jaw_pred  = pred[:, expr_num:expr_num + jaw_num]
            eyes_pred = pred[:, expr_num + jaw_num:expr_num + jaw_num + eyes_num]

            loss_expr = F.smooth_l1_loss(expr_pred, expr_gt)
            loss_jaw  = F.smooth_l1_loss(jaw_pred,  jaw_gt)
            loss_eyes = F.smooth_l1_loss(eyes_pred, eyes_gt)

            loss = (w_expr*loss_expr + w_jaw*loss_jaw + w_eyes*loss_eyes) / w_sum
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train = total_loss / len(train_loader)

        # Val
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                xb = (xb - input_mean) / input_std
                expr_gt = yb[:, :expr_num]
                jaw_gt  = yb[:, expr_num:expr_num + jaw_num]
                eyes_gt = yb[:, expr_num + jaw_num:expr_num + jaw_num + eyes_num]

                expr_gt = (expr_gt - expr_mean) / expr_std
                jaw_gt  = (jaw_gt  - jaw_mean)  / jaw_std
                eyes_gt = (eyes_gt - eyes_mean) / eyes_std

                pred = model(xb)
                expr_pred = pred[:, :expr_num]
                jaw_pred  = pred[:, expr_num:expr_num + jaw_num]
                eyes_pred = pred[:, expr_num + jaw_num:expr_num + jaw_num + eyes_num]

                loss_expr = F.smooth_l1_loss(expr_pred, expr_gt)
                loss_jaw  = F.smooth_l1_loss(jaw_pred,  jaw_gt)
                loss_eyes = F.smooth_l1_loss(eyes_pred, eyes_gt)

                loss = (w_expr*loss_expr + w_jaw*loss_jaw + w_eyes*loss_eyes) / w_sum
                total_val += loss.item()
        avg_val = total_val / len(val_loader)
        print(f"[Epoch {epoch}] Train {avg_train:.6f} | Val {avg_val:.6f}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), os.path.join(output_path, "best_mlp_model.pth"))
            print(f"📦 Saved best baseline @ epoch {epoch}")

    # TorchScript (optional)
    if args.torch_script:
        model.eval()
        example = torch.randn(1, train_dataset.in_dim).to(device)
        traced = torch.jit.trace(model, example)
        traced.save(os.path.join(output_path, "baseline_script.pt"))
        print("✅ Saved TorchScript baseline")

    # Save stats
    np.savez(os.path.join(output_path, "stats.npz"),
        input_mean=train_dataset.input_mean,
        input_std=train_dataset.input_std,
        expr_mean=train_dataset.expr_mean,
        expr_std=train_dataset.expr_std,
        jaw_mean=train_dataset.jaw_mean,
        jaw_std=train_dataset.jaw_std,
        eyes_mean=train_dataset.eyes_mean,
        eyes_std=train_dataset.eyes_std
    )
    print("🧮 Saved stats_baseline.npz")


if __name__ == "__main__":
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", type=str, required=True)
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--num_epoch", type=int, required=True)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--num_blocks", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--torch_script", action="store_true")
    args = p.parse_args()
    train(args)

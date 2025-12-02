import numpy as np
import torch
import os
import sys
from pytorch3d.transforms import matrix_to_rotation_6d, axis_angle_to_matrix

# from model_residual import ARKitToFLAMENet
from model import ARKitToFLAMENet

sys.path.append(os.path.abspath("/root/mediapipe-blendshapes-to-flame"))
from mp_2_flame import MP_2_FLAME

weight_idx = 0

dataset_list = ["NeRSemble-dataset", "insta", "ava-256-dataset"]
subject_ids = ["037", "wojtek_1", "20230920--1404--INQ807"]

for i in range(len(dataset_list)):
    dataset = dataset_list[i]
    subject_id = subject_ids[i]
    
    dataset_path = f"/root/Dataset/{dataset}"

    if dataset == "ava-256-dataset":
        arkit_base_path = os.path.join(dataset_path, subject_id, "decoder", "arkit")
        met_base_path = os.path.join(dataset_path, subject_id, "decoder", "metrical-tracker", "output", subject_id[-6:])
    else:
        arkit_base_path = os.path.join(dataset_path, subject_id, "ysh", "arkit")
        met_base_path = os.path.join(dataset_path, subject_id, "ysh", "met", "output", subject_id)

    npy_path = os.path.join(arkit_base_path, "video_bs_param_scores.npy")         # (T, 51)
    checkpoint_path = os.path.join(met_base_path, "checkpoint.npz")
    output_mp_path = os.path.join(met_base_path, f"checkpoint_mp{weight_idx}.npz")
    output_linear_path = os.path.join(met_base_path, f"checkpoint_linear{weight_idx}.npz")
    output_mat_path = os.path.join(met_base_path, f"checkpoint_mat{weight_idx}.npz")

    # model
    # mlp
    weight_list = ["dummy" for _ in range(100)]
    
    weight_list[0] = "./weights_ori/2025-08-09_17-26-29"
    
    weight_list[1] = "./weights/2025-07-28_15-54-43"
    weight_list[2] = "./weights/2025-07-28_15-56-45"
    weight_list[3] = "./weights/2025-07-28_15-58-00"
    weight_list[4] = "./weights/2025-07-28_16-06-49"
    weight_list[5] = "./weights/2025-07-28_16-08-29"
    weight_list[6] = "./weights/2025-07-28_16-10-33"
    weight_list[7] = "./weights/2025-07-28_16-12-14"

    weight_list[11] = "./weights/2025-07-29_23-29-17"
    weight_list[12] = "./weights/2025-07-29_23-29-44"
    weight_list[13] = "./weights/2025-07-29_23-33-08"
    weight_list[14] = "./weights/2025-07-29_23-33-20"
    weight_list[15] = "./weights/2025-07-29_23-36-36"
    weight_list[16] = "./weights/2025-07-29_23-36-48"
    weight_list[17] = "./weights/2025-07-29_23-38-01"
    
    weight_list[21] = "./weights_ori/2025-08-09_17-04-49"
    weight_list[22] = "./weights_ori/2025-08-09_17-26-29"
    
    weight_list[31] = "./weights_sep/2025-08-09_17-06-32"
    weight_list[32] = "./weights_sep/2025-08-09_17-27-59"
    
    weight_list[41] = "./weights_sha/2025-08-09_17-10-12"
    weight_list[42] = "./weights_sha/2025-08-09_17-31-00"
    
    weight_path = weight_list[weight_idx]
    mlp_weight_path = os.path.join(weight_path, "best_mlp_model.pth")
    stats_path = os.path.join(weight_path, "stats.npz")

    stats_data = np.load(stats_path)
    input_mean = stats_data['input_mean']
    input_std = stats_data['input_std']
    expr_mean = stats_data['expr_mean']
    expr_std = stats_data['expr_std']
    jaw_mean = stats_data['jaw_mean']
    jaw_std = stats_data['jaw_std']
    eyes_mean = stats_data['eyes_mean']
    eyes_std = stats_data['eyes_std']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp = ARKitToFLAMENet(hidden_dim=128, num_blocks=4)
    mlp.load_state_dict(torch.load(mlp_weight_path, map_location=device))
    mlp.to(device)
    mlp.eval()

    # linear mapping
    mp2flame = MP_2_FLAME(mappings_path='/root/mediapipe-blendshapes-to-flame/mappings', index=weight_idx)

    # matrix
    bs_mat_path = "./data/mat_final.npy"
    bs_mat = np.load(bs_mat_path)

    assert bs_mat.shape == (51, 103), f"Expected shape (51, 103), but got {matrix.shape}"

    # ARKit data load
    arkit = np.load(npy_path)  # shape: (T, 52)
    arkit = arkit[1:] # drop first flame for metrical-tracker

    frame_num = arkit.shape[0]

    ######################### inference ############################
    # linear mapping
    expr_mapped, jaw_mapped, eyes_mapped = mp2flame.convert(blendshape_scores=arkit)

    jaw_mapped = jaw_mapped[:, 3:]

    expr_mapped = expr_mapped.astype(np.float32)

    jaw_mapped = torch.tensor(jaw_mapped, dtype=torch.float32)
    leye_mapped = torch.tensor(eyes_mapped[:, :3], dtype=torch.float32)
    reye_mapped = torch.tensor(eyes_mapped[:, 3:], dtype=torch.float32)

    jaw_mapped = matrix_to_rotation_6d(axis_angle_to_matrix(jaw_mapped))
    leye_mapped = matrix_to_rotation_6d(axis_angle_to_matrix(leye_mapped))
    reye_mapped = matrix_to_rotation_6d(axis_angle_to_matrix(reye_mapped))

    jaw_mapped = jaw_mapped.cpu().numpy()
    eyes_mapped = np.zeros((frame_num, 12), dtype=np.float32)
    eyes_mapped[:, :6] = leye_mapped.cpu().numpy()
    eyes_mapped[:, 6:] = reye_mapped.cpu().numpy()

    # matrix
    if arkit.shape[1] == 52:
        arkit = arkit[:, 1:] # drop _neutral parameter
    
    result = np.dot(arkit, bs_mat)
    
    expr_mat = result[:, :100].astype(np.float32)
    
    jaw_mat = result[:, 100:103].astype(np.float32)
    jaw_mat = torch.tensor(jaw_mat, dtype=torch.float32)
    jaw_mat = matrix_to_rotation_6d(axis_angle_to_matrix(jaw_mat))
    jaw_mat = jaw_mat.cpu().numpy()
    
    eye_identity = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    eye_pair_identity = np.concatenate([eye_identity, eye_identity])  # shape (12,)
    eyes_mat = np.tile(eye_pair_identity, (frame_num, 1))  # shape (frame_num, 12)

    # mlp
    if arkit.shape[1] == 52:
        arkit = arkit[:, 1:] # drop _neutral parameter

    arkit_norm = (arkit - input_mean) / input_std
    arkit_tensor = torch.tensor(arkit_norm, dtype=torch.float32).to(device)

    expr_num = 50
    jaw_num = 6
    eyes_num = 12

    with torch.no_grad():
        out = mlp(arkit_tensor)
        if isinstance(out, (tuple, list)) and len(out) == 4:
            # SEP / SHA: (expr, jaw, eyeL, eyeR)
            expr_pred, jaw_pred, eyeL_pred, eyeR_pred = out
            y_pred_t = torch.cat([expr_pred, jaw_pred, eyeL_pred, eyeR_pred], dim=-1)
        else:
            # Baseline: (B, 68)
            y_pred_t = out
        output = y_pred_t.squeeze(0).cpu().numpy()

    expr_pred = output[:, :expr_num]
    jaw_pred = output[:, expr_num:expr_num + jaw_num]
    eyes_pred = output[:, expr_num + jaw_num:]

    expr_denorm = expr_pred * expr_std + expr_mean
    jaw_denorm = jaw_pred * jaw_std + jaw_mean
    eyes_denorm = eyes_pred * eyes_std + eyes_mean

    temp = np.zeros((frame_num, 100), dtype=np.float32)
    temp[:, :expr_num] = expr_denorm
    expr_denorm = temp

    original = dict(np.load(checkpoint_path))
    original["exp"] = expr_denorm
    original["jaw"] = jaw_denorm
    original["eyes"] = eyes_denorm

    np.savez(output_mp_path, **original)
    print(f"✅ Saved updated checkpoint to {output_mp_path}")

    original["exp"] = expr_mapped
    original["jaw"] = jaw_mapped
    original["eyes"] = eyes_mapped

    np.savez(output_linear_path, **original)
    print(f"✅ Saved updated checkpoint to {output_linear_path}")
    
    original["exp"] = expr_mat
    original["jaw"] = jaw_mat
    original["eyes"] = eyes_mat
    
    np.savez(output_mat_path, **original)
    print(f"✅ Saved updated checkpoint to {output_mat_path}")
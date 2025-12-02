import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import torch
import os
import sys
import glob
import argparse
# from model_residual import ARKitToFLAMENet
from model import ARKitToFLAMENet, ARKitToFLAMENet_SHA, ARKitToFLAMENet_SEP
from tqdm import tqdm


parser = argparse.ArgumentParser(description="MICA → MLP")
parser.add_argument(
    "--subject",
    type=str,
    required=True,
    help="subject name"
)
args = parser.parse_args()

subject_name = args.subject
print(f"🔹 Subject: {subject_name}")


print("Loading mediapipe face landmarker...")
base_options = python.BaseOptions(model_asset_path='./data/face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=False,
                                        num_faces=1)
mediapipe_detector = vision.FaceLandmarker.create_from_options(options)

print("Processing video & extracting blendshape scores...")

img_dir = f"/root/Dataset/ysh/metrical-tracker/input/{subject_name}/source"
img_ext = "*.png"

img_paths = sorted(glob.glob(os.path.join(img_dir, img_ext)))
total_frames = len(img_paths)
print(f"Total image frames: {total_frames}")
img_paths = img_paths[1:]
total_frames = len(img_paths)
print(f"Proccessing image frames: {total_frames}")

smooth_blendshape_scores = np.zeros((total_frames, 51), dtype=np.float32)
blendshape_scores = np.zeros((total_frames, 51), dtype=np.float32)
smooth_prev_blend_scores = np.zeros([51], dtype=np.float32)
prev_blend_scores = np.zeros([51], dtype=np.float32)

for frame_idx, img_path in enumerate(tqdm(img_paths, desc="Processing images")):
    frame = cv2.imread(img_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=frame
    )

    detection_result = mediapipe_detector.detect(image)

    if len(detection_result.face_blendshapes) == 0:
        tqdm.write(f"Frame {frame_idx} ({img_path}): No face detected")
        smooth_blendshape_scores[frame_idx] = smooth_prev_blend_scores
        blendshape_scores[frame_idx] = prev_blend_scores
        continue

    blend_scores = detection_result.face_blendshapes[0]
    blend_scores = np.array([l.score for l in blend_scores], dtype=np.float32)
    blend_scores = blend_scores[1:]

    SMOOTHING_FACTOR = 0.8
    smooth_blend_scores = SMOOTHING_FACTOR * blend_scores + (1 - SMOOTHING_FACTOR) * smooth_prev_blend_scores
    smooth_blend_scores = np.clip(smooth_blend_scores, 0, 1.0)

    smooth_blendshape_scores[frame_idx] = smooth_blend_scores
    blendshape_scores[frame_idx] = blend_scores

    smooth_prev_blend_scores = smooth_blend_scores
    prev_blend_scores = blend_scores
    

print("Loading model...")
model = ARKitToFLAMENet(hidden_dim=128, num_blocks=4)
weight_path = "/root/arkit2flame_dataset/weights_ori/2025-08-09_17-26-29"
model.load_state_dict(torch.load(f"{weight_path}/best_mlp_model.pth"))
model.eval()

stats_path = f"{weight_path}/stats.npz"
stats = np.load(stats_path)

input_mean = stats["input_mean"]
input_std = stats["input_std"]
expr_mean = stats["expr_mean"]
expr_std = stats["expr_std"]
jaw_mean = stats["jaw_mean"]
jaw_std = stats["jaw_std"]
eyes_mean = stats["eyes_mean"]
eyes_std = stats["eyes_std"]

x_norm = (blendshape_scores - input_mean) / input_std
x_tensor = torch.tensor(x_norm, dtype=torch.float32)

print("shape of x_tensor:", x_tensor.shape)

with torch.no_grad():
    out = model(x_tensor)
    if isinstance(out, (tuple, list)) and len(out) == 4:
        # SEP / SHA: (expr, jaw, eyeL, eyeR)
        expr_pred, jaw_pred, eyeL_pred, eyeR_pred = out
        y_pred_t = torch.cat([expr_pred, jaw_pred, eyeL_pred, eyeR_pred], dim=-1)
    else:
        # Baseline: (B, 68)
        y_pred_t = out
    y_pred = y_pred_t.squeeze(0).numpy()

print("shape of y_pred:", y_pred.shape)

output_expr = np.zeros((y_pred.shape[0], 100), dtype=np.float32)
flame_out_dim = y_pred.shape[1] - 6 - 12
output_expr[:, :flame_out_dim] = y_pred[:,:flame_out_dim] * expr_std[None, :] + expr_mean[None, :]
# print(("flame_out_dim", flame_out_dim))

output_jaw_pose = y_pred[:, flame_out_dim:flame_out_dim+6] * jaw_std[None, :] + jaw_mean[None, :]
output_eyes_pose = y_pred[:, flame_out_dim+6:] * eyes_std[None, :] + eyes_mean[None, :]

print("Output shapes:", output_expr.shape, output_jaw_pose.shape, output_eyes_pose.shape)
# print("Expression coefficients:", output_expr[:3])

output_param = np.concatenate([
    output_expr,
    output_jaw_pose,
    output_eyes_pose
], axis=1)


exp_dir = f'/root/Dataset/ysh/FateAvatar/data/insta/{subject_name}_mlp/flame/exp'
jaw_dir = f'/root/Dataset/ysh/FateAvatar/data/insta/{subject_name}_mlp/flame/jaw'
eyes_dir = f'/root/Dataset/ysh/FateAvatar/data/insta/{subject_name}_mlp/flame/eyes'

F = output_param.shape[0]

file_names = [f'{i:05d}.txt' for i in range(F)]

for i in tqdm(range(F), desc="Saving FLAME param files"):
    fname = file_names[i]
    param = output_param[i]  # shape: (118,)

    full_exp = np.zeros(100, dtype=np.float32)
    full_exp[:50] = param[:50]
    np.savetxt(os.path.join(exp_dir, fname), full_exp.reshape(-1, 1), fmt="%.8f")

    jaw_6d = param[100:106]
    np.savetxt(os.path.join(jaw_dir, fname), jaw_6d.reshape(-1, 1), fmt="%.8f")

    l_eye_6d = param[106:112]
    r_eye_6d = param[112:118]
    eye_6d = np.concatenate([l_eye_6d, r_eye_6d], axis=0)
    np.savetxt(os.path.join(eyes_dir, fname), eye_6d.reshape(-1, 1), fmt="%.8f")


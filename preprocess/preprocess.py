import os
import torch
import numpy as np
from tqdm import tqdm
import csv

def merge_frames_to_npz(frame_dir, output_path, failures, keys_to_include=['exp', 'jaw', 'eyes', 'shape', 'R', 't', 'fl', 'pp']):
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".frame")])
    
    frame_files.insert(0, "dummy.frame")
    
    num_frames = len(frame_files)
    valid_indices = sorted(set(range(num_frames)) - failures)
    
    frame_files = [frame_files[i] for i in valid_indices]
    frame_files = frame_files[1:]
    
    collected = {key: [] for key in keys_to_include}

    for fname in tqdm(frame_files, desc="Merging frames"):
        fpath = os.path.join(frame_dir, fname)
        frame = torch.load(fpath, map_location="cpu")
        flame = frame['flame']
        cam = frame['camera']
        
        for key in keys_to_include:
            if key in flame:
                collected[key].append(flame[key])
            elif key in cam:
                collected[key].append(cam[key])
            else:
                print(f"⚠️ Key '{key}' missing in {fname}")
                

    # Stack and save
    for key in collected:
        collected[key] = np.stack(collected[key], axis=0) # (T, 1, K)
        collected[key] = collected[key].squeeze(axis=1)  # (T, K)

    np.savez(output_path, **collected)
    print(f"✅ Saved merged data to {output_path}")


dataset = "NeRSemble-dataset"  # "insta", "ava-256-dataset"
dataset_path = "/root/Dataset/" + dataset

subject_ids = [
    "036",
    "037",
    "057",
    "074",
    "100",
    "145",
    "165",
    "178",
    "210",
    "251"
]

if dataset == "NeRSemble-dataset":
    failure_path = "/root/arkit2flame_dataset/preprocess/nersemble/detection_failures_nersemble.csv"
elif dataset == "insta":
    failure_path = "/root/arkit2flame_dataset/preprocess/insta/detection_failures_insta.csv"
else:
    failure_path = "/root/arkit2flame_dataset/preprocess/ava256/detection_failures_ava256.csv"

failure_dict = {}  # {subject_id: set(frame_idx)}

with open(failure_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    reader.__next__()  # Skip header
    for row in reader:
        subject, _, frame_idx = row
        if frame_idx == "0":
            continue
        if subject not in failure_dict:
            failure_dict[subject] = set()
        failure_dict[subject].add(int(frame_idx))

for subject_id in subject_ids:
    if dataset == "ava-256-dataset":
        base_path = os.path.join(dataset_path, subject_id, "decoder", "metrical-tracker", "output", subject_id[-6:])
    else:
        base_path = os.path.join(dataset_path, subject_id, "ysh", "met", "output", subject_id)
    
    flame_root = os.path.join(base_path, "checkpoint")

    if not os.path.exists(flame_root):
        print(f"❌ Flame data missing for {subject_id}")
        continue
    
    output_dir = base_path
    os.makedirs(output_dir, exist_ok=True)

    failures = failure_dict.get(subject_id, set())

    merge_frames_to_npz(
        frame_dir=flame_root, 
        output_path=os.path.join(output_dir, f"checkpoint.npz"),
        failures=failures
    )


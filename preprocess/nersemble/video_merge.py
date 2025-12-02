import os
import subprocess

subject_id = "210"
root_dir = f"/root/Dataset/NeRSemble-dataset/{subject_id}"

merge_list_dir = f"{root_dir}/ysh/merge_lists"
os.makedirs(merge_list_dir, exist_ok=True)

output_dir = f"{root_dir}/ysh/merged_videos"
os.makedirs(output_dir, exist_ok=True)

cam_names = [
    "cam_222200042", "cam_222200044", "cam_222200046", "cam_222200040",
    "cam_222200036", "cam_222200048", "cam_220700191", "cam_222200041",
    "cam_222200037", "cam_222200038", "cam_222200047", "cam_222200043",
    "cam_222200049", "cam_222200039", "cam_222200045", "cam_221501007"
]

target_dirs = [
    d for d in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, d))
    and d[:3] in {"EMO", "EXP", "SEN"}
    and "tongue" not in d.lower()
    and d != "FREE"
]

target_dirs.sort()

for cam_name in cam_names:
    list_path = os.path.join(merge_list_dir, f"merge_list_{cam_name}.txt")
    with open(list_path, "w") as f:
        for task in target_dirs:
            video_path = os.path.join(root_dir, task, f"{cam_name}.mp4")
            if os.path.exists(video_path):
                f.write(f"file '{video_path}'\n")

def merge_videos_ffmpeg(video_list, output_path):
    if not video_list:
        print(f"[!] No videos to merge for {output_path}")
        return

    list_file = "temp_merge_list.txt"
    with open(list_file, "w") as f:
        for path in video_list:
            f.write(f"file '{path}'\n")

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_file, "-c", "copy", output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(list_file)

for cam_name in cam_names:
    video_paths = []
    for task_dir in target_dirs:
        video_path = os.path.join(root_dir, task_dir, f"{cam_name}.mp4")
        if os.path.exists(video_path):
            video_paths.append(video_path)

    output_path = os.path.join(output_dir, f"{cam_name}.mp4")
    print(f"▶️ {cam_name}: {len(video_paths)} → {output_path}")
    merge_videos_ffmpeg(video_paths, output_path)


import os
import cv2
import shutil

cam_name = "cam_222200037"
subject_ids = ["036", "037", "057", "074", "100", "145", "165", "178", "210", "251"]

mica = False
met = True

for subject_id in subject_ids:

    root_dir = f"/root/Dataset/NeRSemble-dataset/{subject_id}/ysh"
    video_path = os.path.join(root_dir, "merged_videos", f"{cam_name}.mp4")

    # MICA
    if mica:
        mica_input_dir = "/root/MICA/demo/input"
        os.makedirs(mica_input_dir, exist_ok=True)

        saved = False

        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()

            if ret:
                output_path = os.path.join(mica_input_dir, f"{subject_id}.png")
                cv2.imwrite(output_path, frame)
                print(f"✅ Saved first frame to {output_path}")
                saved = True

        if not saved:
            print(f"[!] No valid video found for {cam_name}")


    # metrical-tracker
    if met:
        met_target_dir = f"{root_dir}/met/input"
        met_target_path = os.path.join(met_target_dir, "video.mp4")

        os.makedirs(met_target_dir, exist_ok=True)

        shutil.copy2(video_path, met_target_path)

        print(f"✅ Copied '{video_path}' → '{met_target_path}'")

        mica_output_path = f"/root/MICA/demo/output/{subject_id}/identity.npy"
        met_target_path = os.path.join(met_target_dir, "identity.npy")

        shutil.copy2(mica_output_path, met_target_path)

        print(f"✅ Copied '{mica_output_path}' → '{met_target_path}'")


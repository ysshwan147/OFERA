import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
import cv2
import os
from multiprocessing import Pool, cpu_count
from typing import List, Tuple
from scipy.ndimage import uniform_filter1d
import csv


def save_detection_failures(results, output_csv="detection_failures.csv"):
    failed_paths = [img_path for sublist in results for img_path in sublist if img_path is not None]

    extracted_info = []
    for path in failed_paths:
        try:
            parts = path.split(os.sep)
            subject_id = parts[-6]
            frame_num_png = parts[-1]
            frame_num = os.path.splitext(frame_num_png)[0]  # '000045.png' → '000045'

            paths = sorted(os.listdir(os.path.dirname(path)))

            frame_idx = paths.index(frame_num_png)

            extracted_info.append([subject_id, frame_num, frame_idx])
        except Exception as e:
            print(f"[!] Failed to parse path: {path} -- {e}")

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject_id", "frame_num", "frame_idx"])
        writer.writerows(extracted_info)

    print(f"[✓] Detection failure CSV saved: {output_csv}")


def arkit_extract(paths: Tuple[str, str]):
    input_path, output_path = paths

    base_options = BaseOptions(
        model_asset_path='../data/face_landmarker_v2_with_blendshapes.task', 
        delegate= "CPU"
    )
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    os.makedirs(output_path, exist_ok=True)

    detect_fail_list = []
    video_bs_param_scores = []
    smoothed_video_bs_param_scores = []
    bs_param_names = None
    prev_bs_param_scores = np.zeros([51], dtype=np.float32)
    smoothed_frame_bs_param_scores = np.zeros([51], dtype=np.float32)

    SMOOTHING_FACTOR = 0.8
    
    base_landmarks = None
    base_yaw_deg = None
    base_pitch_deg = None

    for image_file in sorted(os.listdir(input_path)):
        if not image_file.lower().endswith('.png'):
            continue

        image_path = os.path.join(input_path, image_file)
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"Warning: failed to read {image_path}")
            detect_fail_list.append(image_path)
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = detector.detect(mp_image)

        if not detection_result.face_blendshapes:
            print(f"Warning: no face detected in {image_path}")
            detect_fail_list.append(image_path)
            continue
        
        not_frontal = False

        try:
            landmarks = detection_result.face_landmarks[0]  # 468 landmarks
            
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            nose_tip = landmarks[1]
            mouth_bottom = landmarks[17]

            eye_center_x = (left_eye.x + right_eye.x) / 2
            eye_center_z = (left_eye.z + right_eye.z) / 2

            yaw_rad = np.arctan2(nose_tip.x - eye_center_x, nose_tip.z - eye_center_z)
            yaw_deg = np.degrees(yaw_rad)

            pitch_rad = np.arctan2(nose_tip.y - mouth_bottom.y, nose_tip.z - mouth_bottom.z)
            pitch_deg = np.degrees(pitch_rad)
            

            if base_landmarks is None:
                base_yaw_deg = yaw_deg
                base_pitch_deg = pitch_deg
                base_landmarks = landmarks
                print(f"📌 First frame baseline set in {image_path}: yaw={base_yaw_deg:.2f}, pitch={base_pitch_deg:.2f}")


            yaw_diff = (yaw_deg - base_yaw_deg + 180) % 360 - 180
            pitch_diff = (pitch_deg - base_pitch_deg + 180) % 360 - 180

            if yaw_diff > 20:
                print(f"⚠️ Non-frontal face detected in {image_path} (yaw: {yaw_deg:.2f}, pitch: {pitch_deg:.2f})")
                detect_fail_list.append(image_path)
                not_frontal = True

        except Exception as e:
            print(f"[!] Failed to compute yaw/pitch for {image_path}: {e}")
            detect_fail_list.append(image_path)
            not_frontal = True

        if not_frontal:
            continue

        blendshapes = detection_result.face_blendshapes[0]
        frame_bs_param_scores = np.array([b.score for b in blendshapes])

        if bs_param_names is None:
            smoothed_frame_bs_param_scores = frame_bs_param_scores.copy()
        else:
            smoothed_frame_bs_param_scores = SMOOTHING_FACTOR*frame_bs_param_scores + (1-SMOOTHING_FACTOR)*prev_bs_param_scores
            smoothed_frame_bs_param_scores = np.clip(0,1.0, smoothed_frame_bs_param_scores)
            
        prev_bs_param_scores = smoothed_frame_bs_param_scores

        video_bs_param_scores.append(frame_bs_param_scores)
        smoothed_video_bs_param_scores.append(smoothed_frame_bs_param_scores)

        if bs_param_names is None:
            bs_param_names = [b.category_name for b in blendshapes]

    if not video_bs_param_scores:
        raise RuntimeError("No valid frames processed with face detection.")

    video_bs_param_scores = np.stack(video_bs_param_scores, axis=0)
    smoothed_video_bs_param_scores = np.stack(smoothed_video_bs_param_scores, axis=0)
    
    np.save(os.path.join(output_path, "video_bs_param_scores_filtered.npy"), video_bs_param_scores)
    # np.save(os.path.join(output_path, "video_bs_param_scores_smoothed.npy"), smoothed_video_bs_param_scores)
    # np.save(os.path.join(output_path, "bs_param_names.npy"), bs_param_names)

    print(f"Saved ARKit blendshape coefficients to {os.path.join(output_path, 'video_bs_param_scores.npy')}")

    return detect_fail_list


if __name__ == "__main__":

    dataset_path = "/root/Dataset/NeRSemble-dataset"

    paths: List[Tuple] = []


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

    for subject_id in subject_ids:
        input_path = os.path.join(dataset_path, subject_id, "ysh", "met", "input", "source")
        output_path = os.path.join(dataset_path, subject_id, "ysh", "arkit")

        # input_path = os.path.join(dataset_path, subject_id, "decoder", "metrical-tracker", "input", "source")
        # output_path = os.path.join(dataset_path, subject_id, "decoder", "arkit")

        assert os.path.isdir(input_path)

        paths.append((input_path, output_path))
        

    print(f"#Paths: {len(paths)}")
    assert len(paths) == 10

    with Pool(processes=10) as pool:
        results = pool.map(arkit_extract, paths)

    save_detection_failures(results, output_csv="detection_failures_nersemble_filtered.csv")
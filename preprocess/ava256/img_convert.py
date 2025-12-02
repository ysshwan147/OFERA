import pillow_avif
from PIL import Image
import os
from multiprocessing import Pool, cpu_count
from typing import List, Tuple


def convert_avif_to_png(paths: Tuple[str, str, str]):
    avif_path, target_img_path, target_mica_path = paths

    try:
        with Image.open(avif_path) as img:
            img = img.convert('RGB')
            img.save(target_img_path)
            img.save(target_mica_path)

        print(f"[✓] Converted: {avif_path}")
    except Exception as e:
        print(f"[✗] Failed: {avif_path} - {e}")



if __name__ == "__main__":
    dataset_path = "/root/Dataset/ava-256-dataset"

    target_cams = ['401168']

    paths: List[Tuple] = []

    for subject_id in sorted(os.listdir(dataset_path)):
        decoder_image_path = os.path.join(dataset_path, subject_id, "decoder", "image")

        if subject_id != "20230410--1017--GVM433":
            continue
        
        if not os.path.isdir(decoder_image_path):
            continue

        for cam_dir in sorted(os.listdir(decoder_image_path)):
            cam_path = os.path.join(decoder_image_path, cam_dir)
            
            if not os.path.isdir(cam_path):
                continue

            if cam_dir.startswith('cam'):
                cam_name = cam_dir[3:]

                if cam_name not in target_cams:
                    continue

                image_png_path = os.path.join(dataset_path, subject_id, 'decoder', 'image_png', cam_dir)
                os.makedirs(image_png_path, exist_ok=True)

                mica_input_path = os.path.join(dataset_path, subject_id, 'decoder', 'metrical-tracker', 'input', 'source')
                os.makedirs(mica_input_path, exist_ok=True)

                for img in sorted(os.listdir(cam_path)):
                    img_path = os.path.join(cam_path, img)

                    target_img = img.replace('.avif', '.png')
                    target_img_path = os.path.join(image_png_path, target_img)

                    mica_input_img_path = os.path.join(mica_input_path, target_img)

                    paths.append((img_path, target_img_path, mica_input_img_path))

    print(f"#Paths: {len(paths)}")

    with Pool(processes=1) as pool:
        pool.map(convert_avif_to_png, paths)



import socket
import json
import struct
import io
from PIL import Image
import time
import numpy as np
import os

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2


def receive_blendshape_and_image(conn):
    import struct
    import numpy as np
    from PIL import Image
    import io

    # 1. Blendshape, Image 길이 읽기
    blendshape_len_buf = conn.recv(4)
    if len(blendshape_len_buf) < 4:
        print("❌ Failed to receive length prefix.")
        return None, None

    blendshape_len = struct.unpack('>I', blendshape_len_buf)[0]

    image_len_buf = conn.recv(4)
    if len(image_len_buf) < 4:
        print("❌ Failed to receive length prefix.")
        return None, None

    image_len = struct.unpack('>I', image_len_buf)[0]

    # 2. Blendshape 바이트 읽기
    blendshape_bytes = b''
    while len(blendshape_bytes) < blendshape_len:
        packet = conn.recv(blendshape_len - len(blendshape_bytes))
        if not packet:
            print("❌ Blendshape data receive failed.")
            return None, None
        blendshape_bytes += packet

    # 3. Image 바이트 읽기
    image_bytes = b''
    while len(image_bytes) < image_len:
        packet = conn.recv(image_len - len(image_bytes))
        if not packet:
            print("❌ Image data receive failed.")
            return None, None
        image_bytes += packet

    # 4. Blendshape float32 복원
    try:
        blendshape = np.frombuffer(blendshape_bytes, dtype=np.float32)
    except Exception as e:
        print(f"❌ Error decoding blendshape: {e}")
        blendshape = None

    # 5. Image 디코딩
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)
    except Exception as e:
        print(f"❌ Error decoding image: {e}")
        image_np = None

    return blendshape, image_np


def send_message(conn, message):
    msglen = struct.pack('>I', len(message))
    conn.sendall(msglen + message)



def run_server(host='0.0.0.0', port=50051):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(1)

    print(f"✅ Server listening on {host}:{port}")

    dummy_payload = np.zeros((1, 1), dtype=np.float32).tobytes()

    base_options = python.BaseOptions(model_asset_path='weights/face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=False,
                                            num_faces=1)
    mediapipe_detector = vision.FaceLandmarker.create_from_options(options)

    all_blend_scores = []
    meta_blend_scores = []

    SMOOTHING_FACTOR = 1
    prev_blend_scores = np.zeros([51], dtype=np.float32)

    while True:
        print("⏳ Waiting for Unity client to connect...")
        conn, addr = server_socket.accept()
        print(f"🎮 Unity connected from {addr}")

        try:
            while True:
                meta_blendshape, image_np = receive_blendshape_and_image(conn)
                if meta_blendshape is None or image_np is None:
                    print("⚠️ Disconnected.")
                    break

                meta_blend_scores.append(meta_blendshape)

                # Inference 처리
                try:
                    image = mp.Image(
                        image_format=mp.ImageFormat.SRGB, data=image_np
                    )

                    detection_result = mediapipe_detector.detect(image)

                    blend_scores = None

                    if len(detection_result.face_blendshapes) == 0:
                        blend_scores = prev_blend_scores
                    else:
                        # Post-process mediapipe face blendshape scores
                        blend_scores = detection_result.face_blendshapes[0]
                        blend_scores = np.array(list(map(lambda l: l.score, blend_scores)), dtype=np.float32)

                        blend_scores = blend_scores[1:]

                        # Smoothing blendshape scores
                        blend_scores = SMOOTHING_FACTOR*blend_scores + (1-SMOOTHING_FACTOR)*prev_blend_scores
                        blend_scores = np.clip(0,1.0, blend_scores)
                        prev_blend_scores = blend_scores
                    all_blend_scores.append(blend_scores)
                    
                except Exception as e:
                    print(f"❌ Error during inference: {e}")
                    send_message(conn, dummy_payload)
                    continue

                # 결과 전송
                send_message(conn, dummy_payload)

        finally:
            print("Close connection.")
            conn.close()

            subject_name = "sys"

            # Save the blendshape scores to a file
            if all_blend_scores:
                np.save(os.path.join('mp_npy_dir', f'blendshape_scores_{subject_name}.npy'), np.array(all_blend_scores, dtype=np.float32))
                print("✅ Blendshape scores saved to blendshape_scores.npy")
            else:
                print("⚠️ No blendshape scores to save.")

            if meta_blend_scores:
                np.save(os.path.join('meta_npy_dir', f'blendshape_scores_{subject_name}.npy'), np.array(meta_blend_scores, dtype=np.float32))
                print("✅ Blendshape scores saved to meta_blendshape_scores.npy")
            else:
                print("⚠️ No meta blendshape scores to save.")

            break


if __name__ == "__main__":
    run_server(host="0.0.0.0", port=50051)

import socket
import json
import struct
import time
import numpy as np

from inference import Infer

def receive_message(conn):
    # 먼저 4바이트 길이 받기
    raw_msglen = conn.recv(4)

    if not raw_msglen:
        return None

    msglen = struct.unpack('>I', raw_msglen)[0]

    # 메시지 본문 받기
    buf = b''
    while len(buf) < msglen:
        packet = conn.recv(msglen - len(buf))
        if not packet:
            return None
        buf += packet

    # 컨트롤 메시지일 수도, blendshape 딕셔너리일 수도 있음
    try:
        return json.loads(buf.decode('utf-8'))
    except Exception as e:
        # 혹시 바이너리일 경우(호환 대비) None 반환
        print(f"⚠️ Failed to decode JSON: {e}")
        return None


def send_message(conn, message):
    msglen = struct.pack('>I', len(message))
    conn.sendall(msglen + message)


def make_init_payload(inferencer):
    xyz = inferencer.xyz.detach().cpu().numpy()
    rotation = inferencer.rotation.detach().cpu().numpy()
    scale = inferencer.scale.detach().cpu().numpy()
    opacities = inferencer.opacity.detach().cpu().numpy()
    f_dc = inferencer.features_dc.detach().flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = inferencer.features_rest.detach().flatten(start_dim=1).contiguous().cpu().numpy()
    sh = np.concatenate([f_dc, f_rest], axis=1)

    expr = inferencer.flame_exp_param.detach().cpu().numpy().astype(np.float32).tobytes()
    jaw_pose = inferencer.flame_pose_param[6:9].detach().cpu().numpy().astype(np.float32).tobytes()
    eyes_pose = inferencer.flame_pose_param[9:15].detach().cpu().numpy().astype(np.float32).tobytes()

    init_payload = np.concatenate([
        xyz, rotation, scale, opacities, sh
    ], axis=1).astype(np.float32).tobytes()
    init_payload = expr + jaw_pose + eyes_pose + init_payload

    dummy_payload = np.concatenate([
        xyz, rotation, scale
    ], axis=1).astype(np.float32).tobytes()
    dummy_payload = expr + jaw_pose + eyes_pose + dummy_payload

    return init_payload, dummy_payload


def run_server(host='0.0.0.0', port=50051):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(1)

    print(f"✅ Server listening on {host}:{port}")

    models = {
        "ours": None,
        "non_bda": None,
        "non_mia": None,
        "non_mia_non_bda": None,
        "mp2flame": None,
        "mat": None,
    }

    weight_path = "weights/2025-08-09_17-26-29"
    
    subject_name = "ysh"

    for name in models.keys():
        if name == "ours": # Condition 5 (ours)
            models[name] = Infer(weights_path=weight_path, subject_name=subject_name, infer_type="mlp", blendshape_alignment=True, mapper_integrated=True)

        elif name == "non_bda": # Condition 3
            models[name] = Infer(weights_path=weight_path, subject_name=subject_name, infer_type="mlp", blendshape_alignment=False, mapper_integrated=True)

        elif name == "non_mia": # Condition 4
            models[name] = Infer(weights_path=weight_path, subject_name=subject_name, infer_type="mlp", blendshape_alignment=True, mapper_integrated=False)

        elif name == "non_mia_non_bda":
            models[name] = Infer(weights_path=weight_path, subject_name=subject_name, infer_type="mlp", blendshape_alignment=False, mapper_integrated=False)

        elif name == "mp2flame": # Condition 2
            models[name] = Infer(weights_path=weight_path, subject_name=subject_name, infer_type="mp2flame", blendshape_alignment=False, mapper_integrated=False)

        elif name == "mat": # Condition 1
            models[name] = Infer(weights_path=weight_path, subject_name=subject_name, infer_type="mat", blendshape_alignment=False, mapper_integrated=False)

    # 각 모델별 init/dummy 바이너리 캐시
    cached_init = {}
    cached_dummy = {}
    for name, infer in models.items():
        init_b, dummy_b = make_init_payload(infer)
        cached_init[name] = init_b
        cached_dummy[name] = dummy_b

    current_model = "ours"  # 기본값

    while True:
        print("⏳ Waiting for Unity client to connect...")
        conn, addr = server_socket.accept()
        print(f"🎮 Unity connected from {addr}")

        # send_message(conn, cached_init[current_model])

        try:
            while True:
                msg = receive_message(conn)
                if msg is None:
                    print("⚠️ Disconnected.")
                    break

                # 컨트롤 메시지 처리
                if isinstance(msg, dict) and "cmd" in msg:
                    # print(f"🔧 Control command received: {msg}")
                    cmd = msg.get("cmd")
                    if cmd == "select":
                        name = msg.get("model", "")
                        no_init = msg.get("no_init", False)
                        if name in models:
                            current_model = name
                            # 새 모델 init 바로 전송
                            if not no_init:
                                send_message(conn, cached_init[current_model])
                            print(f"🔁 switched to model: {current_model}")
                        else:
                            # 없는 모델이면 목록 반환(텍스트/json)
                            send_message(conn, json.dumps({
                                "error":"unknown_model",
                                "available": list(models.keys())
                            }).encode("utf-8"))
                    elif cmd == "list":
                        send_message(conn, json.dumps({"models": list(models.keys())}).encode("utf-8"))
                    else:
                        send_message(conn, json.dumps({"error":"unknown_cmd"}).encode("utf-8"))
                    continue

                start_time = time.time()

                # 일반 추론(블렌드셰이프 딕셔너리)
                try:
                    payload = models[current_model].infer(msg)  # infer가 bytes 반환
                    send_message(conn, payload)
                except Exception as e:
                    print(f"❌ Error during inference: {e}")
                    # 실패 시 현재 모델의 dummy 전송
                    send_message(conn, cached_dummy[current_model])
                    continue

                end_time = time.time()

                elapsed_ms = (end_time - start_time) * 1000  # ms 단위
                # print(f"✅ Inference time: {elapsed_ms:.3f} ms")

        finally:
            print("Close connection.")
            conn.close()
            break


if __name__ == "__main__":
    run_server()

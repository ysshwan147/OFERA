import socket
import json
import struct
import time
import mmap
import numpy as np

from inference import Infer

# ====== 기존 server_fate.py 의 헬퍼 그대로 사용 ======
def receive_message(conn):
    # 먼저 4바이트 길이 받기 (big-endian)
    raw_msglen = conn.recv(4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # 메시지 본문 받기
    data = b''
    while len(data) < msglen:
        packet = conn.recv(msglen - len(data))
        if not packet:
            return None
        data += packet
    return json.loads(data.decode('utf-8'))


def send_message(conn, message: bytes):
    # message: bytes
    msglen = struct.pack('>I', len(message))
    conn.sendall(msglen + message)


# ====== Shared Memory 관련 설정 ======
SHM_NAME = r"Local\GaussianSharedBuffer"  # Unity와 동일해야 함


def create_shared_memory_buffer(flame_dim: int, gaussian_count: int) -> mmap.mmap:
    """
    Unity 쪽 ARKit2FLAMEGaussianAvatarsMMap.cs 의 TryReadFromSharedMemory()
    레이아웃에 맞춰 shared memory 크기를 계산하고 생성합니다.
    layout:
        int32 readyFlag      (4)
        int32 count          (4)
        float expr[flame_dim]
        float jaw[3]
        float eyes[6]
        float pos[count * 3]
        float rot[count * 4]
        float scale[count * 3]
    """
    header_bytes = 8  # readyFlag + count

    float_count = (
        flame_dim +      # expr
        3 +              # jaw
        6 +              # eyes
        gaussian_count * 3 +  # pos
        gaussian_count * 4 +  # rot
        gaussian_count * 3    # scale
    )
    total_bytes = header_bytes + float_count * 4  # float32 = 4 bytes

    # 약간의 여유를 주고 싶다면 total_bytes에 + 여유값 해도 됨.
    shm = mmap.mmap(-1, total_bytes, tagname=SHM_NAME)
    print(f"[SharedMemory] Created mmap '{SHM_NAME}' with size {total_bytes / (1024*1024):.2f} MB")

    # 초기 readyFlag = 0, count = gaussian_count 로 세팅
    shm.seek(0)
    shm.write(struct.pack('i', 0))  # readyFlag
    shm.write(struct.pack('i', gaussian_count))  # count

    return shm


def write_to_shared_memory(
    shm: mmap.mmap,
    expr: np.ndarray,
    jaw: np.ndarray,
    eyes: np.ndarray,
    pos: np.ndarray,
    rot: np.ndarray,
    scale: np.ndarray,
):
    """
    expr, jaw, eyes, pos, rot, scale 를 shared memory에 기록.
    Unity에서는 readyFlag == 1 일 때만 읽고, 읽은 후 0으로 되돌립니다.
    여기서는:
        readyFlag = 0으로 clear → 데이터 쓰기 → readyFlag = 1
    """
    # 안전을 위해 dtype 보정
    expr = np.asarray(expr, dtype=np.float32)
    jaw = np.asarray(jaw, dtype=np.float32)
    eyes = np.asarray(eyes, dtype=np.float32)
    pos = np.asarray(pos, dtype=np.float32)
    rot = np.asarray(rot, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)

    gaussian_count = pos.shape[0]

    # 1) readyFlag = 0 (Unity는 이 상태에서는 읽지 않음)
    shm.seek(0)
    shm.write(struct.pack('i', 0))      # readyFlag
    shm.write(struct.pack('i', gaussian_count))  # count (필요시 업데이트)

    # 2) float 데이터 순서대로 쓰기
    offset = 8  # readyFlag(4) + count(4)

    def write_array(arr: np.ndarray):
        nonlocal offset
        shm.seek(offset)
        shm.write(arr.tobytes())
        offset += arr.nbytes

    write_array(expr.reshape(-1))
    write_array(jaw.reshape(-1))
    write_array(eyes.reshape(-1))
    write_array(pos.reshape(-1))
    write_array(rot.reshape(-1))
    write_array(scale.reshape(-1))

    # 3) readyFlag = 1 (Unity가 읽을 수 있음)
    shm.seek(0)
    shm.write(struct.pack('i', 1))


def parse_payload_to_components(payload: bytes, flame_dim: int):
    """
    inferencer.infer(data) 가 반환하는 payload(bytes)를
    expr / jaw / eyes / pos / rot / scale 로 나눕니다.

    payload float32 layout (기존 Unity DataParsing과 동일 가정):
        expr[flame_dim] + jaw[3] + eyes[6] + gaussian[...]
    gaussian 부분은:
        [x,y,z, qx,qy,qz,qw, sx,sy,sz] 가 각 가우시안마다 반복
    """
    arr = np.frombuffer(payload, dtype=np.float32)

    # 앞부분: expr, jaw, eyes
    expr = arr[:flame_dim]
    jaw = arr[flame_dim:flame_dim + 3]
    eyes = arr[flame_dim + 3: flame_dim + 3 + 6]

    gaussian_arr = arr[flame_dim + 3 + 6:]

    if gaussian_arr.size % 10 != 0:
        raise ValueError(f"Gaussian part length {gaussian_arr.size} is not divisible by 10")

    g = gaussian_arr.reshape(-1, 10)
    pos = g[:, 0:3]
    rot = g[:, 3:7]
    scale = g[:, 7:10]

    return expr, jaw, eyes, pos, rot, scale


# ====== 메인 서버 ======
def run_server_mmap(host='0.0.0.0', port=50051):
    # 1) TCP 서버 준비 (ARKit JSON + init payload 용)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(1)

    print(f"✅ Server (MMAP) listening on {host}:{port}")

    # 2) Fate/FLAME/Gaussian 모델 로드 (기존 server_fate.py와 동일)
    weight_path = "weights/mlp"
    subject_name = "ysh"
    model_name = "ours"

    if model_name == "ours":  # Condition 5
        inferencer = Infer(weights_path=weight_path, subject_name=subject_name, infer_type="mlp", blendshape_alignment=True, mapper_integrated=True)
    elif model_name == "non_bda":  # Condition 3
        inferencer = Infer(weights_path=weight_path, subject_name=subject_name, infer_type="mlp", blendshape_alignment=False, mapper_integrated=True)
    elif model_name == "non_mia":  # Condition 4
        inferencer = Infer(weights_path=weight_path, subject_name=subject_name, infer_type="mlp", blendshape_alignment=True, mapper_integrated=False)
    elif model_name == "non_mia_non_bda":
        inferencer = Infer(weights_path=weight_path, subject_name=subject_name, infer_type="mlp", blendshape_alignment=False, mapper_integrated=False)
    elif model_name == "mp2flame":  # Condition 2
        inferencer = Infer(weights_path=weight_path, subject_name=subject_name, infer_type="mp2flame", blendshape_alignment=False, mapper_integrated=False)
    elif model_name == "mat":  # Condition 1
        inferencer = Infer(weights_path=weight_path, subject_name=subject_name, infer_type="mat", blendshape_alignment=False, mapper_integrated=False)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # 3) 초기 Gaussian / SH / FLAME 파라미터 준비 (init payload)
    xyz = inferencer.xyz.detach().cpu().numpy()          # (N,3)
    rotation = inferencer.rotation.detach().cpu().numpy()  # (N,4)
    scale = inferencer.scale.detach().cpu().numpy()      # (N,3)
    opacities = inferencer.opacity.detach().cpu().numpy()  # (N,1) 또는 (N,)
    f_dc = inferencer.features_dc.detach().flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = inferencer.features_rest.detach().flatten(start_dim=1).contiguous().cpu().numpy()
    sh = np.concatenate([f_dc, f_rest], axis=1)

    # FLAME 초기 파라미터
    flame_exp = inferencer.flame_exp_param[:50].detach().cpu().numpy().astype(np.float32)
    flame_pose = inferencer.flame_pose_param.detach().cpu().numpy().astype(np.float32)
    # jaw_pose: pose[6:9], eyes_pose: pose[9:15]
    jaw_pose_init = flame_pose[6:9]
    eyes_pose_init = flame_pose[9:15]

    flame_dim = flame_exp.shape[-1]  # 예: 100
    gaussian_count = xyz.shape[0]

    print(f"[Model] flame_dim={flame_dim}, gaussian_count={gaussian_count}")

    # init payload: expr + jaw + eyes + [xyz,rot,scale,opacity,sh]
    init_concat = np.concatenate(
        [xyz, rotation, scale, opacities.reshape(gaussian_count, -1), sh],
        axis=1
    ).astype(np.float32)
    init_payload = b"".join([
        flame_exp.astype(np.float32).tobytes(),
        jaw_pose_init.astype(np.float32).tobytes(),
        eyes_pose_init.astype(np.float32).tobytes(),
        init_concat.tobytes()
    ])

    # dummy payload: expr + jaw + eyes + [xyz,rot,scale]
    dummy_concat = np.concatenate([xyz, rotation, scale], axis=1).astype(np.float32)
    dummy_payload = b"".join([
        flame_exp.astype(np.float32).tobytes(),
        jaw_pose_init.astype(np.float32).tobytes(),
        eyes_pose_init.astype(np.float32).tobytes(),
        dummy_concat.tobytes()
    ])

    # 4) Shared Memory 생성
    shm = create_shared_memory_buffer(flame_dim=flame_dim, gaussian_count=gaussian_count)

    # 처음 한번, 초기 상태를 shared memory에도 써두자 (원하면 생략 가능)
    expr0, jaw0, eyes0, pos0, rot0, scale0 = parse_payload_to_components(dummy_payload, flame_dim)
    write_to_shared_memory(shm, expr0, jaw0, eyes0, pos0, rot0, scale0)

    # 5) Unity 클라이언트 연결 처리
    while True:
        print("⏳ Waiting for Unity client to connect...")
        conn, addr = server_socket.accept()
        print(f"🎮 Unity connected from {addr}")

        # Unity의 InitReceiveResponse() 에서 읽을 initial payload 전송
        send_message(conn, init_payload)
        print("📨 Sent initial payload (expr+jaw+eyes+full gaussians+opacity+sh)")

        try:
            while True:
                data = receive_message(conn)
                if data is None:
                    print("⚠️ Disconnected.")
                    break

                # data: {blendshape_name: weight, ...}
                start_time = time.time()
                try:
                    # inferencer.infer(data) 는 float32 bytes를 반환한다고 가정
                    payload_bytes = inferencer.infer(data)
                except Exception as e:
                    print(f"❌ Error during inference: {e}")
                    # 실패 시 dummy_payload로 fallback
                    payload_bytes = dummy_payload

                # payload를 expr/jaw/eyes/pos/rot/scale 로 파싱
                try:
                    expr, jaw, eyes, pos, rot, scale = parse_payload_to_components(
                        payload_bytes, flame_dim=flame_dim
                    )
                except Exception as e:
                    print(f"❌ Error parsing payload: {e}")
                    # 파싱 실패 시 초기 dummy 값 사용
                    expr, jaw, eyes, pos, rot, scale = expr0, jaw0, eyes0, pos0, rot0, scale0

                # Shared memory에 결과 쓰기
                write_to_shared_memory(shm, expr, jaw, eyes, pos, rot, scale)

                elapsed_ms = (time.time() - start_time) * 1000.0
                # print(f"✅ Inference + shm write time: {elapsed_ms:.3f} ms")

                # ⚠ 여기서는 더 이상 TCP로 float 배열을 보내지 않습니다.
                # Unity 쪽 ARKit2FLAMEGaussianAvatarsMMap.cs 도
                # SendBlendshapesJsonOnly() 로만 보내고 응답은 읽지 않습니다.

        finally:
            print("Close connection.")
            conn.close()
            # Unity 1개만 쓴다는 가정이면 break로 서버 종료,
            # 여러 번 연결을 허용하려면 break 없이 다음 루프로 가면 됨.
            break

    shm.close()
    server_socket.close()


if __name__ == "__main__":
    run_server_mmap(host="0.0.0.0", port=50051)

import numpy as np
import torch
import os
import sys
import glob
import joblib

# for mlp
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle
from arkit2flame_model import ARKitToFLAMENet

# for mp2flame
sys.path.append(os.path.abspath("mediapipe-blendshapes-to-flame"))
from mp_2_flame import MP_2_FLAME

# for fateavatar
# sys.path.append(os.path.abspath("FateAvatar"))
ROOT = os.path.dirname(os.path.abspath(__file__))
FATE_ROOT = os.path.join(ROOT, "FateAvatar")
if FATE_ROOT not in sys.path:
    sys.path.insert(0, FATE_ROOT)

from train.loader        import Loader
from train.callbacks     import ModelCallbacks

from common  import      load_config
from common  import      load_identity_info
from common  import      construct_model


class Infer:
    def __init__(self, weights_path="weights", subject_name="", infer_type="mlp", blendshape_alignment=False, mapper_integrated=False):

        self.infer_type = infer_type
        self.blendshape_alignment = blendshape_alignment
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        bs_names_path = "./weights/bs_param_names.npy"
        self.bs_names = np.load(bs_names_path)

        assert len(self.bs_names) == 51, f"Expected 51 ARKit blendshapes, got {len(self.bs_names)}"


        print("Loading model...")
        # for mat
        if infer_type == "mat":
            bs_mat_path = ".\weights\mat_final.npy"
            self.bs_mat = np.load(bs_mat_path)

        # for mp2flame
        elif infer_type == "mp2flame":
            self.mp2flame = MP_2_FLAME(mappings_path='./mediapipe-blendshapes-to-flame/mappings', index=1)
        
        # for mlp
        elif infer_type == "mlp":
            self.model = ARKitToFLAMENet(hidden_dim=128, num_blocks=4).to(self.device)
            self.model.load_state_dict(torch.load(f"{weights_path}/best_mlp_model.pth"))
            self.model.eval()

            stats_path = f"{weights_path}/stats.npz"
            stats = np.load(stats_path)

            self.input_mean = stats["input_mean"]
            self.input_std = stats["input_std"]
            self.expr_mean = stats["expr_mean"]
            self.expr_std = stats["expr_std"]
            self.jaw_mean = stats["jaw_mean"]
            self.jaw_std = stats["jaw_std"]
            self.eyes_mean = stats["eyes_mean"]
            self.eyes_std = stats["eyes_std"]


        # bs_param_names.npy 기준(이름 및 순서) → unity stream data 이름 매핑
        self.mapping_dict = {
            'browDownLeft': 'bls_head.browDownLeft',
            'browDownRight': 'bls_head.browDownRight',
            'browInnerUp': 'bls_head.browInnerUp',
            'browOuterUpLeft': 'bls_head.browOuterUpLeft',
            'browOuterUpRight': 'bls_head.browOuterUpRight',
            'cheekPuff': 'bls_head.cheekPuff',
            'cheekSquintLeft': 'bls_head.cheekSquintLeft',
            'cheekSquintRight': 'bls_head.cheekSquintRight',
            'eyeBlinkLeft': 'bls_head.eyeBlinkLeft',
            'eyeBlinkRight': 'bls_head.eyeBlinkRight',
            'eyeLookDownLeft': 'bls_head.eyeLookDownLeft',
            'eyeLookDownRight': 'bls_head.eyeLookDownRight',
            'eyeLookInLeft': 'bls_head.eyeLookInLeft',
            'eyeLookInRight': 'bls_head.eyeLookInRight',
            'eyeLookOutLeft': 'bls_head.eyeLookOutLeft',
            'eyeLookOutRight': 'bls_head.eyeLookOutRight',
            'eyeLookUpLeft': 'bls_head.eyeLookUpLeft',
            'eyeLookUpRight': 'bls_head.eyeLookUpRight',
            'eyeSquintLeft': 'bls_head.eyeSquintLeft',
            'eyeSquintRight': 'bls_head.eyeSquintRight',
            'eyeWideLeft': 'bls_head.eyeWideLeft',
            'eyeWideRight': 'bls_head.eyeWideRight',
            'jawForward': 'bls_head.jawForward',
            'jawLeft': 'bls_head.jawLeft',
            'jawOpen': 'bls_head.jawOpen',
            'jawRight': 'bls_head.jawRight',
            'mouthClose': 'bls_head.mouthClose',
            'mouthDimpleLeft': 'bls_head.mouthDimpleLeft',
            'mouthDimpleRight': 'bls_head.mouthDimpleRight',
            'mouthFrownLeft': 'bls_head.mouthFrownLeft',
            'mouthFrownRight': 'bls_head.mouthFrownRight',
            'mouthFunnel': 'bls_head.mouthFunnel',
            'mouthLeft': 'bls_head.mouthLeft',
            'mouthLowerDownLeft': 'bls_head.mouthLowerDownLeft',
            'mouthLowerDownRight': 'bls_head.mouthLowerDownRight',
            'mouthPressLeft': 'bls_head.mouthPressLeft',
            'mouthPressRight': 'bls_head.mouthPressRight',
            'mouthPucker': 'bls_head.mouthPucker',
            'mouthRight': 'bls_head.mouthRight',
            'mouthRollLower': 'bls_head.mouthRollLower',
            'mouthRollUpper': 'bls_head.mouthRollUpper',
            'mouthShrugLower': 'bls_head.mouthShrugLower',
            'mouthShrugUpper': 'bls_head.mouthShrugUpper',
            'mouthSmileLeft': 'bls_head.mouthSmileLeft',
            'mouthSmileRight': 'bls_head.mouthSmileRight',
            'mouthStretchLeft': 'bls_head.mouthStretchLeft',
            'mouthStretchRight': 'bls_head.mouthStretchRight',
            'mouthUpperUpLeft': 'bls_head.mouthUpperUpLeft',
            'mouthUpperUpRight': 'bls_head.mouthUpperUpRight',
            'noseSneerLeft': 'bls_head.noseSneerLeft',
            'noseSneerRight': 'bls_head.noseSneerRight'
        }

        self.mapping_dict_flipped = {
            'browDownLeft': 'bls_head.browDownRight',
            'browDownRight': 'bls_head.browDownLeft',
            'browInnerUp': 'bls_head.browInnerUp',  # L/R 없음
            'browOuterUpLeft': 'bls_head.browOuterUpRight',
            'browOuterUpRight': 'bls_head.browOuterUpLeft',
            'cheekPuff': 'bls_head.cheekPuff',  # L/R 없음
            'cheekSquintLeft': 'bls_head.cheekSquintRight',
            'cheekSquintRight': 'bls_head.cheekSquintLeft',
            'eyeBlinkLeft': 'bls_head.eyeBlinkRight',
            'eyeBlinkRight': 'bls_head.eyeBlinkLeft',
            'eyeLookDownLeft': 'bls_head.eyeLookDownRight',
            'eyeLookDownRight': 'bls_head.eyeLookDownLeft',
            'eyeLookInLeft': 'bls_head.eyeLookInRight',
            'eyeLookInRight': 'bls_head.eyeLookInLeft',
            'eyeLookOutLeft': 'bls_head.eyeLookOutRight',
            'eyeLookOutRight': 'bls_head.eyeLookOutLeft',
            'eyeLookUpLeft': 'bls_head.eyeLookUpRight',
            'eyeLookUpRight': 'bls_head.eyeLookUpLeft',
            'eyeSquintLeft': 'bls_head.eyeSquintRight',
            'eyeSquintRight': 'bls_head.eyeSquintLeft',
            'eyeWideLeft': 'bls_head.eyeWideRight',
            'eyeWideRight': 'bls_head.eyeWideLeft',
            'jawForward': 'bls_head.jawForward',  # L/R 없음
            'jawLeft': 'bls_head.jawRight',
            'jawOpen': 'bls_head.jawOpen',  # L/R 없음
            'jawRight': 'bls_head.jawLeft',
            'mouthClose': 'bls_head.mouthClose',  # L/R 없음
            'mouthDimpleLeft': 'bls_head.mouthDimpleRight',
            'mouthDimpleRight': 'bls_head.mouthDimpleLeft',
            'mouthFrownLeft': 'bls_head.mouthFrownRight',
            'mouthFrownRight': 'bls_head.mouthFrownLeft',
            'mouthFunnel': 'bls_head.mouthFunnel',  # L/R 없음
            'mouthLeft': 'bls_head.mouthRight',
            'mouthLowerDownLeft': 'bls_head.mouthLowerDownRight',
            'mouthLowerDownRight': 'bls_head.mouthLowerDownLeft',
            'mouthPressLeft': 'bls_head.mouthPressRight',
            'mouthPressRight': 'bls_head.mouthPressLeft',
            'mouthPucker': 'bls_head.mouthPucker',  # L/R 없음
            'mouthRight': 'bls_head.mouthLeft',
            'mouthRollLower': 'bls_head.mouthRollLower',  # L/R 없음
            'mouthRollUpper': 'bls_head.mouthRollUpper',  # L/R 없음
            'mouthShrugLower': 'bls_head.mouthShrugLower',  # L/R 없음
            'mouthShrugUpper': 'bls_head.mouthShrugUpper',  # L/R 없음
            'mouthSmileLeft': 'bls_head.mouthSmileRight',
            'mouthSmileRight': 'bls_head.mouthSmileLeft',
            'mouthStretchLeft': 'bls_head.mouthStretchRight',
            'mouthStretchRight': 'bls_head.mouthStretchLeft',
            'mouthUpperUpLeft': 'bls_head.mouthUpperUpRight',
            'mouthUpperUpRight': 'bls_head.mouthUpperUpLeft',
            'noseSneerLeft': 'bls_head.noseSneerRight',
            'noseSneerRight': 'bls_head.noseSneerLeft'
        }

        # init
        self.cwd = os.getcwd()
        os.chdir("FateAvatar")

        self.config = "./config/fateavatar.yaml"
        self.model_name = "FateAvatar"
        self.name = f"fateavatar_insta_{subject_name}"
        if mapper_integrated:
            self.workspace = f"./workspace/insta/{subject_name}/{subject_name}_mlp"
        else:
            self.workspace = f"./workspace/insta/{subject_name}/{subject_name}"
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.seed = 42
        self.bg_color = 'white'
        self.ckpt_path = f"{self.workspace}/checkpoints_fullhead/fullhead.pth"
        self.use_full_head_resume = True
        self.use_baked_resume = False

        # ----------------------------  override config ---------------------------- #
        self.cfg = load_config(
            self.config,
            overrides={
                "name":         self.name,
                "workspace":    self.workspace,
                "bg_color":     self.bg_color
                }
            )

        # ---------------------------- load identity info ---------------------------- #
        identity_dict = load_identity_info(self, self.cfg)

        # ---------------------------------- model ----------------------------------- #
        avatar = construct_model(
            self,
            self.cfg.model,
            0.0,
            identity_dict = identity_dict
        )

        # -------------------------- loading trained avatar -------------------------- #
        if self.use_full_head_resume and self.use_baked_resume:
            ckpt_dirs = os.path.join(self.workspace, "baking_full_head", "checkpoints_baked")
            ckpt_list = sorted(glob.glob(f'{ckpt_dirs}/*.pth'))
            ckpt_path = ckpt_list[-1]
        
        elif self.use_full_head_resume and self.use_baked_resume is False:
            ckpt_path = os.path.join(self.workspace, "checkpoints_fullhead", "fullhead.pth")

        elif self.use_full_head_resume is False and self.use_baked_resume:
            ckpt_dirs = os.path.join(self.workspace, "baking", "checkpoints_baked")
            ckpt_list = sorted(glob.glob(f'{ckpt_dirs}/*.pth'))
            ckpt_path = ckpt_list[-1]
        else:
            ckpt_path = 'latest'
        
        if self.ckpt_path:
            ckpt_path = self.ckpt_path

        loader = Loader(
            self.name,
            self.cfg,
            avatar,
            self.device,
            workspace       = self.workspace,
            use_checkpoint  = ckpt_path
        )

        # overwrite
        self.avatar = loader.model

        self.flame_exp_param = torch.zeros(self.cfg.model.n_exp).to(self.device)
        self.flame_pose_param = torch.zeros(15).to(self.device)

        # self.xyz, self.rotation, self.scale, self.opacity, self.features_dc, self.features_rest = self.avatar.init_update_gs()
        self.xyz, self.rotation, self.scale, self.opacity, self.features_dc, self.features_rest = self.avatar.update_gs(
            self.flame_exp_param.unsqueeze(0),
            self.flame_pose_param.unsqueeze(0)
        )

        self.rotation = torch.nn.functional.normalize(self.rotation)
        self.scale = torch.exp(self.scale)
        self.opacity = torch.sigmoid(self.opacity)

        self.features_rest = torch.zeros((self.xyz.shape[0], 45), dtype=torch.float32).to(self.device)

        os.chdir(self.cwd)

        self.SMOOTHING_FACTOR = 1
        self.prev_blend_scores = np.zeros([51], dtype=np.float32)

        self.meta2mp_model = joblib.load("weights/linear_meta_to_mediapipe.pkl")


    def infer(self, arkit_stream: dict) -> dict:

        mapping_dict = {}
        if self.infer_type == "mat" or self.infer_type == "mp2flame":
            mapping_dict = self.mapping_dict
        elif self.infer_type == "mlp":
            mapping_dict = self.mapping_dict_flipped

        arkit_input_raw = np.array([arkit_stream[mapping_dict[bs]] for bs in self.bs_names], dtype=np.float32)
        arkit_input = arkit_input_raw / 100.0  # 0~100 값을 0~1 값으로 정규화

        arkit_input = self.SMOOTHING_FACTOR * arkit_input + (1-self.SMOOTHING_FACTOR) * self.prev_blend_scores
        arkit_input = np.clip(arkit_input, 0.0, 1.0)
        self.prev_blend_scores = arkit_input

        if self.blendshape_alignment:
            arkit_input = self.meta2mp_model.predict(arkit_input.reshape(1, -1)).flatten()


        if self.infer_type == "mat":
            result = np.dot(arkit_input, self.bs_mat)

            expr = result[:100].astype(np.float32)
            jaw_pose = result[100:103].astype(np.float32)
            eyes_pose = np.zeros(6, dtype=np.float32)

        elif self.infer_type == "mp2flame":
            arkit_input = np.insert(arkit_input, 0, 0.0).reshape(1, -1)  # [1, 52] neutral 추가

            # blendshape_scores is the np.array object with shape [N,52],
            # N is the number of samples, and by default N=1
            expr, jaw_pose, eyes_pose = self.mp2flame.convert(blendshape_scores=arkit_input)

            jaw_pose = jaw_pose[:, 3:]  # [1, 6] → [1, 3]

            expr = expr[0, :100].astype(np.float32)
            jaw_pose = jaw_pose[0, :3].astype(np.float32)
            eyes_pose = eyes_pose[0, :6].astype(np.float32)

        elif self.infer_type == "mlp":
            # 정규화 & 추론
            x_norm = (arkit_input - self.input_mean) / self.input_std
            x_tensor = torch.tensor(x_norm, dtype=torch.float32, device=self.device).unsqueeze(0)

            with torch.no_grad():
                out = self.model(x_tensor)
                if isinstance(out, (tuple, list)) and len(out) == 4:
                    # SEP / SHA: (expr, jaw, eyeL, eyeR)
                    expr_pred, jaw_pred, eyeL_pred, eyeR_pred = out
                    y_pred_t = torch.cat([expr_pred, jaw_pred, eyeL_pred, eyeR_pred], dim=-1)
                else:
                    # Baseline: (B, 68)
                    y_pred_t = out
                y_pred = y_pred_t.squeeze(0).detach().cpu().numpy()


            # 역정규화
            expr = np.zeros((self.cfg.model.n_exp,), dtype=np.float32)
            flame_out_dim = y_pred.shape[0] - 6 - 12
            expr[:flame_out_dim] = y_pred[:flame_out_dim] * self.expr_std + self.expr_mean
            # print(("flame_out_dim", flame_out_dim))

            jaw_pose_6d = y_pred[flame_out_dim:flame_out_dim + 6] * self.jaw_std + self.jaw_mean
            eyes_pose_6d = y_pred[flame_out_dim + 6:] * self.eyes_std + self.eyes_mean  # (12,)
            eyeL_6d = eyes_pose_6d[:6]
            eyeR_6d = eyes_pose_6d[6:]

            # NumPy → Torch 변환 (batch 차원 추가)
            jaw_pose_6d_t = torch.tensor(jaw_pose_6d, dtype=torch.float32, device=self.device).unsqueeze(0)   # (1, 6)
            eyes_6d_t = torch.tensor(np.stack([eyeL_6d, eyeR_6d], axis=0), dtype=torch.float32, device=self.device)  # (2, 6)

            jaw_rotmat = rotation_6d_to_matrix(jaw_pose_6d_t)   # (1, 3, 3)
            eyes_rotmat = rotation_6d_to_matrix(eyes_6d_t)      # (2, 3, 3)

            jaw_pose = matrix_to_axis_angle(jaw_rotmat).squeeze(0).detach().cpu().numpy()   # (3,)
            eyes_pose = matrix_to_axis_angle(eyes_rotmat).detach().cpu().numpy().reshape(-1)  # (6,)


        # FateAvatar
        self.cwd = os.getcwd()
        os.chdir("FateAvatar")

        self.flame_exp_param = torch.from_numpy(expr).to(self.device)
        self.flame_pose_param[6:9] = torch.from_numpy(jaw_pose).to(self.device)
        self.flame_pose_param[9:15] = torch.from_numpy(eyes_pose).to(self.device)


        self.xyz, self.rotation, self.scale, _, _, _ = self.avatar.update_gs(
            self.flame_exp_param.unsqueeze(0),
            self.flame_pose_param.unsqueeze(0)
        )

        self.rotation = torch.nn.functional.normalize(self.rotation)
        self.scale = torch.exp(self.scale)

        xyz = self.xyz.detach().cpu().numpy()
        rotation = self.rotation.detach().cpu().numpy()
        scale = self.scale.detach().cpu().numpy()
        
        os.chdir(self.cwd)

        jaw_pose *= 180.0 / np.pi
        eyes_pose *= 180.0 / np.pi

        expr_byte = expr[:flame_out_dim].astype(np.float32).tobytes()
        jaw_pose_byte = jaw_pose.astype(np.float32).tobytes()
        eyes_pose_byte = eyes_pose.astype(np.float32).tobytes()

        combined = np.concatenate([
            xyz, rotation, scale
        ], axis=1).astype(np.float32).tobytes()

        payload = expr_byte + jaw_pose_byte + eyes_pose_byte + combined

        return payload


import numpy as np
import glob
import os
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 🔧 설정: .npy 파일이 들어 있는 디렉토리 경로
meta_dir = r".\meta_npy_dir"         # meta_blendshape_scores.npy 파일들이 있는 디렉토리
mediapipe_dir = r".\mp_npy_dir"  # mediapipe 대응 npy 파일들이 있는 디렉토리

# 🔁 모든 .npy 파일 불러와서 누적
meta_all = []
mediapipe_all = []

for meta_path in sorted(glob.glob(os.path.join(meta_dir, "*.npy"))):
    filename = os.path.basename(meta_path)
    if filename.endswith("_shs.npy"):
        continue
    if filename.endswith("_ksy.npy"):
        continue
    if filename.endswith("_ksj.npy"):
        continue
    if filename.endswith("_shi.npy"):
        continue
    if filename.endswith("_ysh.npy"):
        continue
    # if filename.endswith("_sys.npy"):
    #     continue
    mediapipe_path = os.path.join(mediapipe_dir, filename)

    if not os.path.exists(mediapipe_path):
        print(f"⚠️ 대응 mediapipe 파일 없음: {mediapipe_path}")
        continue

    meta = np.load(meta_path)
    mediapipe = np.load(mediapipe_path)

    if meta.shape != mediapipe.shape:
        print(f"⚠️ Shape mismatch in {filename}: {meta.shape} vs {mediapipe.shape}")
        continue

    meta_all.append(meta)
    mediapipe_all.append(mediapipe)

# 🔗 전체 데이터를 하나로 합치기
meta_all = np.concatenate(meta_all, axis=0)
mediapipe_all = np.concatenate(mediapipe_all, axis=0)
print(f"✅ 총 데이터 크기: {meta_all.shape}")

# 📈 모델 학습
model_ridge = Ridge(alpha=1.0)
model_linear = LinearRegression()

model_ridge.fit(meta_all, mediapipe_all)
model_linear.fit(meta_all, mediapipe_all)

# 🎯 예측 및 평가
pred_ridge = model_ridge.predict(meta_all)
pred_linear = model_linear.predict(meta_all)

mse_ridge = mean_squared_error(mediapipe_all, pred_ridge)
r2_ridge = r2_score(mediapipe_all, pred_ridge)

mse_linear = mean_squared_error(mediapipe_all, pred_linear)
r2_linear = r2_score(mediapipe_all, pred_linear)

print(f"✅ Ridge 모델 - MSE: {mse_ridge:.6f}, R²: {r2_ridge:.4f}")
print(f"✅ Linear 모델 - MSE: {mse_linear:.6f}, R²: {r2_linear:.4f}")

# 💾 모델 저장
joblib.dump(model_ridge, "linear_meta_to_mediapipe_ridge.pkl")
print("✅ 모델 저장 완료: linear_meta_to_mediapipe_ridge.pkl")

joblib.dump(model_linear, "linear_meta_to_mediapipe.pkl")
print("✅ 모델 저장 완료: linear_meta_to_mediapipe.pkl")

import numpy as np
import zipfile, io, tempfile, os, asyncio
import pyexr
import viser
from pathlib import Path

# --- 경로 설정 ---
ROOT = Path("/home/nas5/hoiyeongjin/repos/Project/SKT/vipe/2125_2_cut1")
DEPTH_ZIP = ROOT / "depth/2125_2_cut1.zip"
POSE_PATH = ROOT / "pose/2125_2_cut1.npz"
INTR_PATH = ROOT / "intrinsics/2125_2_cut1.npz"

# --- Intrinsics & Pose 불러오기 ---
intr_data = np.load(INTR_PATH)
fx, fy, cx, cy = intr_data["data"][0]
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
print("Intrinsic matrix K:\n", K)

pose_data = np.load(POSE_PATH)
poses = pose_data["data"]  # (N, 4, 4)
print(f"Loaded {len(poses)} poses, first pose:\n", poses[0])

# --- Depth 불러오기 ---
depth_frames = []
with zipfile.ZipFile(DEPTH_ZIP, "r") as zip_ref:
    for name in sorted(zip_ref.namelist()):
        with zip_ref.open(name) as f, tempfile.NamedTemporaryFile(delete=False, suffix=".exr") as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name
        try:
            exr = pyexr.open(tmp_path)
            chs = exr.channels
            if "R" in chs:
                depth = exr.get("R")
            elif "Z" in chs:
                depth = exr.get("Z")
            else:
                continue
            depth_frames.append(depth.astype(np.float32))
        except Exception as e:
            print(f"⚠️ Failed to load {name}: {e}")
        finally:
            os.remove(tmp_path)
print(f"✅ Loaded {len(depth_frames)} EXR depth maps, shape: {depth_frames[0].shape}, dtype: {depth_frames[0].dtype}")

# --- 중심 포인트 ---
u0, v0 = depth_frames[0].shape[1] // 2, depth_frames[0].shape[0] // 2
depth0 = depth_frames[0][v0, u0]
pix = np.array([u0, v0, 1.0])
X_cam0 = np.linalg.inv(K) @ (pix * depth0)
X_cam0 = np.append(X_cam0, 1.0)
X_world = poses[0] @ X_cam0

# --- viser 서버 시작 ---
server = viser.ViserServer()
print("✅ Viser server running → open:", "http://localhost:8080")

server.scene.add_frame(name="ViPE_Point_Tracking")

# 포인트와 라인 세그먼트 생성
point_handle = server.scene.add_point_cloud(
    name="tracked_point",
    points=np.array([[0.0, 0.0, 0.0]]),
    colors=np.array([[1.0, 0.0, 0.0]]),
    point_size=0.05,
)

trail_handle = server.scene.add_line_segments(
    points=np.zeros((1, 2, 3)),
    colors=np.array([[1.0, 0.2, 0.2]]),
    name="trajectory_trail",
)

# --- 추적 루프 ---
async def track_loop():
    traj_points = [X_world[:3]]
    for i in range(1, len(poses)):
        await asyncio.sleep(0.05)
        X_cam_i = np.linalg.inv(poses[i]) @ X_world
        x_i = K @ X_cam_i[:3]
        x_i /= x_i[2]

        depth_i = depth_frames[i]
        u_i, v_i = int(x_i[0]), int(x_i[1])
        if (
            0 <= u_i < depth_i.shape[1]
            and 0 <= v_i < depth_i.shape[0]
            and abs(depth_i[v_i, u_i] - X_cam_i[2]) < 0.05
        ):
            traj_points.append(X_world[:3])

        # 포인트 업데이트
        point_handle.points = np.array([X_world[:3]])

        # 라인 세그먼트 업데이트
        if len(traj_points) > 1:
            trail_points = np.array(traj_points)
            segments = np.stack([trail_points[:-1], trail_points[1:]], axis=1)
            trail_handle.points = segments

# --- 비동기 서버 유지 ---
async def main():
    task1 = asyncio.create_task(track_loop())
    await task1
    await asyncio.Future()  # 서버를 계속 유지

asyncio.run(main())

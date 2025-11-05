# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio
import logging
import socket
import time

from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import viser
import viser.transforms as tf

from matplotlib import cm
import matplotlib.pyplot as plt
from PIL import Image
from rich.logging import RichHandler

from vipe.utils.cameras import CameraType
from vipe.utils.depth import reliable_depth_mask_range
from vipe.utils.io import (
    ArtifactPath,
    read_depth_artifacts,
    read_intrinsics_artifacts,
    read_pose_artifacts,
    read_rgb_artifacts,
)

import open3d as o3d   

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)

from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.io import read_video
from torchvision.utils import flow_to_image
import torchvision.transforms.functional as F

@dataclass
class GlobalContext:
    artifacts: list[ArtifactPath]


_global_context: GlobalContext | None = None


@dataclass
class SceneFrameHandle:
    frame_handle: viser.FrameHandle
    frustum_handle: viser.CameraFrustumHandle
    pcd_handle: viser.PointCloudHandle | None = None

    def __post_init__(self):
        self.visible = False

    @property
    def visible(self) -> bool:
        return self.frame_handle.visible

    @visible.setter
    def visible(self, value: bool):
        self.frame_handle.visible = value
        self.frustum_handle.visible = value
        if self.pcd_handle is not None:
            self.pcd_handle.visible = value


class ClientClosures:
    """
    All class methods automatically capture 'self', ensuring proper locals.
    """

    def __init__(self, client: viser.ClientHandle):
        self.client = client

        async def _run():
            try:
                await self.run()
            except asyncio.CancelledError:
                pass
            finally:
                self.cleanup()

        # Don't await to not block the rest of the coroutine.
        self.task = asyncio.create_task(_run())

        self.gui_playback_handle: viser.GuiFolderHandle | None = None
        self.gui_timestep: viser.GuiSliderHandle | None = None
        self.gui_framerate: viser.GuiSliderHandle | None = None
        self.scene_frame_handles: list[SceneFrameHandle] = []
        self.current_displayed_timestep: int = 0

        weights = Raft_Large_Weights.DEFAULT
        self.raft_transforms = weights.transforms()
        self.raft_model = raft_large(weights=weights, progress=False).to("cuda" if torch.cuda.is_available() else "cpu").eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    async def stop(self):
        self.task.cancel()
        await self.task

    async def run(self):
        logger.info(f"Client {self.client.client_id} connected")

        all_artifacts = self.global_context().artifacts

        # --- 기존 GUI 초기화 ---
        with self.client.gui.add_folder("Sample"):
            self.gui_id = self.client.gui.add_slider(
                "Artifact ID", min=0, max=len(all_artifacts) - 1, step=1, initial_value=0
            )
            gui_id_changer = self.client.gui.add_button_group(label="ID +/-", options=["Prev", "Next"])
            @gui_id_changer.on_click
            async def _(_) -> None:
                if gui_id_changer.value == "Prev":
                    self.gui_id.value = (self.gui_id.value - 1) % len(all_artifacts)
                else:
                    self.gui_id.value = (self.gui_id.value + 1) % len(all_artifacts)

            self.gui_name = self.client.gui.add_text("Artifact Name", "")
            self.gui_t_sub = self.client.gui.add_slider("Temporal subsample", min=1, max=16, step=1, initial_value=1)
            self.gui_s_sub = self.client.gui.add_slider("Spatial subsample", min=1, max=8, step=1, initial_value=2)
            self.gui_id.on_update(self.on_sample_update)
            self.gui_t_sub.on_update(self.on_sample_update)
            self.gui_s_sub.on_update(self.on_sample_update)

        with self.client.gui.add_folder("Scene"):
            self.gui_point_size = self.client.gui.add_slider(
                "Point size", min=0.0001, max=10, step=0.001, initial_value=0.001
            )
            @self.gui_point_size.on_update
            async def _(_) -> None:
                for frame_node in self.scene_frame_handles:
                    if frame_node.pcd_handle is not None:
                        frame_node.pcd_handle.point_size = self.gui_point_size.value

            self.gui_frustum_size = self.client.gui.add_slider(
                "Frustum size", min=0.01, max=0.5, step=0.01, initial_value=0.15
            )
            @self.gui_frustum_size.on_update
            async def _(_) -> None:
                for frame_node in self.scene_frame_handles:
                    frame_node.frustum_handle.scale = self.gui_frustum_size.value

            self.gui_colorful_frustum_toggle = self.client.gui.add_checkbox("Colorful Frustum", initial_value=False)
            @self.gui_colorful_frustum_toggle.on_update
            async def _(_) -> None:
                self._set_frustum_color(self.gui_colorful_frustum_toggle.value)

            self.gui_fov = self.client.gui.add_slider("FoV", min=30.0, max=120.0, step=1.0, initial_value=60.0)
            @self.gui_fov.on_update
            async def _(_) -> None:
                self.client.camera.fov = np.deg2rad(self.gui_fov.value)

            gui_snapshot = self.client.gui.add_button("Snapshot", hint="Take a snapshot of the current scene")
            @gui_snapshot.on_click
            def _(_) -> None:
                current_artifact = self.global_context().artifacts[self.gui_id.value]
                file_name = f"{current_artifact.base_path.name}_{current_artifact.artifact_name}.png"
                snapshot_img = self.client.get_render(height=720, width=1280, transport_format="png")
                self.client.send_file_download(file_name, iio.imwrite("<bytes>", snapshot_img, extension=".png"))

        # --- 첫 frame만 보여주기 ---
        await self.on_sample_update(None)
        # 모든 프레임 숨기고 0번째만 보이도록
        with self.client.atomic():
            for i, handle in enumerate(self.scene_frame_handles):
                handle.visible = (i == 0)
        self.current_displayed_timestep = 0
        self.is_playing = False  # 초기에는 재생 멈춤 상태

        # --- 메인 루프 ---
        while True:
            if self.is_playing and self.gui_framerate is not None and self.gui_framerate.value > 0:
                self._incr_timestep()
                await asyncio.sleep(1.0 / self.gui_framerate.value)
            else:
                await asyncio.sleep(0.1)


    def _set_frustum_color(self, colorful: bool):
        for frame_idx, frame_node in enumerate(self.scene_frame_handles):
            if not colorful:
                frame_node.frustum_handle.color = (0, 0, 0)
            else:
                # Use a rainbow color based on the frame index
                denom = len(self.scene_frame_handles) - 1
                rainbow_value = cm.jet(1.0 - frame_idx / denom)[:3]
                rainbow_value = tuple((int(c * 255) for c in rainbow_value))
                frame_node.frustum_handle.color = rainbow_value

    async def on_sample_update(self, _):
        with self.client.atomic():
            current_artifact = self.global_context().artifacts[self.gui_id.value]
            rgb_seq = list(read_rgb_artifacts(current_artifact.rgb_path))
            self.precompute_optical_flow_all(rgb_seq)
            self._rebuild_scene()
        self._rebuild_playback_gui()
        self._set_frustum_color(self.gui_colorful_frustum_toggle.value)

        # 🔹 Rebuild object control GUI
        self._build_object_gui()
        # 🔹 Rebuild 3D point matching GUI
        self._build_matching_gui()

    def _build_matching_gui(self):
        """3D point matching을 위한 GUI 생성"""
        # 이미 있으면 지우기
        if hasattr(self, "gui_matching_folder"):
            self.gui_matching_folder.remove()

        self.gui_matching_folder = self.client.gui.add_folder("3D Point Matching")

        with self.gui_matching_folder:
            max_frames = len(self.scene_frame_handles) - 1 if len(self.scene_frame_handles) > 0 else 0
            self.gui_frame_i = self.client.gui.add_slider(
                "Frame I", min=0, max=max_frames, step=1, initial_value=0
            )
            self.gui_frame_j = self.client.gui.add_slider(
                "Frame J", min=0, max=max_frames, step=1, initial_value=1
            )

            # Frame I 변경 시 Frame J 자동 업데이트
            @self.gui_frame_i.on_update
            async def _(_) -> None:
                new_frame_j = min(self.gui_frame_i.value + 1, max_frames)
                if self.gui_frame_j.value != new_frame_j:
                    self.gui_frame_j.value = new_frame_j

            gui_match_button = self.client.gui.add_button(
                "Match 3D Points",
                hint="Optical flow를 이용해서 두 프레임 간의 3D 점들을 매칭합니다"
            )

            @gui_match_button.on_click
            def _(_):
                frame_i = int(self.gui_frame_i.value)
                frame_j = int(self.gui_frame_j.value)
                
                # Optical flow는 연속된 프레임 간에만 가능하므로 자동 조정
                if frame_j != frame_i + 1:
                    logger.warning(f"⚠️ Frame J should be Frame I + 1 for optical flow. Auto-adjusting to {frame_i + 1}")
                    frame_j = frame_i + 1
                    self.gui_frame_j.value = frame_j
                
                logger.info(f"🔍 Matching 3D points between frames {frame_i} and {frame_j}")
                
                matched_pairs, valid_mask = self.match_3d_points_with_optical_flow(frame_i, frame_j)
                
                if matched_pairs is not None:
                    self.matched_pairs = matched_pairs
                    self.matched_valid_mask = valid_mask
                    logger.info(f"✅ Stored {len(matched_pairs)} matched point pairs")
                else:
                    logger.warning("❌ Failed to match points")


    def _build_object_gui(self):
        """추가된 point cloud를 제어하는 GUI 생성"""
        # 이미 있으면 지우기
        if hasattr(self, "gui_object_folder"):
            self.gui_object_folder.remove()

        self.gui_object_folder = self.client.gui.add_folder("Object Transform")

        with self.gui_object_folder:
            self.gui_offset_x = self.client.gui.add_slider("Offset X", min=-100.0, max=100.0, step=0.01, initial_value=0.0)
            self.gui_offset_y = self.client.gui.add_slider("Offset Y", min=-100.0, max=100.0, step=0.01, initial_value=0.0)
            self.gui_offset_z = self.client.gui.add_slider("Offset Z", min=-100.0, max=100.0, step=0.01, initial_value=0.0)
            self.gui_scale = self.client.gui.add_slider("Scale", min=0.01, max=2.0, step=0.05, initial_value=0.5)
            self.gui_point_size = self.client.gui.add_slider("Point Scale", min=0.0001, max=0.001, step=0.0001, initial_value=0.0001)
            self.gui_yaw = self.client.gui.add_slider("Yaw (deg)", min=-180, max=180, step=5, initial_value=0)
            self.gui_pitch = self.client.gui.add_slider("Pitch (deg)", min=-180, max=180, step=5, initial_value=0)
            self.gui_roll = self.client.gui.add_slider("Roll (deg)", min=-180, max=180, step=5, initial_value=0)
            
            self.gui_nearby_distance = self.client.gui.add_slider(
                "Nearby Distance", min=0.01, max=10.0, step=0.01, initial_value=1.0,
                hint="Object 근처에서 고려할 점들의 거리 (m)"
            )
            
            self.gui_flow_threshold = self.client.gui.add_slider(
                "Flow Threshold", min=1.0, max=100.0, step=1.0, initial_value=50.0,
                hint="Optical flow 추적 시 급격한 변화를 제외하는 임계값 (픽셀)"
            )
            
            self.gui_bbox_width_scale = self.client.gui.add_slider(
                "Bounding Box Width Scale", min=0.5, max=3.0, step=0.1, initial_value=1.0,
                hint="Bounding box 가로 크기 배율 (1.0 = 원본 크기)"
            )
            
            self.gui_bbox_height_scale = self.client.gui.add_slider(
                "Bounding Box Height Scale", min=0.5, max=3.0, step=0.1, initial_value=1.0,
                hint="Bounding box 세로 크기 배율 (1.0 = 원본 크기)"
            )

            # 업데이트 핸들러
            async def _update_object(_):
                self._update_custom_object()

            for slider in [
                self.gui_offset_x, self.gui_offset_y, self.gui_offset_z,
                self.gui_scale, self.gui_point_size, self.gui_yaw, self.gui_pitch, self.gui_roll
            ]:
                slider.on_update(_update_object)

            # --- 🎥 Reproject Video 버튼 추가 ---
            gui_reproject = self.client.gui.add_button(
                "Reproject Video",
                hint="현재 위치의 물체를 카메라 trajectory로 다시 영상으로 렌더링합니다",
            )

            @gui_reproject.on_click
            def _(_):
                logger.info(f"✅ REPROJECTION")
                save_path = f"reproject_{int(time.time())}.mp4"
                self.reproject_pointcloud_to_video(save_path)
            
            # --- 🖼️ Reproject Single Frame 버튼 추가 ---
            max_frames = len(self.scene_frame_handles) - 1 if len(self.scene_frame_handles) > 0 else 0
            gui_frame_select = self.client.gui.add_slider(
                "Frame to Reproject", min=0, max=max_frames, step=1, initial_value=0
            )
            
            self.gui_inertia_start_frame = self.client.gui.add_slider(
                "Inertia Start Frame", min=-1, max=max_frames, step=1, initial_value=-1,
                hint="관성 적용 시작 프레임 (-1 = 비활성화). 이 프레임부터는 관성 모션이 적용됩니다."
            )
            
            gui_reproject_single = self.client.gui.add_button(
                "Reproject Single Frame",
                hint="선택한 프레임만 reproject하여 이미지로 저장합니다",
            )
            
            @gui_reproject_single.on_click
            def _(_):
                frame_idx = int(gui_frame_select.value)
                logger.info(f"✅ Reprojecting single frame {frame_idx}")
                self.reproject_single_frame(frame_idx)

            # --- 🔍 Overlap Pixels 확인 버튼 추가 ---
            gui_check_overlap = self.client.gui.add_button(
                "Check Overlap Pixels",
                hint="첫 프레임에서 object가 overlap하는 픽셀 좌표를 확인하고, 모든 프레임으로 추적하여 이미지에 표시하고 저장합니다",
            )

            @gui_check_overlap.on_click
            def _(_):
                # 첫 프레임 overlap 픽셀 확인
                overlap_pixels = self._get_object_overlap_pixels(frame_idx=0)
                if overlap_pixels is not None:
                    logger.info(f"✅ Found {len(overlap_pixels)} overlap pixels")
                    logger.info(f"   Pixel range: u=[{overlap_pixels[:, 0].min()}, {overlap_pixels[:, 0].max()}], "
                              f"v=[{overlap_pixels[:, 1].min()}, {overlap_pixels[:, 1].max()}]")
                    
                    # 모든 프레임으로 overlap 픽셀 추적
                    logger.info("🔍 Tracking overlap pixels across all frames...")
                    tracked_overlap = self._track_overlap_pixels_all_frames(frame_idx_start=0)
                    logger.info(f"✅ Tracked overlap pixels across {len(tracked_overlap)} frames")
                    
                    # 근처 픽셀 찾기 및 추적
                    tracked_nearby = None
                    nearby_pixels = self._get_nearby_pixels(overlap_pixels, frame_idx=0, distance_threshold=20.0)
                    if nearby_pixels is not None:
                        logger.info("🔍 Tracking nearby pixels across all frames...")
                        tracked_nearby = self._track_nearby_pixels_all_frames(nearby_pixels)
                    else:
                        logger.warning("❌ Failed to track nearby pixels across frames")
                    
                    if tracked_overlap is not None:
                        logger.info(f"✅ Tracked overlap pixels across {len(tracked_overlap)} frames")
                        if tracked_nearby is not None:
                            logger.info(f"✅ Tracked nearby pixels across {len(tracked_nearby)} frames")
                        # 이미지에 표시하고 저장
                        self._visualize_and_save_tracked_pixels(tracked_overlap, tracked_nearby)
                    else:
                        logger.warning("❌ Failed to track pixels across frames")
                else:
                    logger.warning("❌ Failed to get overlap pixels. Check if object is placed correctly.")

    def _get_object_overlap_pixels(self, frame_idx: int = 0):
        """
        첫 프레임에서 object가 overlap되는 픽셀 좌표를 반환합니다.
        
        Args:
            frame_idx: 프레임 인덱스 (기본값: 0, 첫 프레임)
            
        Returns:
            overlap_pixels: numpy array of shape [N, 2] with (u, v) pixel coordinates
            또는 None if object가 없거나 프로젝션 실패
        """
        if not hasattr(self, "pc_world"):
            return None
        
        current_artifact = self.global_context().artifacts[self.gui_id.value]
        pose_seq = read_pose_artifacts(current_artifact.pose_path)[1]
        intr_seq = read_intrinsics_artifacts(current_artifact.intrinsics_path, current_artifact.camera_type_path)[1]
        rgb_seq = list(read_rgb_artifacts(current_artifact.rgb_path))
        
        # 프레임 인덱스 유효성 검사
        if frame_idx >= len(rgb_seq):
            return None
        
        # pose_seq와 intr_seq는 인덱싱 가능하지만 len()이 없을 수 있음
        try:
            if frame_idx >= len(intr_seq):
                return None
        except TypeError:
            # intr_seq가 len()을 지원하지 않는 경우, rgb_seq 길이와 비교
            pass
        
        # 첫 프레임의 pose와 intrinsics
        try:
            pose = pose_seq[frame_idx].matrix().cpu().numpy()
            intr = intr_seq[frame_idx].cpu().numpy()
        except (IndexError, AttributeError):
            return None
        h, w = rgb_seq[frame_idx][1].shape[:2]
        
        # Object 점들을 카메라 좌표로 변환
        R = pose[:3, :3]
        t = pose[:3, 3]
        obj_points = self.pc_world.copy()
        
        cam_points = (obj_points - t) @ R.T
        
        # Depth가 양수인 점들만 유효
        valid_depth = cam_points[:, 2] > 1e-6
        if not valid_depth.any():
            return None
        
        cam_points_valid = cam_points[valid_depth]
        
        # 픽셀 좌표로 프로젝션
        fx, fy, cx, cy = intr[:4]
        u = fx * (cam_points_valid[:, 0] / cam_points_valid[:, 2]) + cx
        v = fy * (cam_points_valid[:, 1] / cam_points_valid[:, 2]) + cy
        
        # 이미지 범위 내에 있는 픽셀만 선택
        in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not in_bounds.any():
            return None
        
        overlap_pixels = np.stack([u[in_bounds], v[in_bounds]], axis=1).astype(int)
        
        logger.info(f"✅ Found {len(overlap_pixels)} overlapping pixels for object in frame {frame_idx}")
        
        return overlap_pixels

    def _track_overlap_pixels_all_frames(self, frame_idx_start: int = 0):
        """
        첫 프레임의 overlap 픽셀들을 optical flow로 추적하여 모든 프레임에서의 위치를 구합니다.
        
        Args:
            frame_idx_start: 시작 프레임 인덱스 (기본값: 0)
            
        Returns:
            tracked_pixels: dict {frame_idx: numpy array [N, 2]} 형태로 각 프레임의 픽셀 좌표
            또는 None if 실패
        """
        # 첫 프레임의 overlap 픽셀 가져오기
        overlap_pixels_frame0 = self._get_object_overlap_pixels(frame_idx=frame_idx_start)
        if overlap_pixels_frame0 is None or len(overlap_pixels_frame0) == 0:
            return None
        
        if not hasattr(self, "flow_cache") or len(self.flow_cache) == 0:
            return None
        
        current_artifact = self.global_context().artifacts[self.gui_id.value]
        rgb_seq = list(read_rgb_artifacts(current_artifact.rgb_path))
        total_frames = len(rgb_seq)
        
        if total_frames == 0:
            return None
        
        h, w = rgb_seq[0][1].shape[:2]
        
        # 추적 결과 저장
        tracked_pixels = {frame_idx_start: overlap_pixels_frame0.astype(float)}
        
        # 현재 픽셀 좌표 (float로 추적)
        current_pixels = overlap_pixels_frame0.astype(float).copy()
        valid_mask = np.ones(len(current_pixels), dtype=bool)
        
        # 각 프레임별로 추적
        for frame_idx in range(frame_idx_start + 1, total_frames):
            if frame_idx - 1 >= len(self.flow_cache) or self.flow_cache[frame_idx - 1] is None:
                # flow가 없으면 이전 프레임의 위치 유지
                tracked_pixels[frame_idx] = current_pixels[valid_mask].copy()
                continue
            
            # Optical flow 적용
            flow = self.flow_cache[frame_idx - 1].numpy()
            
            # 유효한 픽셀 좌표만 처리
            if not valid_mask.any():
                tracked_pixels[frame_idx] = np.empty((0, 2))
                continue
            
            # 현재 픽셀 좌표가 이미지 범위 내에 있는지 확인
            in_bounds = (
                (current_pixels[:, 0] >= 0) & (current_pixels[:, 0] < w) &
                (current_pixels[:, 1] >= 0) & (current_pixels[:, 1] < h)
            )
            valid_mask &= in_bounds
            
            if not valid_mask.any():
                tracked_pixels[frame_idx] = np.empty((0, 2))
                continue
            
            # Optical flow 샘플링
            u_int = np.clip(current_pixels[valid_mask, 0].astype(int), 0, w - 1)
            v_int = np.clip(current_pixels[valid_mask, 1].astype(int), 0, h - 1)
            
            # Flow 값 가져오기
            u_flow = flow[0, v_int, u_int]
            v_flow = flow[1, v_int, u_int]
            
            # Flow 크기 계산 (급격한 변화 필터링)
            flow_magnitude = np.sqrt(u_flow**2 + v_flow**2)
            flow_threshold = getattr(self, "gui_flow_threshold", None)
            if flow_threshold is not None:
                flow_threshold_value = flow_threshold.value
            else:
                flow_threshold_value = 50.0  # 기본값
            
            # Threshold 이하인 flow만 유효
            valid_flow_mask = flow_magnitude <= flow_threshold_value
            
            # valid_mask에서 valid_flow_mask를 적용 (valid_mask 내부의 인덱스에만 적용)
            valid_mask_indices = np.where(valid_mask)[0]
            valid_mask[valid_mask_indices[~valid_flow_mask]] = False
            
            if not valid_mask.any():
                tracked_pixels[frame_idx] = np.empty((0, 2))
                continue
            
            # 유효한 flow만 적용 (valid_mask가 업데이트되었으므로 다시 flow를 가져와야 함)
            u_int_filtered = np.clip(current_pixels[valid_mask, 0].astype(int), 0, w - 1)
            v_int_filtered = np.clip(current_pixels[valid_mask, 1].astype(int), 0, h - 1)
            u_flow_filtered = flow[0, v_int_filtered, u_int_filtered]
            v_flow_filtered = flow[1, v_int_filtered, u_int_filtered]
            
            current_pixels[valid_mask, 0] += u_flow_filtered
            current_pixels[valid_mask, 1] += v_flow_filtered
            
            # 결과 저장
            tracked_pixels[frame_idx] = current_pixels[valid_mask].copy()
        
        logger.info(f"✅ Tracked overlap pixels across {len(tracked_pixels)} frames")
        
        return tracked_pixels

    def _get_nearby_pixels(self, overlap_pixels: np.ndarray, frame_idx: int = 0, distance_threshold: float = 20.0):
        """
        Overlap 픽셀의 bounding box 내부에 있으면서 object 바깥의 scene 포인트들을 찾습니다.
        
        Args:
            overlap_pixels: overlap 픽셀 좌표 [N, 2]
            frame_idx: 프레임 인덱스
            distance_threshold: 사용하지 않음 (호환성을 위해 유지)
            
        Returns:
            nearby_pixels: bounding box 내부의 object 바깥 픽셀 좌표 [M, 2] 또는 None
        """
        current_artifact = self.global_context().artifacts[self.gui_id.value]
        depth_seq = list(read_depth_artifacts(current_artifact.depth_path))
        
        if frame_idx >= len(depth_seq):
            logger.warning("❌ No depth sequence found")
            return None
        
        depth = depth_seq[frame_idx][1].cpu().numpy()
        h, w = depth.shape
        
        # Overlap 픽셀의 bounding box 계산
        u_min_raw = overlap_pixels[:, 0].min()
        u_max_raw = overlap_pixels[:, 0].max()
        v_min_raw = overlap_pixels[:, 1].min()
        v_max_raw = overlap_pixels[:, 1].max()
        
        # Bounding box 중심
        u_center = (u_min_raw + u_max_raw) / 2
        v_center = (v_min_raw + v_max_raw) / 2
        
        # Bounding box 크기
        u_width = u_max_raw - u_min_raw
        v_height = v_max_raw - v_min_raw
        
        # Scale 적용 (가로/세로 따로)
        bbox_width_scale = getattr(self, "gui_bbox_width_scale", None)
        bbox_height_scale = getattr(self, "gui_bbox_height_scale", None)
        
        if bbox_width_scale is not None:
            width_scale = bbox_width_scale.value
        else:
            width_scale = 1.0
            
        if bbox_height_scale is not None:
            height_scale = bbox_height_scale.value
        else:
            height_scale = 1.0
        
        # Scale된 크기 (가로/세로 따로)
        u_width_scaled = u_width * width_scale
        v_height_scaled = v_height * height_scale
        
        # Scale된 bounding box
        u_min = int(u_center - u_width_scaled / 2)
        u_max = int(u_center + u_width_scaled / 2) + 1
        v_min = int(v_center - v_height_scaled / 2)
        v_max = int(v_center + v_height_scaled / 2) + 1
        
        # 이미지 범위 내로 제한
        u_min = max(0, u_min)
        u_max = min(w, u_max)
        v_min = max(0, v_min)
        v_max = min(h, v_max)
        
        logger.info(f"   Bounding box (width_scale={width_scale:.2f}, height_scale={height_scale:.2f}): u=[{u_min}, {u_max}], v=[{v_min}, {v_max}]")
        
        # 유효한 depth 마스크
        mask = reliable_depth_mask_range(torch.from_numpy(depth)).numpy()
        ys, xs = np.where(mask)
        
        if len(xs) == 0:
            logger.warning("❌ No valid depth pixels found")
            return None
        
        # 모든 유효한 픽셀 좌표
        all_pixels = np.stack([xs, ys], axis=1).astype(float)
        
        # Bounding box 내부에 있는 픽셀만 선택
        in_bbox_mask = (
            (all_pixels[:, 0] >= u_min) & (all_pixels[:, 0] < u_max) &
            (all_pixels[:, 1] >= v_min) & (all_pixels[:, 1] < v_max)
        )
        
        if in_bbox_mask.sum() == 0:
            logger.warning("❌ No pixels inside bounding box")
            return None
        
        bbox_pixels = all_pixels[in_bbox_mask]
        
        # Overlap 픽셀 자체는 제외 (object 바깥만)
        overlap_set = set(map(tuple, overlap_pixels.astype(int)))
        bbox_pixels_int = bbox_pixels.astype(int)
        not_overlap_mask = np.array([tuple(p) not in overlap_set for p in bbox_pixels_int])
        
        if not_overlap_mask.sum() == 0:
            logger.warning("❌ No pixels found outside object in bounding box")
            return None
        
        nearby_pixels = bbox_pixels[not_overlap_mask]
        
        logger.info(f"✅ Found {len(nearby_pixels)} pixels inside bounding box but outside object")
        
        return nearby_pixels

    def _track_nearby_pixels_all_frames(self, nearby_pixels_frame0: np.ndarray):
        """
        첫 프레임의 근처 픽셀들을 optical flow로 추적합니다.
        
        Args:
            nearby_pixels_frame0: 첫 프레임의 근처 픽셀 좌표 [N, 2]
            
        Returns:
            tracked_pixels: dict {frame_idx: numpy array [N, 2]}
        """
        if not hasattr(self, "flow_cache") or len(self.flow_cache) == 0:
            logger.warning("❌ No flow cache found")
            return None
        
        current_artifact = self.global_context().artifacts[self.gui_id.value]
        rgb_seq = list(read_rgb_artifacts(current_artifact.rgb_path))
        total_frames = len(rgb_seq)
        
        if total_frames == 0:
            logger.warning("❌ No total frames found")
            return None
        
        h, w = rgb_seq[0][1].shape[:2]
        
        tracked_pixels = {0: nearby_pixels_frame0.astype(float)}
        current_pixels = nearby_pixels_frame0.astype(float).copy()
        valid_mask = np.ones(len(current_pixels), dtype=bool)
        
        for frame_idx in range(1, total_frames):
            if frame_idx - 1 >= len(self.flow_cache) or self.flow_cache[frame_idx - 1] is None:
                tracked_pixels[frame_idx] = current_pixels[valid_mask].copy()
                continue
            
            flow = self.flow_cache[frame_idx - 1].numpy()
            
            if not valid_mask.any():
                tracked_pixels[frame_idx] = np.empty((0, 2))
                continue
            
            in_bounds = (
                (current_pixels[:, 0] >= 0) & (current_pixels[:, 0] < w) &
                (current_pixels[:, 1] >= 0) & (current_pixels[:, 1] < h)
            )
            valid_mask &= in_bounds
            
            if not valid_mask.any():
                tracked_pixels[frame_idx] = np.empty((0, 2))
                continue
            
            u_int = np.clip(current_pixels[valid_mask, 0].astype(int), 0, w - 1)
            v_int = np.clip(current_pixels[valid_mask, 1].astype(int), 0, h - 1)
            
            u_flow = flow[0, v_int, u_int]
            v_flow = flow[1, v_int, u_int]
            
            # Flow 크기 계산 (급격한 변화 필터링)
            flow_magnitude = np.sqrt(u_flow**2 + v_flow**2)
            flow_threshold = getattr(self, "gui_flow_threshold", None)
            if flow_threshold is not None:
                flow_threshold_value = flow_threshold.value
            else:
                flow_threshold_value = 50.0  # 기본값
            
            # Threshold 이하인 flow만 유효
            valid_flow_mask = flow_magnitude <= flow_threshold_value
            
            # valid_mask에서 valid_flow_mask를 적용
            valid_mask_indices = np.where(valid_mask)[0]
            valid_mask[valid_mask_indices[~valid_flow_mask]] = False
            
            if not valid_mask.any():
                tracked_pixels[frame_idx] = np.empty((0, 2))
                continue
            
            # 유효한 flow만 적용 (valid_mask가 업데이트되었으므로 다시 flow를 가져와야 함)
            u_int_filtered = np.clip(current_pixels[valid_mask, 0].astype(int), 0, w - 1)
            v_int_filtered = np.clip(current_pixels[valid_mask, 1].astype(int), 0, h - 1)
            u_flow_filtered = flow[0, v_int_filtered, u_int_filtered]
            v_flow_filtered = flow[1, v_int_filtered, u_int_filtered]
            
            current_pixels[valid_mask, 0] += u_flow_filtered
            current_pixels[valid_mask, 1] += v_flow_filtered
            
            tracked_pixels[frame_idx] = current_pixels[valid_mask].copy()
        
        return tracked_pixels

    def _visualize_and_save_tracked_pixels(self, tracked_overlap: dict, tracked_nearby: dict = None):
        """
        추적된 픽셀들을 각 프레임 이미지에 표시하고 저장합니다.
        - Overlap 픽셀: 빨간색
        - 근처 픽셀: 파란색
        
        Args:
            tracked_overlap: {frame_idx: numpy array [N, 2]} - overlap 픽셀
            tracked_nearby: {frame_idx: numpy array [M, 2]} - 근처 픽셀 (optional)
        """
        from pathlib import Path
        from PIL import ImageDraw
        
        current_artifact = self.global_context().artifacts[self.gui_id.value]
        rgb_seq = list(read_rgb_artifacts(current_artifact.rgb_path))
        
        save_dir = Path("overlap_pixels_visualization")
        save_dir.mkdir(exist_ok=True)
        
        logger.info(f"🎨 Visualizing and saving tracked pixels to {save_dir}/")
        
        all_frame_indices = set(tracked_overlap.keys())
        if tracked_nearby:
            all_frame_indices.update(tracked_nearby.keys())
        
        for frame_idx in sorted(all_frame_indices):
            if frame_idx >= len(rgb_seq):
                continue
            
            # RGB 이미지 가져오기
            rgb_tensor = rgb_seq[frame_idx][1]
            if isinstance(rgb_tensor, torch.Tensor):
                rgb_img = rgb_tensor.cpu().numpy()
            else:
                rgb_img = rgb_tensor
            
            if rgb_img.max() <= 1.0:
                rgb_img = (rgb_img * 255).astype(np.uint8)
            else:
                rgb_img = rgb_img.astype(np.uint8)
            
            img = Image.fromarray(rgb_img)
            draw = ImageDraw.Draw(img)
            h, w = rgb_img.shape[:2]
            
            # 근처 픽셀 (파란색) 먼저 그리기
            if tracked_nearby and frame_idx in tracked_nearby:
                nearby_pixels = tracked_nearby[frame_idx]
                if len(nearby_pixels) > 0:
                    valid_nearby = nearby_pixels[
                        (nearby_pixels[:, 0] >= 0) & (nearby_pixels[:, 0] < w) &
                        (nearby_pixels[:, 1] >= 0) & (nearby_pixels[:, 1] < h)
                    ]
                    for u, v in valid_nearby.astype(int):
                        draw.ellipse([u-1, v-1, u+1, v+1], fill=(0, 0, 255))  # 파란색
            
            # Overlap 픽셀 (빨간색) 그리기
            if frame_idx in tracked_overlap:
                overlap_pixels = tracked_overlap[frame_idx]
                if len(overlap_pixels) > 0:
                    valid_overlap = overlap_pixels[
                        (overlap_pixels[:, 0] >= 0) & (overlap_pixels[:, 0] < w) &
                        (overlap_pixels[:, 1] >= 0) & (overlap_pixels[:, 1] < h)
                    ]
                    for u, v in valid_overlap.astype(int):
                        draw.ellipse([u-1, v-1, u+1, v+1], fill=(255, 0, 0))  # 빨간색
            
            # 이미지 저장
            save_path = save_dir / f"frame_{frame_idx:04d}_overlap.png"
            img.save(save_path)
            
            if frame_idx % 10 == 0:
                logger.info(f"  Saved frame {frame_idx}/{len(all_frame_indices)-1}")
        
        logger.info(f"✅ Saved {len(all_frame_indices)} visualization images to {save_dir}/")

    def _update_custom_object(self):
        """GUI 슬라이더 값으로 물체 transform"""
        if not hasattr(self, "pc_raw"):
            return  # 아직 안 로드됐으면 skip

        offset = np.array([
            self.gui_offset_x.value,
            self.gui_offset_y.value,
            self.gui_offset_z.value,
        ])
        scale = self.gui_scale.value
        point_size = self.gui_point_size.value
        yaw, pitch, roll = np.deg2rad([
            self.gui_yaw.value,
            self.gui_pitch.value,
            self.gui_roll.value,
        ])

        def rotation_matrix_xyz(rx, ry, rz):
            Rx = np.array([[1, 0, 0],
                        [0, np.cos(rx), -np.sin(rx)],
                        [0, np.sin(rx), np.cos(rx)]])
            Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                        [0, 1, 0],
                        [-np.sin(ry), 0, np.cos(ry)]])
            Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                        [np.sin(rz), np.cos(rz), 0],
                        [0, 0, 1]])
            return Rz @ Ry @ Rx

        R_local = rotation_matrix_xyz(pitch, yaw, roll)
        pc = self.pc_raw * scale
        pc_rotated = (R_local @ pc.T).T

        base_frame = self.scene_frame_handles[0]
        R = tf.SO3(base_frame.frame_handle.wxyz).as_matrix()
        t = np.array(base_frame.frame_handle.position)
        pc_world = (R @ pc_rotated.T).T + t + offset

        # ✅ 기존 객체 제거 후 새로 추가
        with self.client.atomic():
            try:
                self.client.scene.remove_node("/custom_object/green_object")
            except Exception:
                pass

            self.client.scene.add_point_cloud(
                name="/custom_object/green_object",
                points=pc_world,
                colors=np.tile(np.array([[0, 255, 0]]), (pc_world.shape[0], 1)),
                point_size=point_size,
            )

        self.pc_world = pc_world.copy()

    # def reproject_pointcloud_to_video(self, save_path="reproject_dynamic_tracked_affine.mp4"):
    #     """
    #     Optical flow 기반으로 local affine transform (rotation + scale + translation)
    #     을 추정해 object point cloud가 scene과 함께 움직이는 안정 버전.
    #     """
    #     import subprocess, tempfile
    #     from PIL import Image
    #     from vipe.ext.lietorch import SE3
    #     from vipe.utils.visualization import project_points

    #     logger.info("🎬 Starting flow-based dynamic object reprojection (local affine)...")

    #     current_artifact = self.global_context().artifacts[self.gui_id.value]
    #     rgb_seq = list(read_rgb_artifacts(current_artifact.rgb_path))
    #     pose_seq = read_pose_artifacts(current_artifact.pose_path)[1]
    #     intr_seq = read_intrinsics_artifacts(current_artifact.intrinsics_path, current_artifact.camera_type_path)[1]
    #     depth_seq = list(read_depth_artifacts(current_artifact.depth_path))
    #     h, w, _ = rgb_seq[0][1].shape

    #     if not hasattr(self, "pc_world"):
    #         logger.warning("❌ No object point cloud found.")
    #         return

    #     obj_points = self.pc_world.copy()
    #     obj_colors = np.tile(np.array([[0, 255, 0]]), (obj_points.shape[0], 1))

    #     frames = []
    #     tracked_points = obj_points.copy()

    #     for frame_idx in range(min(len(depth_seq)-1, pose_seq.shape[0]-1)):
    #         rgb = rgb_seq[frame_idx][1].cpu().numpy()
    #         depth = depth_seq[frame_idx][1].cpu().numpy()
    #         intr = intr_seq[frame_idx].cpu().numpy()
    #         pose = pose_seq[frame_idx].matrix().cpu().numpy()
    #         flow = self.flow_cache[frame_idx]

    #         if flow is None:
    #             continue

    #         fx, fy, cx, cy = intr[:4]
    #         R, t = pose[:3, :3], pose[:3, 3]

    #         # --- scene backprojection ---
    #         mask = reliable_depth_mask_range(torch.from_numpy(depth)).numpy()
    #         ys, xs = np.where(mask)
    #         if len(xs) == 0:
    #             continue
    #         z = depth[ys, xs]
    #         X = (xs - cx) * z / fx
    #         Y = (ys - cy) * z / fy
    #         pts_cam = np.stack([X, Y, z], axis=-1)
    #         pts_world = (pts_cam @ R.T) + t
    #         colors_scene = rgb[ys, xs] if rgb.max() > 1 else (rgb[ys, xs] * 255).astype(np.uint8)

    #         # --- object projection footprint ---
    #         pts_cam_obj = (tracked_points - t) @ R
    #         uv = np.stack([
    #             fx * (pts_cam_obj[:, 0] / pts_cam_obj[:, 2]) + cx,
    #             fy * (pts_cam_obj[:, 1] / pts_cam_obj[:, 2]) + cy
    #         ], axis=-1)
    #         valid_mask = (
    #             (uv[:, 0] >= 0) & (uv[:, 0] < w) &
    #             (uv[:, 1] >= 0) & (uv[:, 1] < h)
    #         )
    #         uv_valid = uv[valid_mask].astype(int)
    #         if len(uv_valid) < 20:
    #             continue

    #         # --- local flow sampling ---
    #         flow_np = flow.numpy()
    #         u_disp = flow_np[0, uv_valid[:, 1], uv_valid[:, 0]]
    #         v_disp = flow_np[1, uv_valid[:, 1], uv_valid[:, 0]]

    #         # --- local affine estimation (rotation + scale + translation) ---
    #         x = uv_valid[:, 0] - np.mean(uv_valid[:, 0])
    #         y = uv_valid[:, 1] - np.mean(uv_valid[:, 1])
    #         X_mat = np.stack([x, y, np.ones_like(x)], axis=1)
    #         Y_mat = np.stack([x + u_disp, y + v_disp], axis=1)
    #         A, _, _, _ = np.linalg.lstsq(X_mat, Y_mat, rcond=None)
    #         A_lin = A[:2, :2]
    #         U, S, Vt = np.linalg.svd(A_lin)
    #         R_est = U @ Vt
    #         scale = np.mean(S)
    #         t2d = A[2, :2]


    #         # --- convert 2D rotation (xy plane) to 3D rotation matrix ---
    #         R_est_3d = np.eye(3)
    #         R_est_3d[:2, :2] = R_est

    #         # --- pixel flow → full 3D translation (Δx, Δy, Δz) ---
    #         # use both optical flow and depth difference
    #         z_curr = depth[uv_valid[:, 1], uv_valid[:, 0]]
    #         z_next = depth[uv_valid[:, 1].clip(0, h - 1), (uv_valid[:, 0] + u_disp).clip(0, w - 1).astype(int)]
    #         delta_z = np.nanmean(z_next - z_curr)

    #         # --- pixel flow → 3D translation ---
    #         z_mean = np.mean(z_curr)
    #         delta_x = np.mean(u_disp) * z_mean / fx
    #         delta_y = np.mean(v_disp) * z_mean / fy
    #         #delta_z = 0.0

    #         # --- apply rotation + scale + translation to object points ---
    #         tracked_center = tracked_points.mean(0)
    #         tracked_points = (tracked_points - tracked_center) @ R_est_3d.T * scale + tracked_center
    #         tracked_points += np.array([delta_x, delta_y, delta_z])

    #         # --- scene + object merge ---
    #         all_points = np.concatenate([pts_world, tracked_points], axis=0)
    #         all_colors = np.concatenate([colors_scene, obj_colors], axis=0)

    #         # --- projection ---
    #         img = project_points(
    #             xyz=all_points,
    #             intrinsics=intr,
    #             camera_type=CameraType.PINHOLE,
    #             pose=pose_seq[frame_idx],
    #             frame_size=(h, w),
    #             subsample_factor=1,
    #             color=all_colors,
    #         )

    #         frames.append(img)
    #         Image.fromarray(img).save(f"/home/nas5/hoiyeongjin/repos/Project/SKT/vipe/vipe/check/{frame_idx:04d}.png")
    #         logger.info(f"  rendered frame {frame_idx+1}/{len(depth_seq)} (scale={scale:.4f})")

    #     # --- save video ---
    #     with tempfile.TemporaryDirectory() as tmpdir:
    #         tmp_path = Path(tmpdir)
    #         for i, frame in enumerate(frames):
    #             Image.fromarray(frame).save(tmp_path / f"{i:04d}.png")
    #         subprocess.run([
    #             "ffmpeg", "-y", "-framerate", "30",
    #             "-i", str(tmp_path / "%04d.png"),
    #             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", save_path
    #         ], check=True)

    #     logger.info(f"✅ Saved local-affine tracked video to {save_path}")


    def reproject_pointcloud_to_video(self, save_path="reproject_dynamic_translation.mp4"):
        """
        첫 번째 프레임의 점들을 기준으로 각 프레임에서 동일한 점들을 추적하고,
        그들의 motion vector 평균으로 object를 이동시키는 버전.
        Error accumulation을 방지하기 위해 항상 첫 프레임 기준으로 계산.
        """
        import subprocess, tempfile
        from PIL import Image
        from vipe.ext.lietorch import SE3
        from vipe.utils.visualization import project_points

        logger.info("🎬 Starting flow-based dynamic object reprojection (frame 0 reference tracking)...")

        current_artifact = self.global_context().artifacts[self.gui_id.value]
        rgb_seq = list(read_rgb_artifacts(current_artifact.rgb_path))
        pose_seq = read_pose_artifacts(current_artifact.pose_path)[1]
        intr_seq = read_intrinsics_artifacts(current_artifact.intrinsics_path, current_artifact.camera_type_path)[1]
        depth_seq = list(read_depth_artifacts(current_artifact.depth_path))
        h, w, _ = rgb_seq[0][1].shape

        if not hasattr(self, "pc_world"):
            logger.warning("❌ No object point cloud found.")
            return

        obj_points = self.pc_world.copy()
        obj_colors = np.tile(np.array([[0, 255, 0]]), (obj_points.shape[0], 1))
        nearby_distance = self.gui_nearby_distance.value

        # --- 첫 프레임의 점들 저장 ---
        frame_0_depth = depth_seq[0][1].cpu().numpy()
        frame_0_intr = intr_seq[0].cpu().numpy()
        frame_0_pose = pose_seq[0].matrix().cpu().numpy()
        frame_0_mask = reliable_depth_mask_range(torch.from_numpy(frame_0_depth)).numpy()
        
        ys_0, xs_0 = np.where(frame_0_mask)
        if len(xs_0) == 0:
            logger.warning("❌ No valid points in frame 0")
            return
        
        fx_0, fy_0, cx_0, cy_0 = frame_0_intr[:4]
        R_0, t_0 = frame_0_pose[:3, :3], frame_0_pose[:3, 3]
        z_0 = frame_0_depth[ys_0, xs_0]
        X_0 = (xs_0 - cx_0) * z_0 / fx_0
        Y_0 = (ys_0 - cy_0) * z_0 / fy_0
        pts_cam_0 = np.stack([X_0, Y_0, z_0], axis=-1)
        pts_world_frame_0 = (pts_cam_0 @ R_0.T) + t_0  # 첫 프레임의 world 좌표
        
        # 첫 프레임의 pixel 좌표 저장 (추적용)
        pixel_coords_frame_0 = np.stack([xs_0, ys_0], axis=1).astype(float)  # [N, 2]
        
        logger.info(f"✅ Initialized {len(pts_world_frame_0)} reference points from frame 0")

        frames = []
        tracked_points = obj_points.copy()

        for frame_idx in range(min(len(depth_seq), pose_seq.shape[0])):
            rgb = rgb_seq[frame_idx][1].cpu().numpy()
            depth = depth_seq[frame_idx][1].cpu().numpy()
            intr = intr_seq[frame_idx].cpu().numpy()
            pose = pose_seq[frame_idx].matrix().cpu().numpy()
            fx, fy, cx, cy = intr[:4]
            R, t = pose[:3, :3], pose[:3, 3]

            # --- scene backprojection (현재 프레임) ---
            mask = reliable_depth_mask_range(torch.from_numpy(depth)).numpy()
            colors_scene = rgb if rgb.max() > 1 else (rgb * 255).astype(np.uint8)

            # --- 첫 프레임의 점들을 현재 프레임까지 추적 ---
            if frame_idx == 0:
                # 첫 프레임이면 그대로 사용
                current_pixel_coords = pixel_coords_frame_0.copy()
                valid_tracking = np.ones(len(current_pixel_coords), dtype=bool)
            else:
                # Optical flow를 연속적으로 따라가서 첫 프레임의 점들이 현재 프레임의 어디에 있는지 추적
                current_pixel_coords = pixel_coords_frame_0.copy()
                valid_tracking = np.ones(len(current_pixel_coords), dtype=bool)
                
                for flow_frame_idx in range(frame_idx):
                    if flow_frame_idx >= len(self.flow_cache) or self.flow_cache[flow_frame_idx] is None:
                        # flow가 없으면 해당 점들을 invalid로 표시
                        valid_tracking[:] = False
                        break
                    
                    flow_np = self.flow_cache[flow_frame_idx].numpy()
                    
                    # 현재 pixel 좌표가 유효한 범위 내에 있는지 확인
                    valid_bounds = (
                        (current_pixel_coords[:, 0] >= 0) & (current_pixel_coords[:, 0] < w) &
                        (current_pixel_coords[:, 1] >= 0) & (current_pixel_coords[:, 1] < h)
                    )
                    valid_tracking &= valid_bounds
                    
                    if not valid_tracking.any():
                        break
                    
                    # Optical flow로 다음 프레임의 pixel 좌표 계산
                    valid_coords = current_pixel_coords[valid_tracking].astype(int)
                    u_flow = flow_np[0, valid_coords[:, 1], valid_coords[:, 0]]
                    v_flow = flow_np[1, valid_coords[:, 1], valid_coords[:, 0]]
                    
                    # 다음 프레임의 pixel 좌표
                    current_pixel_coords[valid_tracking, 0] += u_flow
                    current_pixel_coords[valid_tracking, 1] += v_flow
                
                # 최종 유효 범위 체크
                valid_bounds = (
                    (current_pixel_coords[:, 0] >= 0) & (current_pixel_coords[:, 0] < w) &
                    (current_pixel_coords[:, 1] >= 0) & (current_pixel_coords[:, 1] < h)
                )
                valid_tracking &= valid_bounds

            # --- Scene backprojection (렌더링용) ---
            ys, xs = np.where(mask)
            if len(xs) > 0:
                z = depth[ys, xs]
                X = (xs - cx) * z / fx
                Y = (ys - cy) * z / fy
                pts_cam = np.stack([X, Y, z], axis=-1)
                pts_world_scene = (pts_cam @ R.T) + t
                colors_scene_curr = colors_scene[ys, xs] if colors_scene.ndim == 3 else colors_scene
            else:
                pts_world_scene = np.empty((0, 3))
                colors_scene_curr = np.empty((0, 3))

            # --- 유효한 추적만 사용하여 현재 프레임의 3D 점 계산 ---
            if valid_tracking.sum() == 0:
                logger.warning(f"  frame {frame_idx}: No valid tracked points")
                # Object는 첫 프레임 위치 유지
                tracked_points = obj_points.copy()
                motion_vectors = np.empty((0, 3))
            else:
                # 유효한 추적된 pixel 좌표
                valid_coords = current_pixel_coords[valid_tracking].astype(int)
                
                # 현재 프레임에서의 depth 가져오기
                z_curr = depth[valid_coords[:, 1], valid_coords[:, 0]]
                depth_mask_curr = reliable_depth_mask_range(torch.from_numpy(depth)).numpy()
                valid_depth_curr = depth_mask_curr[valid_coords[:, 1], valid_coords[:, 0]] & (z_curr > 0)
                
                if valid_depth_curr.sum() == 0:
                    logger.warning(f"  frame {frame_idx}: No valid depth for tracked points")
                    pts_world_curr = np.empty((0, 3))
                    colors_scene_curr = np.empty((0, 3))
                    motion_vectors = np.empty((0, 3))
                else:
                    # 유효한 depth만 사용
                    valid_final = valid_tracking.copy()
                    valid_final[valid_tracking] = valid_depth_curr
                    
                    z_curr_valid = z_curr[valid_depth_curr]
                    valid_coords_final = valid_coords[valid_depth_curr]
                    
                    # 현재 프레임의 3D 점들 (world 좌표)
                    X_curr = (valid_coords_final[:, 0] - cx) * z_curr_valid / fx
                    Y_curr = (valid_coords_final[:, 1] - cy) * z_curr_valid / fy
                    pts_cam_curr = np.stack([X_curr, Y_curr, z_curr_valid], axis=-1)
                    pts_world_curr = (pts_cam_curr @ R.T) + t
                    
                    # 첫 프레임 기준의 motion vector 계산
                    pts_world_frame_0_valid = pts_world_frame_0[valid_final]
                    motion_vectors = pts_world_curr - pts_world_frame_0_valid  # [N, 3]
                    
                    # Object 중심점 계산 (첫 프레임 기준 초기 위치 + 현재 motion)
                    if frame_idx == 0:
                        obj_center_ref = obj_points.mean(axis=0)
                    else:
                        # Object의 초기 위치 (첫 프레임에서)
                        obj_center_ref = obj_points.mean(axis=0)
                    
                    # Object 근처의 점들 찾기 (첫 프레임 기준 거리)
                    distances_to_obj = np.linalg.norm(pts_world_frame_0_valid - obj_center_ref, axis=1)
                    nearby_mask = distances_to_obj <= nearby_distance
                    
                    if nearby_mask.sum() > 0:
                        # 근처 점들의 motion vector 평균 계산
                        nearby_motion_vectors = motion_vectors[nearby_mask]
                        avg_motion = np.mean(nearby_motion_vectors, axis=0)
                        
                        # Object를 첫 프레임 위치에서 평균 motion만큼 이동 (error accumulation 방지)
                        tracked_points = obj_points + avg_motion
                        
                        logger.info(
                            f"  frame {frame_idx+1}/{len(depth_seq)}: "
                            f"tracked {valid_final.sum()}/{len(valid_tracking)} points, "
                            f"found {nearby_mask.sum()} nearby points, "
                            f"avg motion from frame 0 = [{avg_motion[0]:.3f}, {avg_motion[1]:.3f}, {avg_motion[2]:.3f}]"
                        )
                    else:
                        logger.warning(f"  frame {frame_idx+1}: No nearby points found (distance={nearby_distance:.2f})")
                        # Object는 첫 프레임 위치 유지
                        tracked_points = obj_points.copy()
                    
            # --- scene + object merge ---
            all_points = np.concatenate([pts_world_scene, tracked_points], axis=0)
            all_colors = np.concatenate([colors_scene_curr, obj_colors], axis=0)

            # --- projection ---
            img = project_points(
                xyz=all_points,
                intrinsics=intr,
                camera_type=CameraType.PINHOLE,
                pose=pose_seq[frame_idx],
                frame_size=(h, w),
                subsample_factor=1,
                color=all_colors,
            )

            frames.append(img)
            Image.fromarray(img).save(f"/home/nas_main/kinamkim/Repos/vipe/vipe/check/{frame_idx:04d}.png")

        # --- save video ---
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            for i, frame in enumerate(frames):
                Image.fromarray(frame).save(tmp_path / f"{i:04d}.png")
            subprocess.run([
                "ffmpeg", "-y", "-framerate", "30",
                "-i", str(tmp_path / "%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", save_path
            ], check=True)

        logger.info(f"✅ Saved translation-only tracked video to {save_path}")

    def reproject_single_frame(self, frame_idx: int):
        """
        특정 프레임만 reproject하여 이미지로 저장합니다.
        첫 프레임 기준으로 포인트들을 추적하여 object를 이동시킵니다.
        
        Args:
            frame_idx: reproject할 프레임 인덱스
        """
        from pathlib import Path
        from PIL import Image
        from vipe.utils.visualization import project_points
        
        logger.info(f"🎬 Reprojecting single frame {frame_idx}...")
        
        current_artifact = self.global_context().artifacts[self.gui_id.value]
        rgb_seq = list(read_rgb_artifacts(current_artifact.rgb_path))
        pose_seq = read_pose_artifacts(current_artifact.pose_path)[1]
        intr_seq = read_intrinsics_artifacts(current_artifact.intrinsics_path, current_artifact.camera_type_path)[1]
        depth_seq = list(read_depth_artifacts(current_artifact.depth_path))
        
        if frame_idx >= len(rgb_seq) or frame_idx >= len(depth_seq):
            logger.warning(f"❌ Frame index {frame_idx} out of range")
            return
        
        if not hasattr(self, "pc_world"):
            logger.warning("❌ No object point cloud found.")
            return
        
        h, w, _ = rgb_seq[0][1].shape
        
        obj_points = self.pc_world.copy()
        obj_colors = np.tile(np.array([[0, 255, 0]]), (obj_points.shape[0], 1))
        nearby_distance = self.gui_nearby_distance.value
        
        # --- 첫 프레임의 점들 저장 ---
        frame_0_depth = depth_seq[0][1].cpu().numpy()
        frame_0_intr = intr_seq[0].cpu().numpy()
        frame_0_pose = pose_seq[0].matrix().cpu().numpy()
        frame_0_mask = reliable_depth_mask_range(torch.from_numpy(frame_0_depth)).numpy()
        
        ys_0, xs_0 = np.where(frame_0_mask)
        if len(xs_0) == 0:
            logger.warning("❌ No valid points in frame 0")
            return
        
        fx_0, fy_0, cx_0, cy_0 = frame_0_intr[:4]
        R_0, t_0 = frame_0_pose[:3, :3], frame_0_pose[:3, 3]
        z_0 = frame_0_depth[ys_0, xs_0]
        X_0 = (xs_0 - cx_0) * z_0 / fx_0
        Y_0 = (ys_0 - cy_0) * z_0 / fy_0
        pts_cam_0 = np.stack([X_0, Y_0, z_0], axis=-1)
        pts_world_frame_0 = (pts_cam_0 @ R_0.T) + t_0  # 첫 프레임의 world 좌표
        
        # 첫 프레임의 pixel 좌표 저장 (추적용)
        pixel_coords_frame_0 = np.stack([xs_0, ys_0], axis=1).astype(float)  # [N, 2]
        
        # 현재 프레임 데이터
        rgb = rgb_seq[frame_idx][1].cpu().numpy()
        depth = depth_seq[frame_idx][1].cpu().numpy()
        intr = intr_seq[frame_idx].cpu().numpy()
        pose = pose_seq[frame_idx].matrix().cpu().numpy()
        fx, fy, cx, cy = intr[:4]
        R, t = pose[:3, :3], pose[:3, 3]
        
        # Scene backprojection
        mask = reliable_depth_mask_range(torch.from_numpy(depth)).numpy()
        colors_scene = rgb if rgb.max() > 1 else (rgb * 255).astype(np.uint8)
        
        ys, xs = np.where(mask)
        if len(xs) > 0:
            z = depth[ys, xs]
            X = (xs - cx) * z / fx
            Y = (ys - cy) * z / fy
            pts_cam = np.stack([X, Y, z], axis=-1)
            pts_world_scene = (pts_cam @ R.T) + t
            colors_scene_curr = colors_scene[ys, xs] if colors_scene.ndim == 3 else colors_scene
        else:
            pts_world_scene = np.empty((0, 3))
            colors_scene_curr = np.empty((0, 3))
        
        # --- 첫 프레임의 점들을 현재 프레임까지 추적 ---
        if frame_idx == 0:
            # 첫 프레임이면 그대로 사용
            current_pixel_coords = pixel_coords_frame_0.copy()
            valid_tracking = np.ones(len(current_pixel_coords), dtype=bool)
        else:
            # Optical flow를 연속적으로 따라가서 첫 프레임의 점들이 현재 프레임의 어디에 있는지 추적
            current_pixel_coords = pixel_coords_frame_0.copy()
            valid_tracking = np.ones(len(current_pixel_coords), dtype=bool)
            
            for flow_frame_idx in range(frame_idx):
                if flow_frame_idx >= len(self.flow_cache) or self.flow_cache[flow_frame_idx] is None:
                    # flow가 없으면 해당 점들을 invalid로 표시
                    valid_tracking[:] = False
                    break
                
                flow_np = self.flow_cache[flow_frame_idx].numpy()
                
                # 현재 pixel 좌표가 유효한 범위 내에 있는지 확인
                valid_bounds = (
                    (current_pixel_coords[:, 0] >= 0) & (current_pixel_coords[:, 0] < w) &
                    (current_pixel_coords[:, 1] >= 0) & (current_pixel_coords[:, 1] < h)
                )
                valid_tracking &= valid_bounds
                
                if not valid_tracking.any():
                    break
                
                # Optical flow로 다음 프레임의 pixel 좌표 계산
                valid_coords = current_pixel_coords[valid_tracking].astype(int)
                u_flow = flow_np[0, valid_coords[:, 1], valid_coords[:, 0]]
                v_flow = flow_np[1, valid_coords[:, 1], valid_coords[:, 0]]
                
                # 다음 프레임의 pixel 좌표
                current_pixel_coords[valid_tracking, 0] += u_flow
                current_pixel_coords[valid_tracking, 1] += v_flow
            
            # 최종 유효 범위 체크
            valid_bounds = (
                (current_pixel_coords[:, 0] >= 0) & (current_pixel_coords[:, 0] < w) &
                (current_pixel_coords[:, 1] >= 0) & (current_pixel_coords[:, 1] < h)
            )
            valid_tracking &= valid_bounds
        
        # --- 관성 시작 프레임 확인 ---
        inertia_start_frame = getattr(self, "gui_inertia_start_frame", None)
        if inertia_start_frame is not None:
            inertia_start_frame_value = int(inertia_start_frame.value)
        else:
            inertia_start_frame_value = -1  # 비활성화
        
        use_inertia = (inertia_start_frame_value >= 0 and frame_idx >= inertia_start_frame_value)
        
        # --- 유효한 추적만 사용하여 현재 프레임의 3D 점 계산 ---
        if valid_tracking.sum() == 0:
            logger.warning(f"  frame {frame_idx}: No valid tracked points")
            # Object는 첫 프레임 위치 유지
            tracked_points = obj_points.copy()
        else:
            if use_inertia:
                # 관성 적용: inertia_start_frame에서의 모션 벡터를 선형 확장
                logger.info(f"  frame {frame_idx}: Using inertia from frame {inertia_start_frame_value}")
                
                # Inertia 시작 프레임의 데이터 가져오기
                inertia_depth = depth_seq[inertia_start_frame_value][1].cpu().numpy()
                inertia_intr = intr_seq[inertia_start_frame_value].cpu().numpy()
                inertia_pose = pose_seq[inertia_start_frame_value].matrix().cpu().numpy()
                inertia_fx, inertia_fy, inertia_cx, inertia_cy = inertia_intr[:4]
                inertia_R, inertia_t = inertia_pose[:3, :3], inertia_pose[:3, 3]
                
                # Inertia 시작 프레임까지의 pixel 좌표 추적
                inertia_pixel_coords = pixel_coords_frame_0.copy()
                inertia_valid_tracking = np.ones(len(inertia_pixel_coords), dtype=bool)
                
                for flow_frame_idx in range(inertia_start_frame_value):
                    if flow_frame_idx >= len(self.flow_cache) or self.flow_cache[flow_frame_idx] is None:
                        inertia_valid_tracking[:] = False
                        break
                    
                    flow_np = self.flow_cache[flow_frame_idx].numpy()
                    valid_bounds = (
                        (inertia_pixel_coords[:, 0] >= 0) & (inertia_pixel_coords[:, 0] < w) &
                        (inertia_pixel_coords[:, 1] >= 0) & (inertia_pixel_coords[:, 1] < h)
                    )
                    inertia_valid_tracking &= valid_bounds
                    
                    if not inertia_valid_tracking.any():
                        break
                    
                    valid_coords = inertia_pixel_coords[inertia_valid_tracking].astype(int)
                    u_flow = flow_np[0, valid_coords[:, 1], valid_coords[:, 0]]
                    v_flow = flow_np[1, valid_coords[:, 1], valid_coords[:, 0]]
                    
                    inertia_pixel_coords[inertia_valid_tracking, 0] += u_flow
                    inertia_pixel_coords[inertia_valid_tracking, 1] += v_flow
                
                # 최종 유효 범위 체크
                valid_bounds = (
                    (inertia_pixel_coords[:, 0] >= 0) & (inertia_pixel_coords[:, 0] < w) &
                    (inertia_pixel_coords[:, 1] >= 0) & (inertia_pixel_coords[:, 1] < h)
                )
                inertia_valid_tracking &= valid_bounds
                
                if inertia_valid_tracking.sum() == 0:
                    logger.warning(f"  frame {frame_idx}: No valid tracked points at inertia start frame {inertia_start_frame_value}")
                    tracked_points = obj_points.copy()
                else:
                    # Inertia 시작 프레임의 3D 점 계산
                    inertia_valid_coords = inertia_pixel_coords[inertia_valid_tracking].astype(int)
                    inertia_z_curr = inertia_depth[inertia_valid_coords[:, 1], inertia_valid_coords[:, 0]]
                    inertia_depth_mask = reliable_depth_mask_range(torch.from_numpy(inertia_depth)).numpy()
                    inertia_valid_depth = inertia_depth_mask[inertia_valid_coords[:, 1], inertia_valid_coords[:, 0]] & (inertia_z_curr > 0)
                    
                    if inertia_valid_depth.sum() == 0:
                        logger.warning(f"  frame {frame_idx}: No valid depth at inertia start frame {inertia_start_frame_value}")
                        tracked_points = obj_points.copy()
                    else:
                        inertia_valid_final = inertia_valid_tracking.copy()
                        inertia_valid_final[inertia_valid_tracking] = inertia_valid_depth
                        
                        inertia_z_valid = inertia_z_curr[inertia_valid_depth]
                        inertia_valid_coords_final = inertia_valid_coords[inertia_valid_depth]
                        
                        # Inertia 시작 프레임의 3D 점들 (world 좌표)
                        inertia_X_curr = (inertia_valid_coords_final[:, 0] - inertia_cx) * inertia_z_valid / inertia_fx
                        inertia_Y_curr = (inertia_valid_coords_final[:, 1] - inertia_cy) * inertia_z_valid / inertia_fy
                        inertia_pts_cam_curr = np.stack([inertia_X_curr, inertia_Y_curr, inertia_z_valid], axis=-1)
                        inertia_pts_world_curr = (inertia_pts_cam_curr @ inertia_R.T) + inertia_t
                        
                        # 첫 프레임 기준의 motion vector 계산 (inertia 시작 프레임에서)
                        inertia_pts_world_frame_0_valid = pts_world_frame_0[inertia_valid_final]
                        inertia_motion_vectors = inertia_pts_world_curr - inertia_pts_world_frame_0_valid  # [N, 3]
                        
                        # Object 중심점 계산
                        obj_center_ref = obj_points.mean(axis=0)
                        
                        # Object 근처의 점들 찾기 (첫 프레임 기준 거리)
                        inertia_distances_to_obj = np.linalg.norm(inertia_pts_world_frame_0_valid - obj_center_ref, axis=1)
                        inertia_nearby_mask = inertia_distances_to_obj <= nearby_distance
                        
                        if inertia_nearby_mask.sum() > 0:
                            # 근처 점들의 motion vector 평균 계산 (inertia 시작 프레임에서)
                            inertia_nearby_motion_vectors = inertia_motion_vectors[inertia_nearby_mask]
                            inertia_avg_motion = np.mean(inertia_nearby_motion_vectors, axis=0)
                            
                            # 선형 확장: (frame_idx / inertia_start_frame_value) 배만큼 확장
                            if inertia_start_frame_value > 0:
                                scale_factor = frame_idx / inertia_start_frame_value
                            else:
                                scale_factor = 1.0
                            
                            extended_motion = inertia_avg_motion * scale_factor
                            
                            # Object를 첫 프레임 위치에서 확장된 motion만큼 이동
                            tracked_points = obj_points + extended_motion
                            
                            logger.info(
                                f"  frame {frame_idx}: INERTIA MODE "
                                f"(base frame {inertia_start_frame_value}, scale={scale_factor:.2f}) "
                                f"tracked {inertia_valid_final.sum()}/{len(inertia_valid_tracking)} points, "
                                f"found {inertia_nearby_mask.sum()} nearby points, "
                                f"base motion = [{inertia_avg_motion[0]:.3f}, {inertia_avg_motion[1]:.3f}, {inertia_avg_motion[2]:.3f}], "
                                f"extended motion = [{extended_motion[0]:.3f}, {extended_motion[1]:.3f}, {extended_motion[2]:.3f}]"
                            )
                        else:
                            logger.warning(f"  frame {frame_idx}: No nearby points at inertia start frame {inertia_start_frame_value}")
                            tracked_points = obj_points.copy()
            else:
                # 기존 로직: 현재 프레임에서 모션 벡터 계산
                # 유효한 추적된 pixel 좌표
                valid_coords = current_pixel_coords[valid_tracking].astype(int)
                
                # 현재 프레임에서의 depth 가져오기
                z_curr = depth[valid_coords[:, 1], valid_coords[:, 0]]
                depth_mask_curr = reliable_depth_mask_range(torch.from_numpy(depth)).numpy()
                valid_depth_curr = depth_mask_curr[valid_coords[:, 1], valid_coords[:, 0]] & (z_curr > 0)
                
                if valid_depth_curr.sum() == 0:
                    logger.warning(f"  frame {frame_idx}: No valid depth for tracked points")
                    tracked_points = obj_points.copy()
                else:
                    # 유효한 depth만 사용
                    valid_final = valid_tracking.copy()
                    valid_final[valid_tracking] = valid_depth_curr
                    
                    z_curr_valid = z_curr[valid_depth_curr]
                    valid_coords_final = valid_coords[valid_depth_curr]
                    
                    # 현재 프레임의 3D 점들 (world 좌표)
                    X_curr = (valid_coords_final[:, 0] - cx) * z_curr_valid / fx
                    Y_curr = (valid_coords_final[:, 1] - cy) * z_curr_valid / fy
                    pts_cam_curr = np.stack([X_curr, Y_curr, z_curr_valid], axis=-1)
                    pts_world_curr = (pts_cam_curr @ R.T) + t
                    
                    # 첫 프레임 기준의 motion vector 계산
                    pts_world_frame_0_valid = pts_world_frame_0[valid_final]
                    motion_vectors = pts_world_curr - pts_world_frame_0_valid  # [N, 3]
                    
                    # Object 중심점 계산
                    obj_center_ref = obj_points.mean(axis=0)
                    
                    # Object 근처의 점들 찾기 (첫 프레임 기준 거리)
                    distances_to_obj = np.linalg.norm(pts_world_frame_0_valid - obj_center_ref, axis=1)
                    nearby_mask = distances_to_obj <= nearby_distance
                    
                    if nearby_mask.sum() > 0:
                        # 근처 점들의 motion vector 평균 계산
                        nearby_motion_vectors = motion_vectors[nearby_mask]
                        avg_motion = np.mean(nearby_motion_vectors, axis=0)
                        
                        # Object를 첫 프레임 위치에서 평균 motion만큼 이동
                        tracked_points = obj_points + avg_motion
                        
                        logger.info(
                            f"  frame {frame_idx}: "
                            f"tracked {valid_final.sum()}/{len(valid_tracking)} points, "
                            f"found {nearby_mask.sum()} nearby points, "
                            f"avg motion from frame 0 = [{avg_motion[0]:.3f}, {avg_motion[1]:.3f}, {avg_motion[2]:.3f}]"
                        )
                    else:
                        logger.warning(f"  frame {frame_idx}: No nearby points found (distance={nearby_distance:.2f})")
                        # Object는 첫 프레임 위치 유지
                        tracked_points = obj_points.copy()
        
        # Scene + object merge
        all_points = np.concatenate([pts_world_scene, tracked_points], axis=0)
        all_colors = np.concatenate([colors_scene_curr, obj_colors], axis=0)
        
        # Projection
        img = project_points(
            xyz=all_points,
            intrinsics=intr,
            camera_type=CameraType.PINHOLE,
            pose=pose_seq[frame_idx],
            frame_size=(h, w),
            subsample_factor=1,
            color=all_colors,
        )
        
        # 별도 폴더에 저장
        save_dir = Path("single_frame_reprojections")
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / f"frame_{frame_idx:04d}_reproject.png"
        Image.fromarray(img).save(save_path)
        
        logger.info(f"✅ Saved reprojected frame to {save_path}")

    def match_3d_points_with_optical_flow(self, frame_idx_i: int, frame_idx_j: int):
        """
        Optical flow를 이용해서 두 프레임 간의 3D 점들을 매칭합니다.
        
        Args:
            frame_idx_i: 첫 번째 프레임 인덱스
            frame_idx_j: 두 번째 프레임 인덱스 (일반적으로 frame_idx_i + 1)
            
        Returns:
            matched_pairs: numpy array of shape [N, 2, 3] where each row contains [pt3d_i, pt3d_j] in world coordinates
            valid_mask: Boolean array of shape [N] indicating which matches are valid (currently always True for returned pairs)
        """
        current_artifact = self.global_context().artifacts[self.gui_id.value]
        depth_seq = list(read_depth_artifacts(current_artifact.depth_path))
        pose_seq = read_pose_artifacts(current_artifact.pose_path)[1]
        intr_seq = read_intrinsics_artifacts(current_artifact.intrinsics_path, current_artifact.camera_type_path)[1]
        
        if frame_idx_i >= len(depth_seq) or frame_idx_j >= len(depth_seq):
            logger.warning(f"❌ Frame indices out of range: {frame_idx_i}, {frame_idx_j}")
            return None, None
            
        if frame_idx_j != frame_idx_i + 1:
            logger.warning(f"⚠️ frame_idx_j should be frame_idx_i + 1 for optical flow matching")
            # 연속된 프레임이 아니면 매칭할 수 없음
            return None, None
        
        # Optical flow 가져오기
        if frame_idx_i >= len(self.flow_cache) or self.flow_cache[frame_idx_i] is None:
            logger.warning(f"❌ No optical flow available for frame pair ({frame_idx_i}, {frame_idx_j})")
            return None, None
        
        flow = self.flow_cache[frame_idx_i]  # flow from frame_i to frame_j
        flow_np = flow.numpy()  # [2, H, W]
        
        # Depth와 pose 가져오기
        depth_i = depth_seq[frame_idx_i][1].cpu().numpy()
        depth_j = depth_seq[frame_idx_j][1].cpu().numpy()
        intr_i = intr_seq[frame_idx_i].cpu().numpy()
        intr_j = intr_seq[frame_idx_j].cpu().numpy()
        pose_i = pose_seq[frame_idx_i].matrix().cpu().numpy()
        pose_j = pose_seq[frame_idx_j].matrix().cpu().numpy()
        
        h, w = depth_i.shape
        fx_i, fy_i, cx_i, cy_i = intr_i[:4]
        fx_j, fy_j, cx_j, cy_j = intr_j[:4]
        R_i, t_i = pose_i[:3, :3], pose_i[:3, 3]
        R_j, t_j = pose_j[:3, :3], pose_j[:3, 3]
        
        # Frame i의 depth mask
        mask_i = reliable_depth_mask_range(torch.from_numpy(depth_i)).numpy()
        
        # 유효한 픽셀 위치들
        ys_i, xs_i = np.where(mask_i)
        if len(xs_i) == 0:
            logger.warning("❌ No valid depth pixels in frame_i")
            return None, None
        
        # Frame i의 3D 점들 (camera space)
        z_i = depth_i[ys_i, xs_i]
        X_i = (xs_i - cx_i) * z_i / fx_i
        Y_i = (ys_i - cy_i) * z_i / fy_i
        pts_cam_i = np.stack([X_i, Y_i, z_i], axis=-1)
        
        # Frame i의 3D 점들 (world space)
        pts_world_i = (pts_cam_i @ R_i.T) + t_i
        
        # Optical flow로 frame j에서의 픽셀 위치 찾기
        u_disp = flow_np[0, ys_i, xs_i]  # flow[0] = u displacement
        v_disp = flow_np[1, ys_i, xs_i]  # flow[1] = v displacement
        
        xs_j = xs_i + u_disp
        ys_j = ys_i + v_disp
        
        # 유효 범위 체크
        valid_bounds = (
            (xs_j >= 0) & (xs_j < w) &
            (ys_j >= 0) & (ys_j < h)
        )
        
        if valid_bounds.sum() == 0:
            logger.warning("❌ No valid matches after optical flow")
            return None, None
        
        # 유효한 매칭만 사용
        xs_j_valid = xs_j[valid_bounds].astype(int)
        ys_j_valid = ys_j[valid_bounds].astype(int)
        pts_world_i_valid = pts_world_i[valid_bounds]
        
        # Frame j에서 해당 픽셀 위치의 depth 가져오기
        z_j = depth_j[ys_j_valid, xs_j_valid]
        
        # Depth 유효성 체크 (너무 차이나는 depth는 제외)
        depth_mask_j = reliable_depth_mask_range(torch.from_numpy(depth_j)).numpy()
        valid_depth_j = depth_mask_j[ys_j_valid, xs_j_valid]
        
        # Frame j의 3D 점들 (camera space)
        X_j = (xs_j_valid - cx_j) * z_j / fx_j
        Y_j = (ys_j_valid - cy_j) * z_j / fy_j
        pts_cam_j = np.stack([X_j, Y_j, z_j], axis=-1)
        
        # Frame j의 3D 점들 (world space)
        pts_world_j = (pts_cam_j @ R_j.T) + t_j
        
        # 최종 유효성 마스크 (depth가 유효한 경우만)
        final_valid = valid_depth_j & (z_j > 0)
        
        matched_pairs = np.stack([pts_world_i_valid[final_valid], pts_world_j[final_valid]], axis=1)
        
        # valid_mask는 반환되는 모든 매칭이 유효하므로 모두 True
        valid_mask = np.ones(len(matched_pairs), dtype=bool)
        
        logger.info(
            f"✅ Matched {final_valid.sum()}/{len(valid_bounds)} 3D points "
            f"between frames {frame_idx_i} and {frame_idx_j}"
        )
        
        return matched_pairs, valid_mask

    def compute_optical_flow(self, frame1, frame2):
        """
        Compute optical flow between two RGB frames using RAFT.
        Returns tensor [2, H_orig, W_orig] (u,v) resized back to original frame size.
        """
        H_orig, W_orig = frame1.shape[1], frame1.shape[2]  # TCHW 포맷이므로
        target_size = [520, 960]

        # --- Resize for RAFT input ---
        img1 = F.resize(frame1, size=target_size, antialias=False)
        img2 = F.resize(frame2, size=target_size, antialias=False)
        img1, img2 = self.raft_transforms(img1.unsqueeze(0), img2.unsqueeze(0))

        with torch.no_grad():
            list_of_flows = self.raft_model(img1.to(self.device), img2.to(self.device))
        flow = list_of_flows[-1][0].cpu()  # [2, H_raft, W_raft]

        # --- Rescale flow back to original resolution ---
        scale_y = H_orig / target_size[0]
        scale_x = W_orig / target_size[1]

        flow = F.resize(flow, size=[H_orig, W_orig], antialias=False)  # bilinear upsample
        flow[0] *= scale_x
        flow[1] *= scale_y

        return flow  # [2, H_orig, W_orig]

    def precompute_optical_flow_all(self, rgb_seq):
        """
        Compute RAFT optical flow for all consecutive frame pairs in advance.
        Stores results in self.flow_cache[frame_idx] = flow (torch.Tensor).
        """
        self.flow_cache = []
        device = self.device
        total = len(rgb_seq) - 1
        logger.info(f"🚀 Starting optical flow precomputation for {total} frame pairs...")

        for i in range(total):
            try:
                frame1 = rgb_seq[i][1].permute(2, 0, 1)  # (H,W,3)->(3,H,W)
                frame2 = rgb_seq[i + 1][1].permute(2, 0, 1)
                flow = self.compute_optical_flow(frame1, frame2)
                self.flow_cache.append(flow)
                logger.info(f"✅ Computed flow {i}/{total-1}")
            except Exception as e:
                self.flow_cache.append(None)
                logger.warning(f"⚠️ Flow computation failed at pair {i}: {e}")

        logger.info(f"✅ Optical flow precomputation finished ({len(self.flow_cache)} valid flows)")

    def _rebuild_scene(self):
        current_artifact = self.global_context().artifacts[self.gui_id.value]
        spatial_subsample: int = self.gui_s_sub.value
        temporal_subsample: int = self.gui_t_sub.value

        rays: np.ndarray | None = None
        first_frame_y: np.ndarray | None = None

        self.client.scene.reset()
        self.client.camera.fov = np.deg2rad(self.gui_fov.value)
        self.scene_frame_handles = []

        def none_it(inner_it):
            try:
                for item in inner_it:
                    yield item
            except FileNotFoundError:
                while True:
                    yield None, None

        for frame_idx, (c2w, (_, rgb), intr, camera_type, (_, depth)) in enumerate(
            zip(
                read_pose_artifacts(current_artifact.pose_path)[1].matrix().numpy(),
                read_rgb_artifacts(current_artifact.rgb_path),
                *read_intrinsics_artifacts(current_artifact.intrinsics_path, current_artifact.camera_type_path)[1:3],
                none_it(read_depth_artifacts(current_artifact.depth_path)),
            )
        ):
            if frame_idx % temporal_subsample != 0:
                continue

            pinhole_intr = camera_type.build_camera_model(intr).pinhole().intrinsics
            frame_height, frame_width = rgb.shape[:2]
            fov = 2 * np.arctan2(frame_height / 2, pinhole_intr[0].item())

            sampled_rgb = (rgb.cpu().numpy() * 255).astype(np.uint8)
            sampled_rgb = sampled_rgb[::spatial_subsample, ::spatial_subsample]

            if first_frame_y is None:
                first_frame_y = c2w[:3, 1]
                self.client.scene.set_up_direction(-first_frame_y)

            if rays is None:
                camera_model = camera_type.build_camera_model(intr)
                disp_v, disp_u = torch.meshgrid(
                    torch.arange(frame_height).float()[::spatial_subsample],
                    torch.arange(frame_width).float()[::spatial_subsample],
                    indexing="ij",
                )
                if camera_type == CameraType.PANORAMA:
                    disp_v = disp_v / (frame_height - 1)
                    disp_u = disp_u / (frame_width - 1)
                disp = torch.ones_like(disp_v)
                pts, _, _ = camera_model.iproj_disp(disp, disp_u, disp_v)
                rays = pts[..., :3].numpy()
                if camera_type != CameraType.PANORAMA:
                    rays /= rays[..., 2:3]

            if depth is not None:
                pcd = rays * depth.numpy()[::spatial_subsample, ::spatial_subsample, None]
                depth_mask = reliable_depth_mask_range(depth)[::spatial_subsample, ::spatial_subsample].numpy()
            else:
                pcd, depth_mask = None, None

            frame_node = self._make_frame_nodes(
                frame_idx,
                c2w,
                sampled_rgb,
                fov,
                pcd,
                depth_mask,
            )
            self.scene_frame_handles.append(frame_node)
        
        pcd = o3d.io.read_point_cloud(
            "/home/nas_main/kinamkim/Repos/vipe/2125_2_cut1/2023_hyundai_creta_more.ply"
            #"/home/nas4_user/hoiyeongjin/repos/Project/SKT/mesh2pointcloud/data/stanley/stanley_more.ply",
            #"/home/nas4_user/hoiyeongjin/repos/Project/SKT/mesh2pointcloud/data/fanta2/330ml_can_of_fanta_coconut.ply"
        )
        self.pc_raw = np.asarray(pcd.points)
        pc_colors = np.tile(np.array([[0, 255, 0]]), (self.pc_raw.shape[0], 1))

        base_frame = self.scene_frame_handles[0]
        R = tf.SO3(base_frame.frame_handle.wxyz).as_matrix()
        t = np.array(base_frame.frame_handle.position)
        #offset = np.array([0.0, 0.3, 0.0])

        pc_world = (R @ (self.pc_raw * 0.5).T).T + t#+ offset

        self.client.scene.add_point_cloud(
            name="/custom_object/green_object",
            points=pc_world,
            colors=pc_colors,
            point_size=0.005,
        )
            
    def _make_frame_nodes(
        self,
        frame_idx: int,
        c2w: np.ndarray,
        rgb: np.ndarray,
        fov: float,
        pcd: np.ndarray | None,
        pcd_mask: np.ndarray | None = None,
    ) -> SceneFrameHandle:
        handle = self.client.scene.add_frame(
            f"/frames/t{frame_idx}",
            axes_length=0.05,
            axes_radius=0.005,
            wxyz=tf.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, 3],
        )
        frame_height, frame_width = rgb.shape[:2]

        frame_thumbnail = Image.fromarray(rgb)
        frame_thumbnail.thumbnail((200, 200), Image.Resampling.LANCZOS)
        frustum_handle = self.client.scene.add_camera_frustum(
            f"/frames/t{frame_idx}/frustum",
            fov=fov,
            aspect=frame_width / frame_height,
            scale=self.gui_frustum_size.value,
            image=np.array(frame_thumbnail),
        )

        if pcd is not None:
            pcd = pcd.reshape(-1, 3)
            rgb = rgb.reshape(-1, 3)
            if pcd_mask is not None:
                pcd_mask = pcd_mask.reshape(-1)
                pcd = pcd[pcd_mask]
                rgb = rgb[pcd_mask]
            pcd_world = (c2w[:3, :3] @ pcd.T).T + c2w[:3, 3]
            pcd_handle = self.client.scene.add_point_cloud(
                name=f"/frames/t{frame_idx}/point_cloud",
                points=pcd_world,
                colors=rgb,
                point_size=self.gui_point_size.value,
                point_shape="rounded",
            )
        else:
            pcd_handle = None

        return SceneFrameHandle(
            frame_handle=handle,
            frustum_handle=frustum_handle,
            pcd_handle=pcd_handle,
        )

    def _incr_timestep(self):
        if self.gui_timestep is not None:
            self.gui_timestep.value = (self.gui_timestep.value + 1) % len(self.scene_frame_handles)

    def _decr_timestep(self):
        if self.gui_timestep is not None:
            self.gui_timestep.value = (self.gui_timestep.value - 1) % len(self.scene_frame_handles)

    def _rebuild_playback_gui(self):
        current_artifact = self.global_context().artifacts[self.gui_id.value]
        self.gui_name.value = current_artifact.artifact_name
        if self.gui_playback_handle is not None:
            self.gui_playback_handle.remove()

        self.gui_playback_handle = self.client.gui.add_folder("Playback")

        with self.gui_playback_handle:
            self.gui_timestep = self.client.gui.add_slider(
                "Timeline", min=0, max=len(self.scene_frame_handles) - 1, step=1, initial_value=0
            )

            gui_frame_control = self.client.gui.add_button_group("Control", options=["Prev", "Next"])
            gui_play_toggle = self.client.gui.add_button_group("Playback", options=["Play", "Pause"])

            self.gui_framerate = self.client.gui.add_slider("FPS", min=0, max=30, step=1.0, initial_value=15)

            @gui_frame_control.on_click
            async def _(_) -> None:
                if gui_frame_control.value == "Prev":
                    self._decr_timestep()
                else:
                    self._incr_timestep()

            # ✅ Play / Pause 버튼 동작
            @gui_play_toggle.on_click
            async def _(_) -> None:
                if gui_play_toggle.value == "Play":
                    self.is_playing = True
                    logger.info("▶️ Playback started")
                else:
                    self.is_playing = False
                    logger.info("⏸ Playback paused")

            self.current_displayed_timestep = self.gui_timestep.value

            @self.gui_timestep.on_update
            async def _(_) -> None:
                current_timestep = self.gui_timestep.value
                prev_timestep = self.current_displayed_timestep
                with self.client.atomic():
                    self.scene_frame_handles[current_timestep].visible = True
                    self.scene_frame_handles[prev_timestep].visible = False
                self.current_displayed_timestep = current_timestep


    def cleanup(self):
        logger.info(f"Client {self.client.client_id} disconnected")

    @classmethod
    def global_context(cls) -> GlobalContext:
        global _global_context
        assert _global_context is not None, "Global context not initialized"
        return _global_context


def get_host_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            # Doesn't even have to be reachable
            s.connect(("8.8.8.8", 1))
            internal_ip = s.getsockname()[0]
        except Exception:
            internal_ip = "127.0.0.1"
    return internal_ip


def run_viser(base_path: Path, port: int = 20540, obj: str = None):
    # Get list of artifacts.
    logger.info(f"Loading artifacts from {base_path}")
    artifacts: list[ArtifactPath] = list(ArtifactPath.glob_artifacts(base_path, use_video=True))
    if len(artifacts) == 0:
        logger.error("No artifacts found. Exiting.")
        return

    global _global_context
    _global_context = GlobalContext(artifacts=sorted(artifacts, key=lambda x: x.artifact_name))

    server = viser.ViserServer(host="0.0.0.0", port=port, verbose=False)
    client_closures: dict[int, ClientClosures] = {}

    @server.on_client_connect
    async def _(client: viser.ClientHandle):
        client_closures[client.client_id] = ClientClosures(client)

    @server.on_client_disconnect
    async def _(client: viser.ClientHandle):
        # wait synchronously in this function for task to be finished.
        await client_closures[client.client_id].stop()
        del client_closures[client.client_id]

    while True:
        try:
            time.sleep(10.0)
        except KeyboardInterrupt:
            logger.info("Ctrl+C detected. Shutting down server...")
            break
    server.stop()


def main():
    parser = argparse.ArgumentParser(description="3D Visualizer")
    parser.add_argument("base_path", type=Path, help="Base path for the visualizer")
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=20530,
        help="Port number for the viser server.",
    )
    parser.add_argument(
        "--obj",
        type=str,
        default="/home/nas4_user/hoiyeongjin/repos/Project/SKT/mesh2pointcloud/data/stanley/stanley_more.ply",
    )
    args = parser.parse_args()

    run_viser(args.base_path, args.port, args.obj)


if __name__ == "__main__":
    main()

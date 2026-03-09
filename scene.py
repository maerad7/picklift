# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Franka Pick-and-Place Scene Viewer for Isaac Sim 5.1.

Isaac Sim에서 Pick-and-Place 씬을 직접 확인하기 위한 스크립트입니다.
"""

from __future__ import annotations

import argparse

parser = argparse.ArgumentParser(description="Franka Pick-and-Place Scene Viewer")
parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda", help="Simulation device")
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
args, _ = parser.parse_known_args()

# SimulationApp must be instantiated before any other omniverse imports
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

# Now we can import omniverse modules
import numpy as np
import omni.timeline
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.materials import PreviewSurfaceMaterial
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.storage.native import get_assets_root_path


def setup_scene(
    cube_position: np.ndarray = None,
    cube_size: np.ndarray = None,
    target_position: np.ndarray = None,
):
    """씬을 설정합니다: Franka 로봇, 큐브, 타겟 마커.
    
    Args:
        cube_position: 큐브 초기 위치 [x, y, z]
        cube_size: 큐브 크기 [w, h, d]
        target_position: 타겟 위치 [x, y, z]
    
    Returns:
        robot, cube, target_marker 튜플
    """
    if cube_position is None:
        cube_position = np.array([0.5, 0.0, 0.0258])
    if cube_size is None:
        cube_size = np.array([0.0515, 0.0515, 0.0515])
    if target_position is None:
        target_position = np.array([-0.3, -0.3, 0.001])

    # 새 USD 스테이지 생성 (sunlight 템플릿 사용)
    stage_utils.create_new_stage(template="sunlight")

    # Ground plane 추가
    stage_utils.add_reference_to_stage(
        usd_path=get_assets_root_path() + "/Isaac/Environments/Grid/default_environment.usd",
        path="/World/ground",
    )

    # Franka 로봇 생성
    robot_prim = stage_utils.add_reference_to_stage(
        usd_path="FrankaRobotics/FrankaPanda/franka.usd",
        path="/World/Robot",
        variants=[("Gripper", "AlternateFinger"), ("Mesh", "Performance")],
    )
    robot = Articulation("/World/Robot")
    robot.set_default_state(dof_positions=[0.012, -0.568, 0.0, -2.811, 0.0, 3.037, 0.741, 0.04, 0.04])

    # 파란색 큐브 생성
    blue_material = PreviewSurfaceMaterial("/Visual_materials/blue")
    blue_material.set_input_values("diffuseColor", [0.0, 0.0, 1.0])

    cube_shape = Cube(
        paths="/World/Cube",
        positions=cube_position,
        orientations=np.array([1, 0, 0, 0]),
        sizes=[1.0],
        scales=cube_size,
        reset_xform_op_properties=True,
    )
    GeomPrim(paths=cube_shape.paths, apply_collision_apis=True)
    cube = RigidPrim(paths=cube_shape.paths)
    cube_shape.apply_visual_materials(blue_material)

    # 녹색 타겟 마커 생성 (목표 위치 표시용)
    green_material = PreviewSurfaceMaterial("/Visual_materials/green")
    green_material.set_input_values("diffuseColor", [0.0, 1.0, 0.0])
    green_material.set_input_values("opacity", 0.5)

    target_marker_shape = Cube(
        paths="/World/TargetMarker",
        positions=target_position,
        orientations=np.array([1, 0, 0, 0]),
        sizes=[1.0],
        scales=np.array([0.08, 0.08, 0.002]),
        reset_xform_op_properties=True,
    )
    target_marker_shape.apply_visual_materials(green_material)

    print("[INFO] 씬 설정 완료!")
    print(f"  - 로봇: /World/Robot")
    print(f"  - 큐브 위치: {cube_position}")
    print(f"  - 타겟 위치: {target_position}")

    return robot, cube


def main():
    """메인 함수: 씬을 설정하고 시뮬레이션을 실행합니다."""
    print("=" * 60)
    print("Franka Pick-and-Place Scene Viewer")
    print("Isaac Sim 5.1")
    print("=" * 60)

    # 시뮬레이션 디바이스 설정
    SimulationManager.set_physics_sim_device(args.device)
    simulation_app.update()

    # 씬 설정
    robot, cube = setup_scene()

    # 시뮬레이션 시작
    omni.timeline.get_timeline_interface().play()
    simulation_app.update()

    print("\n[INFO] 시뮬레이션 실행 중...")
    print("[INFO] 카메라 조작:")
    print("  - 마우스 휠: 줌 인/아웃")
    print("  - Alt + 마우스 드래그: 카메라 회전")
    print("  - 마우스 중간 버튼 드래그: 카메라 이동")
    print("\n[INFO] 창을 닫으면 종료됩니다.")

    step_count = 0
    while simulation_app.is_running():
        if SimulationManager.is_simulating():
            step_count += 1
            
            # 매 500 스텝마다 상태 출력
            if step_count % 500 == 0:
                cube_pos = cube.get_world_poses()[0].numpy()
                print(f"[Step {step_count}] 큐브 위치: {cube_pos[0]}")

        simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()

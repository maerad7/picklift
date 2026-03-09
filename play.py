# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
RSL-RL로 학습된 정책 평가(Play) 스크립트

이 스크립트는 학습된 체크포인트를 로드하여 로봇 정책을 시각적으로 평가합니다.

사용법:
    python play.py --task FrankaLiftOnly-v0 --num_envs 16
    python play.py --task FrankaLiftOnly-v0 --checkpoint logs/rsl_rl/FrankaLiftOnly/2024-01-01_12-00-00/model_1000.pt
    python play.py --task FrankaLiftOnly-v0 --use_last_checkpoint

주요 기능:
    - RSL-RL 체크포인트 로드
    - 규칙 기반 그리퍼 제어 래퍼 (학습 시와 동일)
    - 실시간 시뮬레이션 옵션
    - 비디오 녹화 옵션
"""

# =============================================================================
# 1단계: SimulationApp 실행 전 설정
# =============================================================================

import argparse
import sys

from isaaclab.app import AppLauncher

# 로컬 CLI 인자 모듈
import cli_args  # isort: skip

# =============================================================================
# 명령줄 인자 파서 설정
# =============================================================================
parser = argparse.ArgumentParser(description="Play a trained RSL-RL checkpoint.")

# 비디오 녹화 옵션
parser.add_argument("--video", action="store_true", default=False, help="비디오 녹화")
parser.add_argument("--video_length", type=int, default=200, help="녹화 비디오 길이 (스텝)")

# 환경 설정
parser.add_argument("--num_envs", type=int, default=16, help="병렬 환경 수 (평가용이므로 적게)")
parser.add_argument("--task", type=str, default=None, help="작업(환경) 이름")

# 체크포인트 설정 (--checkpoint는 cli_args.add_rsl_rl_args에서 추가됨)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="마지막 체크포인트 사용 (기본: best)",
)

# 기타 옵션
parser.add_argument("--seed", type=int, default=None, help="랜덤 시드")
parser.add_argument("--real_time", action="store_true", default=False, help="실시간 실행")

# RSL-RL 및 AppLauncher CLI 인자 추가
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 비디오 녹화 시 카메라 활성화
if args_cli.video:
    args_cli.enable_cameras = True

# Hydra를 위해 sys.argv 정리
sys.argv = [sys.argv[0]] + hydra_args

# =============================================================================
# SimulationApp 실행
# =============================================================================
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# =============================================================================
# 2단계: SimulationApp 실행 후 모듈 import
# =============================================================================

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# 커스텀 환경 등록
import register as env_register
env_register.register_envs()

# =============================================================================
# 규칙 기반 그리퍼 제어 래퍼 (학습 시와 동일)
# =============================================================================

class RuleBasedGripperWrapper(gym.ActionWrapper):
    """규칙 기반 그리퍼 제어 래퍼
    
    학습 시 사용한 것과 동일한 래퍼를 평가 시에도 적용해야 합니다.
    """
    
    def __init__(
        self,
        env,
        xy_thresh=0.03,
        z_thresh=0.06,
        grasp_offset=0.02,
        lift_thresh=0.04,
        open_steps_after_reset=5,
    ):
        super().__init__(env)
        self.xy_thresh = xy_thresh
        self.z_thresh = z_thresh
        self.grasp_offset = grasp_offset
        self.lift_thresh = lift_thresh
        self.open_steps_after_reset = open_steps_after_reset

        self._left_finger_idx = None
        self._right_finger_idx = None
        self._close_latch = None

    def _lazy_init(self, env):
        robot = env.scene["robot"]
        num_envs = env.num_envs

        if self._left_finger_idx is None:
            self._left_finger_idx = robot.find_bodies("panda_leftfinger")[0][0]
        if self._right_finger_idx is None:
            self._right_finger_idx = robot.find_bodies("panda_rightfinger")[0][0]
        if self._close_latch is None or self._close_latch.shape[0] != num_envs:
            self._close_latch = torch.zeros(num_envs, dtype=torch.bool, device=env.device)

    def action(self, action):
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(
                action,
                device=self.env.unwrapped.device,
                dtype=torch.float32,
            )
            return_numpy = True
        else:
            action = action.clone()
            return_numpy = False

        env = self.env.unwrapped
        self._lazy_init(env)

        robot = env.scene["robot"]
        cube = env.scene["cube"]

        ep_len = env.episode_length_buf
        just_reset = ep_len == 0
        force_open = ep_len < self.open_steps_after_reset

        self._close_latch[just_reset] = False

        left_pos = robot.data.body_pos_w[:, self._left_finger_idx, :] - env.scene.env_origins
        right_pos = robot.data.body_pos_w[:, self._right_finger_idx, :] - env.scene.env_origins
        grip_center = 0.5 * (left_pos + right_pos)

        cube_pos = cube.data.root_pos_w[:, :3] - env.scene.env_origins

        dxy = torch.norm(grip_center[:, :2] - cube_pos[:, :2], dim=-1)
        dz = torch.abs(grip_center[:, 2] - (cube_pos[:, 2] + self.grasp_offset))

        aligned = (dxy < self.xy_thresh) & (dz < self.z_thresh)
        already_lifted = cube_pos[:, 2] > self.lift_thresh

        self._close_latch |= aligned
        close_cmd = self._close_latch | already_lifted

        close_cmd = close_cmd & (~force_open)

        action[:, -1] = torch.where(
            close_cmd,
            torch.full_like(dxy, -1.0),
            torch.full_like(dxy, 1.0),
        )

        return action.cpu().numpy() if return_numpy else action


# =============================================================================
# 메인 함수
# =============================================================================

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """RSL-RL 체크포인트 평가 메인 함수"""
    
    # -------------------------------------------------------------------------
    # 1. 설정 오버라이드
    # -------------------------------------------------------------------------
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        agent_cfg.seed = args_cli.seed

    # -------------------------------------------------------------------------
    # 2. 체크포인트 경로 찾기
    # -------------------------------------------------------------------------
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Looking for checkpoints in: {log_root_path}")

    if args_cli.checkpoint is not None:
        # 직접 지정된 체크포인트
        resume_path = args_cli.checkpoint
    else:
        # 자동 검색
        if args_cli.use_last_checkpoint:
            checkpoint_file = "model_*.pt"  # 마지막 모델
        else:
            checkpoint_file = "model_*.pt"  # best 모델 (동일)
        
        resume_path = get_checkpoint_path(log_root_path, ".*", checkpoint_file)
    
    print(f"[INFO] Loading checkpoint: {resume_path}")
    
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # -------------------------------------------------------------------------
    # 3. 환경 생성 및 래퍼 적용
    # -------------------------------------------------------------------------
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # MARL 환경인 경우 단일 에이전트로 변환
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    # 규칙 기반 그리퍼 래퍼 (학습 시와 동일하게 적용)
    env = RuleBasedGripperWrapper(env)

    # 비디오 녹화 래퍼
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # RSL-RL 벡터 환경 래퍼
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # -------------------------------------------------------------------------
    # 4. 에이전트 생성 및 체크포인트 로드
    # -------------------------------------------------------------------------
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    
    # 정책 가져오기
    policy = runner.get_inference_policy(device=agent_cfg.device)

    # -------------------------------------------------------------------------
    # 5. 평가 루프
    # -------------------------------------------------------------------------
    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    timestep = 0

    print("[INFO] Starting evaluation loop. Press Ctrl+C to exit.")
    
    while simulation_app.is_running():
        start_time = time.time()
        
        with torch.inference_mode():
            # 정책으로 행동 추론
            actions = policy(obs)
            
            # 환경 스텝 (RSL-RL wrapper: obs, rewards, dones, extras)
            obs, rewards, dones, extras = env.step(actions)
        
        # 비디오 녹화 종료 조건
        if args_cli.video:
            timestep += 1
            if timestep >= args_cli.video_length:
                print(f"[INFO] Video recording complete ({timestep} steps).")
                break

        # 실시간 실행을 위한 딜레이
        if args_cli.real_time:
            elapsed = time.time() - start_time
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # -------------------------------------------------------------------------
    # 6. 정리
    # -------------------------------------------------------------------------
    env.close()


# =============================================================================
# 스크립트 진입점
# =============================================================================

if __name__ == "__main__":
    main()
    simulation_app.close()

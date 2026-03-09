# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
RSL-RL을 사용한 강화학습 에이전트 학습 스크립트

이 스크립트는 Isaac Lab 환경에서 RSL-RL 라이브러리를 사용하여
PPO 알고리즘으로 로봇 제어 정책을 학습합니다.

사용법:
    python rslrl_train.py --task FrankaLiftOnly-v0 --num_envs 4096
    python rslrl_train.py --task FrankaLiftOnly-Easy-v0 --num_envs 4096 --max_iterations 1000

주요 기능:
    - RSL-RL PPO 학습
    - 규칙 기반 그리퍼 제어 래퍼 (RuleBasedGripperWrapper)
    - 학습 중 비디오 녹화
    - 분산 학습 지원 (multi-GPU)
    - TensorBoard 로깅
"""

# =============================================================================
# 1단계: SimulationApp 실행 전 설정
# Isaac Lab 모듈들은 SimulationApp 실행 후에만 import 가능
# =============================================================================

import argparse
import sys

from isaaclab.app import AppLauncher

# 로컬 CLI 인자 모듈
import cli_args  # isort: skip

# =============================================================================
# 명령줄 인자 파서 설정
# =============================================================================
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")

# 비디오 녹화 옵션
parser.add_argument("--video", action="store_true", default=False, help="학습 중 비디오 녹화")
parser.add_argument("--video_length", type=int, default=200, help="녹화 비디오 길이 (스텝)")
parser.add_argument("--video_interval", type=int, default=2000, help="비디오 녹화 간격 (스텝)")

# 환경 설정
parser.add_argument("--num_envs", type=int, default=None, help="병렬 환경 수")
parser.add_argument("--task", type=str, default=None, help="작업(환경) 이름")

# 에이전트 설정
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL 에이전트 설정 엔트리 포인트"
)
parser.add_argument("--seed", type=int, default=None, help="랜덤 시드")
parser.add_argument("--max_iterations", type=int, default=None, help="최대 학습 반복 횟수")

# 분산 학습
parser.add_argument(
    "--distributed", action="store_true", default=False, help="멀티 GPU 분산 학습"
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="IO 디스크립터 내보내기")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Ray 통합 시 자동 설정됨"
)

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
# SimulationApp 실행 (이후부터 Isaac Lab 모듈 import 가능)
# =============================================================================
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# =============================================================================
# RSL-RL 버전 확인
# =============================================================================

import importlib.metadata as metadata
import platform

from packaging import version

RSL_RL_VERSION = "3.0.1"  # 최소 요구 버전
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

# =============================================================================
# 2단계: SimulationApp 실행 후 나머지 모듈 import
# =============================================================================

import gymnasium as gym
import logging
import os
import torch
from datetime import datetime

# RSL-RL 러너 클래스
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

# Isaac Lab 환경 클래스들
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

# RSL-RL Isaac Lab 통합
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

# Isaac Lab 기본 작업들
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# 커스텀 환경 등록 (register.py)
import register as env_register
env_register.register_envs()

logger = logging.getLogger(__name__)

# =============================================================================
# PyTorch 성능 최적화 설정
# =============================================================================
torch.backends.cuda.matmul.allow_tf32 = True   # TF32 사용 (A100+에서 성능 향상)
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False     # 비결정적 알고리즘 허용 (성능)
torch.backends.cudnn.benchmark = False         # 벤치마크 비활성화

# =============================================================================
# 규칙 기반 그리퍼 제어 래퍼 (Rule-Based Gripper Wrapper)
# =============================================================================

import gymnasium as gym


class RuleBasedGripperWrapper(gym.ActionWrapper):
    """규칙 기반 그리퍼 제어 래퍼
    
    정책 네트워크가 팔 관절만 제어하도록 하고,
    그리퍼 열기/닫기는 휴리스틱 규칙으로 자동 제어합니다.
    
    이점:
        - 행동 공간 단순화 (8D → 7D + 자동 그리퍼)
        - 그리퍼 제어 실패로 인한 학습 어려움 제거
        - 학습 수렴 속도 향상
    
    그리퍼 제어 규칙:
        1. 리셋 직후 N 스텝: 무조건 열기
        2. 그리퍼와 큐브 정렬됨: 닫기 (래치)
        3. 큐브가 들어올려짐: 닫기 유지
        4. 그 외: 열기
    
    Args:
        env: 래핑할 환경
        xy_thresh: XY 정렬 임계값 (m)
        z_thresh: Z 정렬 임계값 (m)
        grasp_offset: 그리퍼 중심과 큐브 중심의 Z 오프셋 (m)
        lift_thresh: 들어올림 판단 높이 (m)
        open_steps_after_reset: 리셋 후 강제 열기 스텝 수
    """
    
    def __init__(
        self,
        env,
        xy_thresh=0.03,           # XY 정렬 임계값 3cm
        z_thresh=0.06,            # Z 정렬 임계값 6cm
        grasp_offset=0.02,        # 그리퍼-큐브 Z 오프셋 2cm
        lift_thresh=0.04,         # 들어올림 판단 높이 4cm
        open_steps_after_reset=5, # 리셋 후 5스텝 강제 열기
    ):
        super().__init__(env)
        self.xy_thresh = xy_thresh
        self.z_thresh = z_thresh
        self.grasp_offset = grasp_offset
        self.lift_thresh = lift_thresh
        self.open_steps_after_reset = open_steps_after_reset

        # 지연 초기화 (lazy init) - 환경 생성 후 첫 스텝에서 초기화
        self._left_finger_idx = None   # 왼쪽 핑거 바디 인덱스
        self._right_finger_idx = None  # 오른쪽 핑거 바디 인덱스
        self._close_latch = None       # 닫기 래치 상태 (한번 닫으면 유지)

    def _lazy_init(self, env):
        """환경 접근 후 필요한 인덱스들을 초기화"""
        robot = env.scene["robot"]
        num_envs = env.num_envs

        # 핑거 바디 인덱스 캐시
        if self._left_finger_idx is None:
            self._left_finger_idx = robot.find_bodies("panda_leftfinger")[0][0]
        if self._right_finger_idx is None:
            self._right_finger_idx = robot.find_bodies("panda_rightfinger")[0][0]
        
        # 닫기 래치 텐서 초기화 (환경 수 변경 시 재초기화)
        if self._close_latch is None or self._close_latch.shape[0] != num_envs:
            self._close_latch = torch.zeros(num_envs, dtype=torch.bool, device=env.device)

    def action(self, action):
        """정책 출력을 받아 그리퍼 명령을 규칙 기반으로 덮어씀
        
        Args:
            action: 정책 출력 [num_envs, action_dim]
                    마지막 차원 [-1]이 그리퍼 명령
        
        Returns:
            수정된 action (그리퍼 명령이 규칙 기반으로 대체됨)
        """
        # 텐서로 변환
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

        # 에피소드 길이 확인 (리셋 직후 판단용)
        ep_len = env.episode_length_buf
        just_reset = ep_len == 0                          # 방금 리셋됨
        force_open = ep_len < self.open_steps_after_reset  # 강제 열기 구간

        # 리셋된 환경의 래치 초기화
        self._close_latch[just_reset] = False

        # 그리퍼 중심 위치 계산 (두 핑거의 평균)
        left_pos = robot.data.body_pos_w[:, self._left_finger_idx, :] - env.scene.env_origins
        right_pos = robot.data.body_pos_w[:, self._right_finger_idx, :] - env.scene.env_origins
        grip_center = 0.5 * (left_pos + right_pos)  # [num_envs, 3]

        # 큐브 위치 (로컬 좌표)
        cube_pos = cube.data.root_pos_w[:, :3] - env.scene.env_origins  # [num_envs, 3]

        # 정렬 판단
        dxy = torch.norm(grip_center[:, :2] - cube_pos[:, :2], dim=-1)  # XY 거리
        dz = torch.abs(grip_center[:, 2] - (cube_pos[:, 2] + self.grasp_offset))  # Z 거리

        aligned = (dxy < self.xy_thresh) & (dz < self.z_thresh)  # 정렬됨
        already_lifted = cube_pos[:, 2] > self.lift_thresh       # 이미 들어올려짐

        # 닫기 래치: 한번 정렬되면 닫기 유지
        self._close_latch |= aligned
        close_cmd = self._close_latch | already_lifted

        # 리셋 직후 강제 열기
        close_cmd = close_cmd & (~force_open)

        # 그리퍼 명령 덮어쓰기: 닫기=-1, 열기=+1
        action[:, -1] = torch.where(
            close_cmd,
            torch.full_like(dxy, -1.0),  # 닫기
            torch.full_like(dxy, 1.0),   # 열기
        )

        return action.cpu().numpy() if return_numpy else action

# =============================================================================
# 메인 학습 함수
# =============================================================================

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """RSL-RL PPO 에이전트 학습 메인 함수
    
    Args:
        env_cfg: 환경 설정 (Hydra가 자동 주입)
        agent_cfg: 에이전트 설정 (Hydra가 자동 주입)
    
    학습 흐름:
        1. CLI 인자로 설정 오버라이드
        2. 환경 생성 (gym.make)
        3. 래퍼 적용 (그리퍼, 비디오, RSL-RL)
        4. 러너 생성 및 학습 시작
        5. 체크포인트 및 로그 저장
    """
    
    # -------------------------------------------------------------------------
    # 1. CLI 인자로 설정 오버라이드
    # -------------------------------------------------------------------------
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    
    # 환경 수 설정
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    
    # 최대 반복 횟수 설정
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # 시드 및 디바이스 설정
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # CPU + 분산 학습 조합 검증
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # -------------------------------------------------------------------------
    # 2. 멀티-GPU 분산 학습 설정
    # -------------------------------------------------------------------------
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # 각 GPU에 다른 시드 할당 (다양성 확보)
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # -------------------------------------------------------------------------
    # 3. 로그 디렉토리 설정
    # -------------------------------------------------------------------------
    # 로그 루트: logs/rsl_rl/{experiment_name}/
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # 실행 디렉토리: {timestamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # IO 디스크립터 내보내기 설정
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    env_cfg.log_dir = log_dir

    # -------------------------------------------------------------------------
    # 4. 환경 생성 및 래퍼 적용
    # -------------------------------------------------------------------------
    # Gymnasium 환경 생성
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # MARL 환경인 경우 단일 에이전트로 변환
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    # 규칙 기반 그리퍼 래퍼 적용 (정책은 팔만 제어)
    env = RuleBasedGripperWrapper(env)

    # 학습 재개 시 체크포인트 경로 저장
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # 비디오 녹화 래퍼 적용
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # RSL-RL 벡터 환경 래퍼 적용
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # -------------------------------------------------------------------------
    # 5. RSL-RL 러너 생성
    # -------------------------------------------------------------------------
    if agent_cfg.class_name == "OnPolicyRunner":
        # PPO 학습용 On-Policy 러너
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        # 지식 증류용 러너
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    # Git 상태 로깅
    runner.add_git_repo_to_log(__file__)

    # 체크포인트 로드 (재개 시)
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    # -------------------------------------------------------------------------
    # 6. 설정 저장 및 학습 시작
    # -------------------------------------------------------------------------
    # 환경 및 에이전트 설정을 YAML로 저장
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # 학습 시작!
    # init_at_random_ep_len=True: 에피소드 시작 위치를 랜덤화하여 bias 제거
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # -------------------------------------------------------------------------
    # 7. 정리
    # -------------------------------------------------------------------------
    env.close()


# =============================================================================
# 스크립트 진입점
# =============================================================================

if __name__ == "__main__":
    main()
    simulation_app.close()
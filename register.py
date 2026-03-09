# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Franka Lift-Only 강화학습 환경 등록 모듈

이 모듈은 Isaac Lab 프레임워크를 사용하여 Franka Panda 로봇의
큐브 들어올리기(Lift) 작업을 위한 강화학습 환경을 정의하고 등록합니다.

주요 구성요소:
    1. Scene Configuration: 로봇, 큐브 등 시뮬레이션 장면 구성
    2. MDP (Markov Decision Process) 설정:
       - Observations: 정책에 입력되는 관측 정보
       - Actions: 로봇 관절 제어 명령
       - Rewards: 보상 함수들
       - Terminations: 에피소드 종료 조건
       - Events: 리셋 시 랜덤화 설정
    3. Agent Configuration: RL-Games, RSL-RL 학습 설정

주의: 이 모듈은 반드시 SimulationApp이 실행된 후에 import 해야 합니다.
"""

from __future__ import annotations

import torch

# =============================================================================
# Isaac Lab 모듈 임포트
# =============================================================================

# 시뮬레이션 유틸리티 (USD 로드, 물리 속성 등)
import isaaclab.sim as sim_utils
# 액추에이터 설정 (PD 제어)
from isaaclab.actuators import ImplicitActuatorCfg
# 자산 설정 (로봇, 강체 객체)
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
# 환경 베이스 클래스
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
# MDP 매니저 설정 클래스들
from isaaclab.managers import ActionTermCfg as ActionTerm      # 행동 정의
from isaaclab.managers import EventTermCfg as EventTerm        # 이벤트(리셋 등)
from isaaclab.managers import ObservationGroupCfg as ObsGroup  # 관측 그룹
from isaaclab.managers import ObservationTermCfg as ObsTerm    # 관측 항목
from isaaclab.managers import RewardTermCfg as RewTerm         # 보상 항목
from isaaclab.managers import SceneEntityCfg                   # 씬 엔티티 참조
from isaaclab.managers import TerminationTermCfg as DoneTerm   # 종료 조건
# 씬 설정
from isaaclab.scene import InteractiveSceneCfg
# 설정 클래스 데코레이터
from isaaclab.utils import configclass

# Isaac Lab 기본 MDP 함수들 (joint_pos_rel, joint_vel_rel 등)
import isaaclab.envs.mdp as mdp
# Gymnasium 환경 등록
import gymnasium as gym

# RSL-RL PPO 학습 설정 클래스들
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,      # On-Policy 러너 설정
    RslRlPpoActorCriticCfg,      # Actor-Critic 네트워크 설정
    RslRlPpoAlgorithmCfg,        # PPO 알고리즘 하이퍼파라미터
)

###############################################################################
# 상수 정의 (Constants)
# 작업 성공 판단 및 보상 계산에 사용되는 임계값들
###############################################################################

TABLE_HEIGHT = 0.025          # 테이블 높이 (m) - 큐브 초기 위치 기준
LIFT_HEIGHT = 0.10            # 들어올리기 목표 높이 (m)
SUCCESS_LIFT_HEIGHT = 0.14    # 성공 판정 높이 (m) - 이 높이 이상이면 성공
SUCCESS_VEL_THRESH = 0.10     # 성공 시 허용 최대 속도 (m/s) - 안정성 확인용

# 규칙 기반 그리퍼 제어용 임계값 (RuleBasedGripperWrapper에서 사용)
ALIGN_XY_THRESH = 0.02        # XY 평면 정렬 허용 오차 (m)
ALIGN_Z_THRESH = 0.025        # Z축 정렬 허용 오차 (m)

###############################################################################
# 로봇 설정 (Robot Configuration)
# Franka Panda 7-DOF 매니퓰레이터 + 2-DOF 그리퍼
###############################################################################

FRANKA_PANDA_CFG = ArticulationCfg(
    # USD 모델 로드 설정
    spawn=sim_utils.UsdFileCfg(
        usd_path="FrankaRobotics/FrankaPanda/franka.usd",  # Nucleus 서버 경로
        activate_contact_sensors=False,  # 접촉 센서 비활성화
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,         # 중력 적용
            max_depenetration_velocity=5.0,  # 충돌 시 최대 분리 속도
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,        # 자기 충돌 감지 활성화
            solver_position_iteration_count=16,  # 위치 솔버 반복 횟수
            solver_velocity_iteration_count=4,   # 속도 솔버 반복 횟수
        ),
    ),
    # 초기 관절 자세 - 큐브를 잡기 좋은 "ready" 자세
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # 7-DOF 팔 관절 (라디안)
            "panda_joint1": 0.012,   # 베이스 회전
            "panda_joint2": -0.568,  # 어깨
            "panda_joint3": 0.0,     # 팔꿈치 회전
            "panda_joint4": -2.811,  # 팔꿈치 굽힘
            "panda_joint5": 0.0,     # 손목 회전
            "panda_joint6": 3.037,   # 손목 굽힘
            "panda_joint7": 0.741,   # 손목 회전
            # 그리퍼 (미터) - 0.04m = 열린 상태
            "panda_finger_joint1": 0.04,
            "panda_finger_joint2": 0.04,
        },
    ),
    # 액추에이터 설정 - Implicit(암시적) PD 제어
    actuators={
        # 어깨/팔 관절 (joint 1-4): 높은 토크 필요
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,  # 최대 토크 (Nm)
            stiffness=80.0,         # PD 제어 강성 (Kp)
            damping=4.0,            # PD 제어 감쇠 (Kd)
        ),
        # 전완/손목 관절 (joint 5-7): 낮은 토크
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            stiffness=80.0,
            damping=4.0,
        ),
        # 그리퍼 핑거: 높은 강성으로 정밀 제어
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit_sim=200.0,
            stiffness=2e3,   # 높은 강성
            damping=1e2,     # 높은 감쇠
        ),
    },
)

###############################################################################
# 씬 설정 (Scene Configuration)
# 시뮬레이션 환경에 배치할 모든 객체(자산)들을 정의
# {ENV_REGEX_NS}는 병렬 환경 인스턴스별 고유 경로로 자동 치환됨
###############################################################################

@configclass
class FrankaLiftOnlySceneCfg(InteractiveSceneCfg):
    """Franka 들어올리기 작업용 씬 설정 클래스
    
    포함 요소:
        - ground: 바닥 평면
        - dome_light: 환경 조명
        - robot: Franka Panda 로봇
        - cube: 집어 올릴 큐브 객체
    """

    # 바닥 평면 - 물리 시뮬레이션 충돌면
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # 환경 조명 (Dome Light) - 전체적인 밝기 제공
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
    )

    # Franka Panda 로봇 - 위에서 정의한 설정 사용
    # {ENV_REGEX_NS}는 각 병렬 환경의 고유 경로 (예: /World/envs/env_0)
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 큐브 객체 - 로봇이 집어서 들어올릴 대상
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),  # 5cm x 5cm x 5cm 정육면체
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,        # 중력 적용
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.04),  # 40g - 가벼운 물체
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # 물리 재질 - 높은 마찰력으로 잡기 쉽게
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2,   # 정지 마찰 계수
                dynamic_friction=1.0,  # 동적 마찰 계수
                restitution=0.0,       # 반발 계수 (튀지 않음)
            ),
            # 시각적 색상 - 파란색 큐브
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        # 초기 위치: 로봇 앞 50cm, 테이블 높이
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.50, 0.00, TABLE_HEIGHT)),
    )

###############################################################################
# 헬퍼 함수 (Helper Functions)
# 관측, 보상, 종료 조건 계산에 사용되는 유틸리티 함수들
# 모든 함수는 병렬 환경을 지원하며 [num_envs, ...] 형태의 텐서 반환
###############################################################################

def _ee_body_idx(env: ManagerBasedRLEnv) -> int:
    """End-effector(그리퍼 손) 바디의 인덱스를 반환"""
    robot = env.scene["robot"]
    return robot.find_bodies("panda_hand")[0][0]


def _finger_joint_indices(env: ManagerBasedRLEnv):
    """그리퍼 핑거 관절들의 인덱스 리스트 반환"""
    robot = env.scene["robot"]
    return robot.find_joints("panda_finger_joint.*")[0]


def ee_pos_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """End-effector의 월드 좌표 위치 [num_envs, 3]"""
    robot = env.scene["robot"]
    return robot.data.body_pos_w[:, _ee_body_idx(env), :]


def cube_pos_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """큐브의 월드 좌표 위치 [num_envs, 3]"""
    cube = env.scene["cube"]
    return cube.data.root_pos_w


def cube_lin_vel_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """큐브의 월드 좌표 선속도 [num_envs, 3]"""
    cube = env.scene["cube"]
    return cube.data.root_lin_vel_w


def ee_pos_local(env: ManagerBasedRLEnv) -> torch.Tensor:
    """End-effector의 로컬 좌표 위치 (환경 원점 기준) [num_envs, 3]"""
    return ee_pos_w(env) - env.scene.env_origins


def cube_pos_local(env: ManagerBasedRLEnv) -> torch.Tensor:
    """큐브의 로컬 좌표 위치 (환경 원점 기준) [num_envs, 3]"""
    return cube_pos_w(env) - env.scene.env_origins


def ee_to_cube_vec(env: ManagerBasedRLEnv) -> torch.Tensor:
    """End-effector에서 큐브로의 방향 벡터 [num_envs, 3]"""
    return cube_pos_local(env) - ee_pos_local(env)


def gripper_open_amount(env: ManagerBasedRLEnv) -> torch.Tensor:
    """그리퍼 열림 정도 (두 핑거 위치 합) [num_envs, 1]
    
    0.0 = 완전히 닫힘, ~0.08 = 완전히 열림
    """
    robot = env.scene["robot"]
    finger_ids = _finger_joint_indices(env)
    finger_pos = robot.data.joint_pos[:, finger_ids]
    return torch.sum(finger_pos, dim=-1, keepdim=True)


def ee_cube_xy_dist(env: ManagerBasedRLEnv) -> torch.Tensor:
    """End-effector와 큐브 사이의 XY 평면 거리 [num_envs]"""
    ee_pos = ee_pos_local(env)
    cube_pos = cube_pos_local(env)
    return torch.norm(ee_pos[:, :2] - cube_pos[:, :2], dim=-1)


def ee_cube_z_dist(env: ManagerBasedRLEnv) -> torch.Tensor:
    """End-effector와 큐브 사이의 Z축 거리 [num_envs]
    
    hand 중심이 cube 중심보다 약간 위(+0.02m)에 위치해야 함
    """
    ee_pos = ee_pos_local(env)
    cube_pos = cube_pos_local(env)
    return torch.abs(ee_pos[:, 2] - (cube_pos[:, 2] + 0.02))


def should_close_gripper(env: ManagerBasedRLEnv) -> torch.Tensor:
    """규칙 기반 그리퍼 닫기 조건 [num_envs] (bool)
    
    조건: 정렬됨 OR 이미 들어올려짐
    """
    dxy = ee_cube_xy_dist(env)
    dz = ee_cube_z_dist(env)

    aligned = (dxy < ALIGN_XY_THRESH) & (dz < ALIGN_Z_THRESH)
    already_lifted = cube_pos_local(env)[:, 2] > 0.04
    return aligned | already_lifted


def is_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    """작업 성공 판단 [num_envs] (bool)
    
    조건: 큐브 높이 > SUCCESS_LIFT_HEIGHT AND 큐브 속도 < SUCCESS_VEL_THRESH
    """
    cube_pos = cube_pos_local(env)
    cube_vel = torch.norm(cube_lin_vel_w(env), dim=-1)
    return (cube_pos[:, 2] > SUCCESS_LIFT_HEIGHT) & (cube_vel < SUCCESS_VEL_THRESH)

###############################################################################
# MDP - 보상 함수 (Reward Functions)
# 에이전트 행동에 대한 보상을 계산하는 함수들
# 보상 설계 원칙: tanh/exp로 정규화하여 0~1 범위로 스케일링
###############################################################################

def reach_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """접근 보상: End-effector가 큐브에 가까울수록 높은 보상
    
    Returns:
        [num_envs]: 0(멀리) ~ 1(가까이)
    """
    dist = torch.norm(ee_to_cube_vec(env), dim=-1)
    return 1.0 - torch.tanh(4.0 * dist)


def xy_align_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """XY 정렬 보상: 수평 방향으로 큐브 위에 위치할수록 높은 보상"""
    dxy = ee_cube_xy_dist(env)
    return 1.0 - torch.tanh(8.0 * dxy)


def z_align_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Z 정렬 보상: 큐브 위 적절한 높이에 위치할수록 높은 보상"""
    dz = ee_cube_z_dist(env)
    return 1.0 - torch.tanh(12.0 * dz)


def lift_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """들어올리기 보상: 큐브가 높이 올라갈수록 높은 보상
    
    지수 함수 사용으로 초기 들어올림에 큰 보상
    """
    cube_pos = cube_pos_local(env)
    lift_amount = torch.clamp(cube_pos[:, 2] - TABLE_HEIGHT, min=0.0)
    return 1.0 - torch.exp(-20.0 * lift_amount)


def success_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """성공 보너스: 작업 완료 시 1.0 반환"""
    return is_success(env).float()


def action_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """행동 변화 페널티: 급격한 행동 변화 억제 (스무딩 효과)"""
    return mdp.action_rate_l2(env)

###############################################################################
# MDP - 종료 조건 함수 (Termination Functions)
# 에피소드를 조기 종료시키는 조건들
# True 반환 시 해당 환경 인스턴스 reset
###############################################################################

def cube_dropped_fn(env: ManagerBasedRLEnv) -> torch.Tensor:
    """큐브 낙하 종료: 큐브가 바닥 아래로 떨어지면 종료"""
    cube_pos = cube_pos_local(env)
    return cube_pos[:, 2] < -0.05


def ee_out_of_bounds_fn(env: ManagerBasedRLEnv) -> torch.Tensor:
    """작업 공간 이탈 종료: End-effector가 허용 범위를 벗어나면 종료
    
    허용 범위 (로컬 좌표):
        x: -0.10m ~ 0.95m
        y: -0.75m ~ 0.75m
        z: 0.00m ~ 0.90m
    """
    ee_pos = ee_pos_local(env)

    x_out = (ee_pos[:, 0] < -0.10) | (ee_pos[:, 0] > 0.95)
    y_out = (ee_pos[:, 1] < -0.75) | (ee_pos[:, 1] > 0.75)
    z_out = (ee_pos[:, 2] < 0.00) | (ee_pos[:, 2] > 0.90)

    return x_out | y_out | z_out


def task_success_fn(env: ManagerBasedRLEnv) -> torch.Tensor:
    """작업 성공 종료: 성공 조건 만족 시 에피소드 종료"""
    return is_success(env)

###############################################################################
# MDP 설정 - 관측 (Observations)
# 정책 네트워크에 입력되는 상태 정보
###############################################################################

@configclass
class ObservationsCfg:
    """관측 설정 클래스"""

    @configclass
    class PolicyCfg(ObsGroup):
        """정책 네트워크용 관측 그룹
        
        관측 구성:
            - joint_pos: 관절 위치 (기본값 대비 상대값) [9]
            - joint_vel: 관절 속도 (기본값 대비 상대값) [9]
            - ee_pos: End-effector 로컬 위치 [3]
            - cube_pos: 큐브 로컬 위치 [3]
            - ee_to_cube: EE→큐브 방향 벡터 [3]
            - cube_vel: 큐브 선속도 [3]
            - gripper_open: 그리퍼 열림 정도 [1]
            - actions: 이전 행동 [8]
        """

        # 로봇 고유 수용 감각 (proprioception)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)  # 관절 위치
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)  # 관절 속도

        # 공간 정보
        ee_pos = ObsTerm(func=ee_pos_local)         # End-effector 위치
        cube_pos = ObsTerm(func=cube_pos_local)     # 큐브 위치
        ee_to_cube = ObsTerm(func=ee_to_cube_vec)   # EE→큐브 벡터

        # 객체 상태
        cube_vel = ObsTerm(func=cube_lin_vel_w)     # 큐브 속도
        gripper_open = ObsTerm(func=gripper_open_amount)  # 그리퍼 상태

        # 이전 행동 (temporal context)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False   # 노이즈 비활성화
            self.concatenate_terms = True    # 모든 관측을 하나의 벡터로 연결

    policy: PolicyCfg = PolicyCfg()

###############################################################################
# MDP 설정 - 행동 (Actions)
# 정책 네트워크의 출력을 로봇 제어 명령으로 변환
###############################################################################

@configclass
class ActionsCfg:
    """행동 설정 클래스
    
    행동 공간:
        - arm_action: 7-DOF 팔 관절 위치 변화량 [7]
        - gripper_action: 그리퍼 열기/닫기 (이진) [1]
    
    Note: 학습 시 RuleBasedGripperWrapper가 그리퍼 명령을 덮어씀
    """

    # 팔 관절 위치 제어 (델타 위치)
    arm_action: ActionTerm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],  # 7개 팔 관절
        scale=0.15,                      # 행동 스케일링
        use_default_offset=True,         # 초기 자세 기준 오프셋
    )

    # 그리퍼 이진 제어 (열기/닫기)
    gripper_action: ActionTerm = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger_joint.*"],
        open_command_expr={"panda_finger_joint.*": 0.04},   # 열림 위치
        close_command_expr={"panda_finger_joint.*": 0.0},   # 닫힘 위치
    )

###############################################################################
# MDP 설정 - 보상 (Rewards)
# 각 보상 항목의 가중치(weight)로 상대적 중요도 조절
###############################################################################

@configclass
class RewardsCfg:
    """보상 설정 클래스
    
    보상 설계: 접근 → 정렬 → 들어올리기 → 성공
    """

    # === 긍정적 보상 ===
    reaching = RewTerm(func=reach_reward, weight=1.5)     # 큐브 접근
    xy_align = RewTerm(func=xy_align_reward, weight=2.0)  # XY 정렬
    z_align = RewTerm(func=z_align_reward, weight=2.0)    # Z 정렬

    lifting = RewTerm(func=lift_reward, weight=12.0)      # 들어올리기 (핵심!)
    success = RewTerm(func=success_bonus, weight=30.0)    # 최종 성공

    # === 부정적 보상 ===
    action_rate = RewTerm(func=action_penalty, weight=-1e-4)  # 급격한 행동 억제

###############################################################################
# MDP 설정 - 종료 조건 (Terminations)
###############################################################################

@configclass
class TerminationsCfg:
    """종료 조건 설정 클래스"""

    # 시간 초과 (truncation)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # 실패 조건 (termination)
    cube_dropped = DoneTerm(func=cube_dropped_fn)
    ee_out_of_bounds = DoneTerm(func=ee_out_of_bounds_fn)
    
    # 성공 조건 (termination)
    task_success = DoneTerm(func=task_success_fn)

###############################################################################
# MDP 설정 - 이벤트 (Events)
# Domain Randomization으로 일반화 성능 향상
###############################################################################

@configclass
class EventsCfg:
    """이벤트/랜덤화 설정 클래스"""

    # 로봇 초기 자세 랜덤화
    reset_robot = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.05, 0.05),  # 관절 위치 ±0.05 rad 랜덤
            "velocity_range": (0.0, 0.0),     # 초기 속도 0
        },
    )

    # 큐브 초기 위치 랜덤화
    reset_cube = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "pose_range": {
                "x": (-0.02, 0.02),  # X축 ±2cm 랜덤
                "y": (-0.03, 0.03),  # Y축 ±3cm 랜덤
                "z": (0.0, 0.0),     # Z축 고정 (테이블 위)
            },
            "velocity_range": {},    # 초기 속도 0
        },
    )

###############################################################################
# 환경 설정 (Environment Configuration)
# 전체 환경을 구성하는 최상위 설정 클래스
###############################################################################

@configclass
class FrankaLiftOnlyEnvCfg(ManagerBasedRLEnvCfg):
    """Franka 들어올리기 환경 설정 클래스
    
    주요 파라미터:
        - num_envs: 병렬 환경 수 (GPU 메모리에 따라 조절)
        - episode_length_s: 에피소드 최대 시간 (초)
        - decimation: 물리 스텝 / 제어 스텝 비율
        
    제어 주파수 = (1/dt) / decimation = 120Hz / 2 = 60Hz
    """

    # 씬 설정 (로봇, 큐브 등)
    scene: FrankaLiftOnlySceneCfg = FrankaLiftOnlySceneCfg(num_envs=4096, env_spacing=2.5)
    
    # MDP 구성요소
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    episode_length_s = 8.0    # 에피소드 최대 8초
    decimation = 2            # 제어 주파수 = 60Hz

    def __post_init__(self):
        self.sim.dt = 1.0 / 120.0        # 물리 스텝 = 120Hz
        self.sim.render_interval = self.decimation


@configclass
class FrankaLiftOnlyEasyEnvCfg(FrankaLiftOnlyEnvCfg):
    """쉬운 버전 - 큐브 위치 고정 (랜덤화 없음)
    
    학습 초기 또는 디버깅 시 사용
    """

    def __post_init__(self):
        super().__post_init__()
        # 큐브 초기 위치 랜덤화 비활성화
        self.events.reset_cube.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
        }

###############################################################################
# RL-Games 에이전트 설정 (Agent Configuration)
# RL-Games 라이브러리용 PPO 하이퍼파라미터
###############################################################################

RL_GAMES_PPO_CFG = {
    "params": {
        "seed": 42,                                   # 재현성을 위한 시드
        "algo": {"name": "a2c_continuous"},           # 알고리즘 종류
        "model": {"name": "continuous_a2c_logstd"},   # 모델 타입
        
        # 신경망 구조
        "network": {
            "name": "actor_critic",
            "separate": False,     # Actor-Critic 가중치 공유
            "space": {
                "continuous": {
                    "mu_activation": "None",
                    "sigma_activation": "None",
                    "mu_init": {"name": "default"},
                    "sigma_init": {"name": "const_initializer", "val": 0.0},
                    "fixed_sigma": True,
                }
            },
            "mlp": {
                "units": [256, 256, 128],    # 히든 레이어 크기
                "activation": "elu",          # 활성화 함수
                "d2rl": False,
                "initializer": {"name": "default"},
                "regularizer": {"name": "None"},
            },
        },
        "load_checkpoint": False,
        "load_path": "",
        
        # 환경 설정
        "env": {
            "clip_observations": 5.0,   # 관측값 클리핑
            "clip_actions": 1.0,        # 행동값 클리핑
        },
        
        # PPO 하이퍼파라미터
        "config": {
            "name": "FrankaLiftOnly",
            "full_experiment_name": "FrankaLiftOnly",
            "env_name": "rlgpu",
            "device": "cuda:0",
            "device_name": "cuda:0",
            "multi_gpu": False,
            "ppo": True,
            "mixed_precision": False,
            "normalize_input": True,      # 입력 정규화
            "normalize_value": True,      # 가치 정규화
            "value_bootstrap": True,
            "num_actors": -1,
            "reward_shaper": {"scale_value": 1.0},
            "normalize_advantage": True,
            "gamma": 0.99,                # 할인율
            "tau": 0.95,                  # GAE lambda
            "learning_rate": 3e-4,        # 학습률
            "lr_schedule": "adaptive",    # KL 기반 적응적 학습률
            "schedule_type": "legacy",
            "kl_threshold": 0.008,
            "score_to_win": 100000,
            "max_epochs": 1500,           # 최대 학습 에폭
            "save_best_after": 100,
            "save_frequency": 100,        # 체크포인트 저장 주기
            "print_stats": True,
            "grad_norm": 1.0,             # 그래디언트 클리핑
            "entropy_coef": 0.004,        # 엔트로피 보너스
            "truncate_grads": True,
            "e_clip": 0.2,                # PPO 클리핑
            "horizon_length": 128,        # 환경당 수집 스텝
            "minibatch_size": 1024,       # 미니배치 크기
            "mini_epochs": 5,             # 에폭당 업데이트 횟수
            "critic_coef": 2,
            "clip_value": True,
            "seq_length": 4,
            "bounds_loss_coef": 0.0001,
        },
    }
}

###############################################################################
# RSL-RL 에이전트 설정 (Agent Configuration)
# RSL-RL 라이브러리용 PPO 하이퍼파라미터
# ETH Zurich의 Robotic Systems Lab에서 개발
###############################################################################

@configclass
class FrankaLiftOnlyPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO 설정 클래스
    
    주요 설정:
        - num_steps_per_env: 환경당 수집 스텝 (horizon)
        - max_iterations: 총 학습 반복 횟수
        - policy: Actor-Critic 네트워크 구조
        - algorithm: PPO 알고리즘 하이퍼파라미터
    """

    # 기본 설정
    seed: int = 42                           # 재현성을 위한 시드
    device: str = "cuda:0"                   # GPU 디바이스
    num_steps_per_env: int = 64              # 환경당 수집 스텝 (horizon_length)
    max_iterations: int = 1500               # 총 학습 반복 횟수
    save_interval: int = 100                 # 체크포인트 저장 주기
    
    # 실험 설정
    experiment_name: str = "FrankaLiftOnly"  # TensorBoard 실험 이름
    run_name: str = ""
    resume: bool = False
    load_run: str = ""
    load_checkpoint: str = ""
    logger: str = "tensorboard"
    neptune_project: str = ""
    wandb_project: str = ""

    # Actor-Critic 네트워크 구조
    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,                  # 초기 행동 노이즈 (탐색용)
        actor_hidden_dims=[256, 256, 128],   # Actor 히든 레이어
        critic_hidden_dims=[256, 256, 128],  # Critic 히든 레이어
        activation="elu",                    # 활성화 함수
    )

    # PPO 알고리즘 하이퍼파라미터
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,                 # Critic 손실 가중치
        use_clipped_value_loss=True,
        clip_param=0.2,                      # PPO 클리핑 파라미터 (ε)
        entropy_coef=0.005,                  # 엔트로피 보너스
        num_learning_epochs=5,               # 수집 데이터로 몇 번 업데이트
        num_mini_batches=4,                  # 미니배치 분할 수
        learning_rate=3e-4,                  # 학습률
        schedule="adaptive",                 # KL 기반 적응적 학습률
        gamma=0.99,                          # 할인율
        lam=0.95,                            # GAE lambda
        desired_kl=0.01,                     # 목표 KL divergence
        max_grad_norm=1.0,                   # 그래디언트 클리핑
    )

###############################################################################
# 환경 등록 (Environment Registration)
# Gymnasium에 환경을 등록하여 gym.make()로 생성 가능하게 함
###############################################################################

def register_envs():
    """Gymnasium에 환경 등록
    
    등록된 환경:
        - FrankaLiftOnly-v0: 기본 버전 (큐브 위치 랜덤화)
        - FrankaLiftOnly-Easy-v0: 쉬운 버전 (큐브 위치 고정)
    
    사용법:
        import register
        register.register_envs()
        env = gym.make("FrankaLiftOnly-v0")
    """

    # 기본 버전 등록
    gym.register(
        id="FrankaLiftOnly-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": FrankaLiftOnlyEnvCfg,
            "rl_games_cfg_entry_point": RL_GAMES_PPO_CFG,
            "rsl_rl_cfg_entry_point": "register:FrankaLiftOnlyPPORunnerCfg",
        },
    )

    # 쉬운 버전 등록 (디버깅/초기 학습용)
    gym.register(
        id="FrankaLiftOnly-Easy-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": FrankaLiftOnlyEasyEnvCfg,
            "rl_games_cfg_entry_point": RL_GAMES_PPO_CFG,
            "rsl_rl_cfg_entry_point": "register:FrankaLiftOnlyPPORunnerCfg",
        },
    )

    print("[INFO] Registered FrankaLiftOnly environments:")
    print("  - FrankaLiftOnly-v0        (랜덤 큐브 위치)")
    print("  - FrankaLiftOnly-Easy-v0   (고정 큐브 위치)")
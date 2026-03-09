# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RSL-RL PPO Configuration for Franka Pick-and-Place."""

from isaaclab_rl.rsl_rl import PPORunnerCfg, OnPolicyPpoActorCriticCfg, OnPolicyPpoAlgorithmCfg


FrankaPickPlacePPORunnerCfg = PPORunnerCfg(
    seed=42,
    device="cuda:0",
    num_steps_per_env=64,
    max_iterations=1000,
    empirical_normalization=False,
    policy=OnPolicyPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    ),
    algorithm=OnPolicyPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ),
    save_interval=100,
    experiment_name="FrankaPickPlace",
    run_name="",
    logger="tensorboard",
    neptune_project="",
    wandb_project="",
    resume=False,
    load_run="",
    load_checkpoint="",
)

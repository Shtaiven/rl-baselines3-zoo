#!/usr/bin/env python3
# Train PandaReach-v2 and PandaReachJoints-v2 with PPO and other algorithms.

from pathlib import Path
import argparse
import difflib
import importlib
import os
from typing import Any, Optional, Dict
import uuid

import gym
import numpy as np
import seaborn
import torch as th
from stable_baselines3.common.utils import set_random_seed
import colorama
from colorama import Fore, Style

# Register custom envs
import utils.import_envs  # noqa: F401 pytype: disable=import-error
from utils.exp_manager import ExperimentManager
from utils.utils import ALGOS, StoreDict

colorama.init()
seaborn.set()


def train(
    args: argparse.Namespace,
    algo: str = "ppo",
    env_id: str = "PandaReach-v2",
    log_folder: str = "logs",
    tensorboard_log: str = "",
    n_timesteps: int = -1,
    eval_freq: int = 10000,
    n_eval_episodes: int = 5,
    save_freq: int = -1,
    hyperparams: Optional[Dict[str, Any]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    trained_agent: str = "",
    optimize_hyperparameters: bool = False,
    storage: Optional[str] = None,
    study_name: Optional[str] = None,
    n_trials: int = 10,
    n_jobs: int = 1,
    sampler: str = "tpe",
    pruner: str = "median",
    optimization_log_path: Optional[str] = None,
    n_startup_trials: int = 10,
    n_evaluations: int = 20,
    truncate_last_trajectory: bool = True,
    uuid_str: str = "",
    seed: int = 0,
    log_interval: int = -1,
    save_replay_buffer: bool = False,
    verbose: int = 1,
    vec_env_type: str = "dummy",
    n_eval_envs: int = 1,
    no_optim_plots: bool = False,
):
    exp_manager = ExperimentManager(
        args,
        algo,
        env_id,
        log_folder,
        tensorboard_log,
        n_timesteps,
        eval_freq,
        n_eval_episodes,
        save_freq,
        hyperparams,
        env_kwargs,
        trained_agent,
        optimize_hyperparameters,
        storage,
        study_name,
        n_trials,
        n_jobs,
        sampler,
        pruner,
        optimization_log_path,
        n_startup_trials=n_startup_trials,
        n_evaluations=n_evaluations,
        truncate_last_trajectory=truncate_last_trajectory,
        uuid_str=uuid_str,
        seed=seed,
        log_interval=log_interval,
        save_replay_buffer=save_replay_buffer,
        verbose=verbose,
        vec_env_type=vec_env_type,
        n_eval_envs=n_eval_envs,
        no_optim_plots=no_optim_plots,
    )

    # Prepare experiment and launch hyperparameter optimization if needed
    model = exp_manager.setup_experiment()

    # Normal training
    if model is not None:
        exp_manager.learn(model)
        exp_manager.save_trained_model(model)
    else:
        exp_manager.hyperparameters_optimization()

    return exp_manager


def run(args):
    separate_obstacles = args.separate_obstacles
    init_pose_type = "neutral" if args.no_random_init else "random"
    source_dir = Path(__file__).parent
    envs = ["PandaReach-v2", "PandaReachJoints-v2"]
    algo = "ppo"
    obstacles = [None, "inline", "bin", "L"]
    reward_types = ["pid", "dense", "sparse"]
    P, I, D = [5.0, 5.0, 1.0]
    trained_agents = {}

    # Train agents for all combinations of reward type and environment
    # Training continues for each obstacle by continuing training on the policy from the previous obstacle within each (env, reward_type) pair
    for reward_type in reward_types:
        for env in envs:
            prev_trained_agent = ""
            for obstacle in obstacles:
                uuid_str = f"_{uuid.uuid4()}"
                print(
                    Style.BRIGHT
                    + Fore.MAGENTA
                    + f"Training {env} with {algo}, obstacle: {obstacle}, reward: {reward_type}"
                    + Style.RESET_ALL
                )
                tensorboard_log = source_dir / "autoruns" / "train-policy" / env / reward_type
                exp_manager = train(
                    args,
                    algo=algo,
                    env_id=env,
                    tensorboard_log=tensorboard_log,
                    save_freq=100000,
                    env_kwargs={
                        "reward_type": reward_type,
                        "obstacle_type": obstacle,
                        "reward_weights": [P, I, D],
                        "init_pose_type": init_pose_type,
                    },
                    uuid_str=uuid_str,
                    trained_agent=prev_trained_agent,
                    verbose=0,
                )
                if not separate_obstacles:
                    prev_trained_agent = str(Path(exp_manager.save_path) / "best_model.zip")
                else:
                    trained_agents[f"{env}_{reward_type}_{obstacle}"] = str(Path(exp_manager.save_path) / "best_model.zip")
            if not separate_obstacles:
                trained_agents[f"{env}_{reward_type}"] = prev_trained_agent

    # Print the locations of the trained agents
    print(f"\n\n{Style.BRIGHT}{Fore.CYAN}Trained agent locations:{Style.RESET_ALL}")
    for key, value in trained_agents.items():
        print(f"{Style.BRIGHT}{key}:{Style.RESET_ALL} {value}")

    return trained_agents


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        help="RL Algorithm",
        default="ppo",
        type=str,
        required=False,
        choices=list(ALGOS.keys()),
    )
    parser.add_argument(
        "--env", type=str, default="PandaReach-v2", help="environment ID"
    )
    parser.add_argument(
        "-tb", "--tensorboard-log", help="Tensorboard log dir", default="", type=str
    )
    parser.add_argument(
        "-i",
        "--trained-agent",
        help="Path to a pretrained agent to continue training",
        default="",
        type=str,
    )
    parser.add_argument(
        "--truncate-last-trajectory",
        help="When using HER with online sampling the last trajectory "
        "in the replay buffer will be truncated after reloading the replay buffer.",
        default=True,
        type=bool,
    )
    parser.add_argument(
        "-n",
        "--n-timesteps",
        help="Overwrite the number of timesteps",
        default=-1,
        type=int,
    )
    parser.add_argument(
        "--num-threads",
        help="Number of threads for PyTorch (-1 to use default)",
        default=-1,
        type=int,
    )
    parser.add_argument(
        "--log-interval",
        help="Override log interval (default: -1, no change)",
        default=-1,
        type=int,
    )
    parser.add_argument(
        "--eval-freq",
        help="Evaluate the agent every n steps (if negative, no evaluation). "
        "During hyperparameter optimization n-evaluations is used instead",
        default=10000,
        type=int,
    )
    parser.add_argument(
        "--optimization-log-path",
        help="Path to save the evaluation log and optimal policy for each hyperparameter tried during optimization. "
        "Disabled if no argument is passed.",
        type=str,
    )
    parser.add_argument(
        "--eval-episodes",
        help="Number of episodes to use for evaluation",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--n-eval-envs",
        help="Number of environments for evaluation",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--save-freq",
        help="Save the model every n steps (if negative, no checkpoint)",
        default=-1,
        type=int,
    )
    parser.add_argument(
        "--save-replay-buffer",
        help="Save the replay buffer too (when applicable)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-f", "--log-folder", help="Log folder", type=str, default="logs"
    )
    parser.add_argument("--seed", help="Random generator seed", type=int, default=-1)
    parser.add_argument(
        "--vec-env",
        help="VecEnv type",
        type=str,
        default="dummy",
        choices=["dummy", "subproc"],
    )
    parser.add_argument(
        "--n-trials",
        help="Number of trials for optimizing hyperparameters. "
        "This applies to each optimization runner, not the entire optimization process.",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-optimize",
        "--optimize-hyperparameters",
        action="store_true",
        default=False,
        help="Run hyperparameters search",
    )
    parser.add_argument(
        "--no-optim-plots",
        action="store_true",
        default=False,
        help="Disable hyperparameter optimization plots",
    )
    parser.add_argument(
        "--n-jobs",
        help="Number of parallel jobs when optimizing hyperparameters",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--sampler",
        help="Sampler to use when optimizing hyperparameters",
        type=str,
        default="tpe",
        choices=["random", "tpe", "skopt"],
    )
    parser.add_argument(
        "--pruner",
        help="Pruner to use when optimizing hyperparameters",
        type=str,
        default="median",
        choices=["halving", "median", "none"],
    )
    parser.add_argument(
        "--n-startup-trials",
        help="Number of trials before using optuna sampler",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--n-evaluations",
        help="Training policies are evaluated every n-timesteps // n-evaluations steps when doing hyperparameter optimization",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--storage",
        help="Database storage path if distributed optimization should be used",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--study-name",
        help="Study name for distributed optimization",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int
    )
    parser.add_argument(
        "--gym-packages",
        type=str,
        nargs="+",
        default=[],
        help="Additional external Gym environment package modules to import (e.g. gym_minigrid)",
    )
    parser.add_argument(
        "--env-kwargs",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Optional keyword argument to pass to the env constructor",
    )
    parser.add_argument(
        "-params",
        "--hyperparams",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10)",
    )
    parser.add_argument(
        "-uuid",
        "--uuid",
        action="store_true",
        default=False,
        help="Ensure that the run has a unique ID",
    )
    parser.add_argument(
        "-s",
        "--separate-obstacles",
        action="store_true",
        default=False,
        help="Train a separate policy for each obstacle instead concurrently training a policy with each obstacle",
    )
    parser.add_argument(
        "--no-random-init",
        action="store_true",
        default=False,
        help="Don't randomize the starting position of the agent",
    )
    args = parser.parse_args()

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    registered_envs = set(
        gym.envs.registry.env_specs.keys()
    )  # pytype: disable=module-attr

    # If the environment is not found, suggest the closest match
    if env_id not in registered_envs:
        try:
            closest_match = difflib.get_close_matches(env_id, registered_envs, n=1)[0]
        except IndexError:
            closest_match = "'no close match found...'"
        raise ValueError(
            f"{env_id} not found in gym registry, you maybe meant {closest_match}?"
        )

    # Unique id to ensure there is no race condition for the folder creation
    uuid_str = f"_{uuid.uuid4()}" if args.uuid else ""
    if args.seed < 0:
        # Seed but with a random one
        args.seed = np.random.randint(2**32 - 1, dtype="int64").item()

    set_random_seed(args.seed)

    # Setting num threads to 1 makes things run faster on cpu
    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    if args.trained_agent != "":
        assert args.trained_agent.endswith(".zip") and os.path.isfile(
            args.trained_agent
        ), "The trained_agent must be a valid path to a .zip file"

    print("=" * 10, env_id, "=" * 10)
    print(f"Seed: {args.seed}")
    run(args)

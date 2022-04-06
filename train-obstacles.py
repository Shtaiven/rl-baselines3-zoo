#!/usr/bin/env python3
# Train PandaReach-v2 and PandaReachJoints-v2 with PPO and other algorithms.

import subprocess
from pathlib import Path
import colorama
from colorama import Fore, Style

colorama.init()


def run():
    source_dir = Path(__file__).parent
    # envs = ["PandaReach-v2", "PandaReachJoints-v2"]
    envs = ["PandaReachJoints-v2"]
    algos = ["ppo"]
    # obstacles = ["bin", "L", "inline", None]
    obstacles = ["bin"]
    reward_types = ["pid", "dense", "sparse"]
    for algo in algos:
        for env in envs:
            for obstacle in obstacles:
                for reward_type in reward_types:
                    print(Style.BRIGHT + Fore.MAGENTA + f"Training {env} with {algo}, obstacle: {obstacle}, reward: {reward_type}" + Style.RESET_ALL)
                    subprocess.call(
                        f"python3 {source_dir}/train.py --env {env} --algo {algo} --tensorboard-log {source_dir}/autoruns/obstacles/{obstacle}/{reward_type} --save-freq 100000 --env-kwargs reward_type:{repr(reward_type)} obstacle_type:{repr(obstacle)}".split()
                    )


if __name__ == "__main__":
    run()

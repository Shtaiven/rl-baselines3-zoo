#!/usr/bin/env python3
# Train PandaReach-v2 and PandaReachJoints-v2 with PPO and other algorithms.

import subprocess
from pathlib import Path
import colorama
from colorama import Fore, Style

colorama.init()


def run():
    source_dir = Path(__file__).parent
    envs = ["PandaReach-v2", "PandaReachJoints-v2"]
    algos = ["ppo"]

    for algo in algos:
        for env in envs:
            print(Style.BRIGHT + Fore.MAGENTA + f"Training {env} with {algo}" + Style.RESET_ALL)
            subprocess.call(
                f"python3 {source_dir}/train.py --env {env} --algo {algo} --tensorboard-log {source_dir}/autoruns/ --save-freq 100000".split()
            )


if __name__ == "__main__":
    run()

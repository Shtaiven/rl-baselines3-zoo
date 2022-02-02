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
    reward_type = "dense"
    P_values = (1.0, 2.0, 5.0)
    I_values = (1.0, 2.0, 5.0)
    D_values = (1.0, 2.0, 5.0)
    reward_weights = [(p,i,d) for p in P_values for i in I_values for d in D_values]

    for algo in algos:
        for env in envs:
            for weights in reward_weights:
                print(Style.BRIGHT + Fore.MAGENTA + f"Training {env} with {algo}, weights: {weights}" + Style.RESET_ALL)
                P,I,D = weights
                subprocess.call(
                    f"python3 {source_dir}/train.py --env {env} --algo {algo} --tensorboard-log {source_dir}/autoruns/pid/{P}_{I}_{D} --save-freq 100000 --env-kwargs reward_type:\"{reward_type}\" reward_weights:({P},{I},{D})".split()
                )


if __name__ == "__main__":
    run()

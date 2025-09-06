import argparse
import logging
import os
import time
from dataclasses import dataclass

import gym
import numpy as np
import torch

from env.london import TowersOfLondon
from Muzero import Muzero
from utils import setup_logger


@dataclass
class TrainingConfig:
    """Configuration options for training."""

    env: str = "London"
    training_loops: int = 5000
    min_replay_size: int = 5000
    dirichlet_alpha: float = 0.25
    n_ep_x_loop: int = 1
    n_update_x_loop: int = 1
    unroll_n_steps: int = 5
    TD_return: bool = True
    n_TD_step: int = 10
    buffer_size: int = 50000
    priority_replay: bool = True
    batch_s: int = 256
    discount: float = 0.8
    n_mcts_simulations: int = 25
    lr: float = 0.002 #0.02 
    seed: int = 1
    profile: bool = False


def get_env(env_name):
    if env_name == "London":
        N = 3
        max_steps = 200
        env = TowersOfLondon(N=N, max_steps=max_steps)
        s_space_size = env.oneH_s_size
        n_action = 6  # n. of action available in each state for Tower of London (including illegal ones)
    else:  # Use for gym env with discrete 1d action space
        env = gym.make(env_name)
        assert isinstance(
            env.action_space, gym.spaces.discrete.Discrete
        ), "Must be discrete action space"
        s_space_size = env.observation_space.shape[0]
        n_action = env.action_space.n
        max_steps = env.spec.max_episode_steps
        N = None
    return env, s_space_size, n_action, max_steps, N


## ======= Set seeds for debugging =======
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    setup_logger(seed)


def save_results(env_name, command_line, stats, muzero, timestamp):
    file_indx = 1
    # Create directory to store results
    file_dir = os.path.dirname(os.path.abspath(__file__))
    file_dir = os.path.join(file_dir, "stats", env_name, str(timestamp))

    # Create directory if it did't exist before
    os.makedirs(file_dir, exist_ok=True)

    # Store command line
    with open(os.path.join(file_dir, "commands.txt"), "w") as f:
        f.write(command_line)

    dict_keys = ["state_action_history", "avg_decisions_per_state"]

    # Save all stats
    for key, value in stats.items():
        if key in dict_keys:
            # Save dictionaries directly without converting to tensor
            torch.save(value, os.path.join(file_dir, f"{key}-{timestamp}.pt"))
        else:
            # Convert other values to tensors
            torch.save(
                torch.tensor(value), os.path.join(file_dir, f"{key}-{timestamp}.pt")
            )

    model_dir = os.path.join(file_dir, "muzero_model.pt")
    # Store model
    torch.save(
        {
            "Muzero_net": (
                muzero.networks._orig_mod.state_dict()
                if hasattr(muzero.networks, "_orig_mod")
                else muzero.networks.state_dict()
            ),
            "Net_optim": muzero.networks.optimiser.state_dict(),
        },
        model_dir,
    )
    logging.info(f"Model saved to {model_dir}")


def log_command_line(
    env_p,
    training_loops,
    min_replay_size,
    lr,
    discount,
    n_mcts_simulations,
    batch_s,
    TD_return,
    priority_replay,
    dev,
    dirichlet_alpha,
    buffer_size,
    TD_step,
    n_disks=None,
):
    command_line = f"Env: {env_p}, Training Loops: {training_loops}, Min replay size: {min_replay_size}, lr: {lr}, discount: {discount}, n. MCTS: {n_mcts_simulations}, batch size: {batch_s}, TD_return: {TD_return}, Priority Buff: {priority_replay}, device: {dev}, dirichlet_alpha: {dirichlet_alpha}, buffer_size: {buffer_size}, TD_step: {TD_step}"

    if env_p == "London":  # if london also print n. of disks
        command_line += f", N. disks: {n_disks}"
    logging.info(command_line)
    return command_line


if __name__ == "__main__":

    ## ======= Select the environment ========
    parser = argparse.ArgumentParser()
    for field in TrainingConfig.__dataclass_fields__.values():
        parser.add_argument(
            f"--{field.name}",
            type=type(field.default),
            default=field.default,
            help=field.metadata.get("help", field.name),
        )

    config = TrainingConfig(**vars(parser.parse_args()))

    set_seed(config.seed)

    ## ========= Useful variables: ===========

    if torch.cuda.is_available():
        dev = "cuda"
    # elif (
    #     hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    # ):  ## for MAC GPU usage
    #     dev = "mps"
    else:
        dev = "cpu"

    ## ========= Initialise env ========
    env, s_space_size, n_action, max_steps, n_disks = get_env(config.env)

    ## ====== Log command line =====

    command_line = log_command_line(
        config.env,
        config.training_loops,
        config.min_replay_size,
        config.lr,
        config.discount,
        config.n_mcts_simulations,
        config.batch_s,
        config.TD_return,
        config.priority_replay,
        dev,
        config.dirichlet_alpha,
        config.buffer_size,
        config.n_TD_step,
        n_disks=n_disks,
    )

    ## ======== Initialise alg. ========
    muzero = Muzero(
        env=env,
        s_space_size=s_space_size,
        n_action=n_action,
        discount=config.discount,
        dirichlet_alpha=config.dirichlet_alpha,
        n_mcts_simulations=config.n_mcts_simulations,
        unroll_n_steps=config.unroll_n_steps,
        batch_s=config.batch_s,
        TD_return=config.TD_return,
        n_TD_step=config.n_TD_step,
        lr=config.lr,
        buffer_size=config.buffer_size,
        priority_replay=config.priority_replay,
        device=dev,
        n_ep_x_loop=config.n_ep_x_loop,
        n_update_x_loop=config.n_update_x_loop,
    )

    ## ======== Run training ==========
    tot_acc = muzero.training_loop(config.training_loops, config.min_replay_size)

    save_results(
        config.env, command_line, {"total_accuracy": tot_acc}, muzero, int(time.time())
    )

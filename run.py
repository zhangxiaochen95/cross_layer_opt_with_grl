import os
import os.path as osp
import json
import yaml
from typing import Any, Optional
from collections.abc import Mapping
from copy import deepcopy
from types import SimpleNamespace as SN
from functools import partial

import random
import numpy as np
import torch as th
import wandb

from gym.spaces.discrete import Discrete
from gym.spaces.multi_discrete import MultiDiscrete
from envs import REGISTRY as env_REGISTRY
from envs.multi_agent_env import MultiAgentEnv

from policies import REGISTRY as policy_REGISTRY
from components.buffers import REGISTRY as buff_REGISTRY
from learners import REGISTRY as learn_REGISTRY, DETERMINISTIC_POLICY_GRADIENT_ALGOS
from runners import REGISTRY as run_REGISTRY

DEFAULT_DATA_DIR = osp.join(osp.abspath(osp.dirname(__file__)), 'results')


def recursive_dict_update(d, u):
    """Merges two dictionaries."""

    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    """Copies configuration."""

    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


def check_args_sanity(config: Mapping[str, Any]) -> dict[str, Any]:
    """Checks the feasibility of configuration."""

    # Setup correct device.
    if config['use_cuda'] and th.cuda.is_available():
        config['device'] = 'cuda:{}'.format(config['cuda_idx'])
    else:
        config['use_cuda'] = False
        config['device'] = 'cpu'
    print(f"Choose to use {config['device']}.")

    # Env specific requirements
    if config['env_id'] == 'mpe':
        assert config['obs'] == 'flat', "MPE only supports flat obs."
        if config['shared_obs'] is not None:
            assert config['shared_obs'] == 'flat', "MPE only supports flat shared obs."
    if config['state'] is not None:
        assert config['state'] == 'flat', f"Unsupported state format 's{config['state']}' is encountered."

    return config


def update_args_from_env(env: MultiAgentEnv, args):
    """Updates args from env."""

    env_info = env.get_env_info()
    args.n_agents = env_info['n_agents']

    if args.env_id.startswith('ad-hoc'):
        args.max_nbrs = getattr(env, 'max_nbrs', None)
        args.n_pow_opts = getattr(env, 'power_options', env.n_pow_lvs)
        args.khops = getattr(env, 'khops', 1)

    if args.runner == 'base':
        if args.rollout_len is None:
            args.rollout_len = env_info['episode_limit']
            print(f"`rollout_len` is set to `episode_limit` as {args.rollout_len}.")
        if args.data_chunk_len is None:
            args.data_chunk_len = env_info['episode_limit']
            print(f"`data_chunk_len` is set to `episode_limit` as {args.data_chunk_len}.")
        assert args.rollout_len is not None and args.data_chunk_len is not None, "Invalid rollout/data chunk length"
    elif args.runner == 'episode':
        args.data_chunk_len = env_info['episode_limit']
        print(f"`data_chunk_len` is set to `episode_limit` as {env_info['episode_limit']}.")
    else:
        raise KeyError("Unrecognized name of runner")

    # Assume that all agents share the same action space and retrieve action info.
    act_space = env.action_space[0]
    args.act_size = env_info['n_actions'][0]
    # Note that `act_size` specifies output layer of modules,
    # while `act_shape` indicates the shape of actions stored in buffers (which may be different from `act_size`).
    if isinstance(act_space, Discrete):
        args.is_discrete = True
        args.is_multi_discrete = False
        args.act_shape = 1 if args.learner not in DETERMINISTIC_POLICY_GRADIENT_ALGOS else args.act_size
    elif isinstance(act_space, MultiDiscrete):
        args.is_discrete = True  # Multi-discrete space is generalization of discrete space.
        args.is_multi_discrete = True
        args.nvec = act_space.nvec.tolist()  # Number of actions in each space
        args.act_shape = len(args.nvec) if args.learner not in DETERMINISTIC_POLICY_GRADIENT_ALGOS else args.act_size
    else:  # TODO: Continuous action use Box space.
        args.is_discrete = False
    # Discrete action selectors use available action mask.
    if args.is_discrete:
        args.pre_decision_fields.append('avail_actions')
    return args


def run(env_id: str, env_kwargs: Mapping[str, Any], seed: int = 0, algo_name: str = 'q',
        train_kwargs: Mapping[str, Any] = dict(), run_tag: Optional[str] = None,
        add_suffix: bool = False, suffix: Optional[str] = None) -> None:
    """Main function to run the training loop"""

    # Set random seed.
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    # Load the default configuration.
    with open(os.path.join(os.path.dirname(__file__), 'config', "default.yaml"), "r") as f:
        config = yaml.safe_load(f)
    # Load hyper-params of algo.
    with open(os.path.join(os.path.dirname(__file__), 'config', f"algos/{algo_name}.yaml"), "r") as f:
        algo_config = yaml.safe_load(f)
    config = recursive_dict_update(config, algo_config)
    # Load mac parameters of communicative agents.
    if config['agent'] == 'comm':
        assert config['comm'] is not None, "Absence of communication protocol for communicative agents!"
        with open(os.path.join(os.path.dirname(__file__), 'config', f"comm/{config['comm']}.yaml"), "r") as f:
            comm_config = yaml.safe_load(f)
        config = recursive_dict_update(config, comm_config)
    # Load preference from train_kwargs.
    config = recursive_dict_update(config, train_kwargs)
    # Add env id.
    config['env_id'] = env_id
    # Make sure the legitimacy of configuration.
    config = check_args_sanity(config)
    del algo_config, train_kwargs  # Delete redundant variables.
    args = SN(**config)  # Convert to simple namespace.

    # Get directory to store models/results.
    # Project identifier includes `env_id` and probably `scenario`.
    scenario = env_kwargs.get('scenario_name', None)
    if add_suffix:
        if suffix is not None:
            project_name = env_id + '_' + suffix
        elif scenario is not None:
            project_name = env_id + '_' + scenario
        else:
            raise Exception("Suffix of project is unavailable.")
    else:
        project_name = env_id
    # Multiple runs are distinguished by algo name and tag.
    run_name = run_tag if run_tag is not None else algo_name
    # Create a subdirectory to distinguish runs with different random seeds.
    args.run_dir = osp.join(DEFAULT_DATA_DIR, project_name, run_name + f"_seed{seed}")
    if not osp.exists(args.run_dir):
        os.makedirs(args.run_dir)
    print(f"Run '{run_name}' under directory '{args.run_dir}'.")

    if args.use_wandb:  # If W&B is used,
        # Runs with the same config except for rand seeds are grouped and their histories are plotted together.
        wandb.init(config=args, project=project_name, group=run_name, name=run_name + f"_seed{seed}", dir=args.run_dir,
                   reinit=True)
        args.wandb_run_dir = wandb.run.dir

    # Define env function.
    env_fn = partial(env_REGISTRY[env_id], **env_kwargs)  # Env function
    test_env_fn = partial(env_REGISTRY[env_id], **env_kwargs)  # Test env function

    # Create runner holding instance(s) of env and get info.
    runner = run_REGISTRY[args.runner](env_fn, test_env_fn, args)
    args = update_args_from_env(runner.env, args)  # Adapt args to env.

    # Setup key components.
    env_info = runner.get_env_info()
    policy = policy_REGISTRY[args.policy](env_info, args)  # Policy making decisions
    policy.to(args.device)  # Move policy to device.

    buffer = buff_REGISTRY[args.buffer](args)  # Buffer holding experiences
    learner = learn_REGISTRY[args.learner](env_info, policy, args)  # Algorithm training policy
    runner.add_components(policy, buffer, learner)  # Add above components to runner.

    # Run the main loop of training.
    runner.run()
    # Clean-up after training.
    runner.cleanup()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=10)
    parser.add_argument('--algo', type=str, default='q')
    parser.add_argument('--tag', type=str, default=None)
    args = parser.parse_args()

    # print(type(args))
    # a = dict(name='bob')
    # args = SN(**a)
    # args2 = argparse.Namespace(**a)
    # print(args)
    # print(vars(args))
    # print(type(args))
    # print(args2)
    # print(vars((args2)))
    # raise ValueError

    # Train UBS coverage.
    # env_id = 'ubs'
    # env_kwargs = dict(scenario_name='simple')

    # # Train Ad Hoc route.
    env_id = 'ad-hoc'
    env_kwargs = dict()

    train_kwargs = dict(use_cuda=True, cuda_idx=0, use_wandb=False, record_tests=True, rollout_len=10, data_chunk_len=5)
    run(env_id, env_kwargs, args.seed, algo_name=args.algo, train_kwargs=train_kwargs, run_tag=args.tag)

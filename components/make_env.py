from components.wrappers.obervation_wrappers import REGISTRY as obs_REGISTRY
from components.wrappers.shared_observation_wrappers import REGISTRY as sobs_REGISTRY
from components.wrappers.state_wrappers import REGISTRY as stat_REGISTRY
from components.wrappers.reward_wrappers import REGISTRY as rew_REGISTRY
from components.wrappers.time_limit import TimeLimit
from components.wrappers.episode_statistics import EpisodeStatistics


def make_env(env_fn, args):
    """Instantiates a raw environment and wraps it."""
    # Instantiate the env.
    env = env_fn()

    # Select format of local/shared observations and global states.
    env = obs_REGISTRY[args.obs](env)
    if args.shared_obs is not None:
        env = sobs_REGISTRY[args.shared_obs](env)
    if args.state is not None:
        env = stat_REGISTRY[args.state](env)

    # Enable multi-agent communication.
    if args.agent == 'comm':
        env = obs_REGISTRY['comm'](env)

    # Even if reward normalization is unused, reward filter is used to track episode returns.
    if args.normalize_reward:
        env = rew_REGISTRY['norm'](env, args)

    # Add proper time limit if requested.
    if args.use_time_limit:
        env = TimeLimit(env, args)

    # Record statistics of episodes.
    env = EpisodeStatistics(env)

    return env


if __name__ == '__main__':
    from functools import partial
    from types import SimpleNamespace as SN
    args = SN(**dict(obs='flat', shared_obs=None, state=None, comm=None, normalize_reward=True, max_reward_limit=10,
                     use_time_limit=True, alert_episode_limit=True))
    from envs import REGISTRY as env_REGISTRY
    env_id = 'mpe'
    env_kwargs = {'scenario_name': 'simple'}
    env_fn = partial(env_REGISTRY[env_id], **env_kwargs)
    env = make_env(env_fn, args)
    env.reset()
    _, _, info = env.step([0])
    print(env.get_obs())
    print(env.get_shared_obs())
    print(env.get_state())
    print(info)
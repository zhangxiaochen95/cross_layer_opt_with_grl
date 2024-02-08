from components.wrappers.wrappers import Wrapper


class TimeLimit(Wrapper):
    """Truncates episode with time limit."""

    def __init__(self, env, args):
        super(TimeLimit, self).__init__(env)

        self._max_episode_steps = getattr(args, 'max_episode_steps', None)
        if self._max_episode_steps is None:
            self._max_episode_steps = self.env.get_env_info()['episode_limit']
        assert self._max_episode_steps is not None, "Maximum step number is not set for wrapper `TimeLimit`."

        self._elapsed_steps = None  # Number of elapsed steps in current episode
        self._alert_ep_lim = args.alert_episode_limit  # Whether arrival of episode limit is alerted in info

    def reset(self):
        self._elapsed_steps = 0
        self.env.reset()

    def step(self, actions):
        self._elapsed_steps += 1
        rewards, terminated, info = self.env.step(actions)
        info['truncated'] = False
        if (self._elapsed_steps == self._max_episode_steps) and not terminated:
            terminated = True  # Truncate episode.
            if self._alert_ep_lim:
                info['truncated'] = True  # Alert arrival of episode limit into info.
        return rewards, terminated, info

    def get_env_info(self):
        env_info = self.env.get_env_info()
        env_info["episode_limit"] = self._max_episode_steps
        return env_info

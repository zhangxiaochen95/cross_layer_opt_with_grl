from components.wrappers.wrappers import Wrapper


class EpisodeStatistics(Wrapper):
    """Tracks episode length and returns."""

    def __init__(self, env):
        super(EpisodeStatistics, self).__init__(env)
        self.ep_len = None  # Length of episode
        self.ep_ret = None  # Return of episode

    def reset(self):
        super(EpisodeStatistics, self).reset()

        # Reset statistics.
        self.ep_len = 0
        self.ep_ret = 0.0

    def step(self, actions):
        rewards, terminated, info = super(EpisodeStatistics, self).step(actions)

        self.ep_len += 1  # One step elapse.
        # Since reward filter may be used, use actual rewards as long as they are available.
        self.ep_ret += info.get('actual_rewards', rewards).mean()  # Average cross agents.

        # When episode terminates, add episode statistics to info.
        if terminated:
            episode_statistics = dict(EpLen=self.ep_len, EpRet=self.ep_ret)
            info.update(episode_statistics)

        return rewards, terminated, info

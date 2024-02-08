from envs.multi_agent_env import MultiAgentEnv
from abc import abstractmethod


class Wrapper(object):
    def __init__(self, env: MultiAgentEnv):
        self.env = env

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)

    def reset(self):
        self.env.reset()

    def step(self, actions):
        return self.env.step(actions)

    def close(self):
        self.env.close()


class ObservationWrapper(Wrapper):
    def __init__(self, env):
        super(ObservationWrapper, self).__init__(env)

    def get_obs(self):
        obs = self.env.get_obs()
        obs = self.observation(obs)
        return obs

    @ abstractmethod
    def get_obs_size(self):
        raise NotImplementedError

    @abstractmethod
    def observation(self, obs):
        raise NotImplementedError

    def get_env_info(self):
        env_info = self.env.get_env_info()
        env_info.update(dict(obs_shape=self.get_obs_size()))
        return env_info


class SharedObservationWrapper(Wrapper):
    def __init__(self, env):
        super(SharedObservationWrapper, self).__init__(env)

    def get_shared_obs(self):
        obs = self.env.get_shared_obs()
        obs = self.shared_observation(obs)
        return obs

    @ abstractmethod
    def get_shared_obs_size(self):
        raise NotImplementedError

    @abstractmethod
    def shared_observation(self, shared_obs):
        raise NotImplementedError

    def get_env_info(self):
        env_info = self.env.get_env_info()
        env_info.update(dict(shared_obs_shape=self.get_shared_obs_size()))
        return env_info


class StateWrapper(Wrapper):
    def __init__(self, env):
        super(StateWrapper, self).__init__(env)

    def get_state(self):
        stat = self.env.get_state()
        stat = self.state(stat)
        return stat

    @abstractmethod
    def get_state_size(self):
        raise NotImplementedError

    @abstractmethod
    def state(self, stat):
        raise NotImplementedError

    def get_env_info(self):
        env_info = self.env.get_env_info()
        env_info.update(dict(state_shape=self.get_state_size()))
        return env_info


class RewardWrapper(Wrapper):
    def __init__(self, env):
        super(RewardWrapper, self).__init__(env)

    def step(self, actions):
        rewards, terminated, info = self.env.step(actions)

        # Record actual rewards.
        info.update(dict(actual_rewards=rewards))
        # Filter rewards.
        rewards = self.reward(rewards)

        return rewards, terminated, info
    
    @abstractmethod
    def reward(self, rewards):
        raise NotImplementedError

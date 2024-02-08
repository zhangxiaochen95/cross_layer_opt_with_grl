class MultiAgentEnv(object):
    def step(self, actions):
        """Returns reward, terminated, info."""
        raise NotImplementedError

    def get_obs(self):
        """Returns all agent observations in a list."""
        raise NotImplementedError

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        raise NotImplementedError

    def get_obs_size(self):
        """Returns the size of the observation."""
        raise NotImplementedError

    def get_shared_obs(self):
        """Returns all agent shared observations in a list."""
        raise NotImplementedError

    def get_shared_obs_size(self):
        """Returns the size of the shared observation."""
        return

    def get_state(self):
        """Returns the global state."""
        raise NotImplementedError

    def get_state_size(self):
        """Returns the size of the global state."""
        return

    def get_avail_actions(self):
        """Returns the available actions of all models in a list."""
        raise NotImplementedError

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        raise NotImplementedError

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        raise NotImplementedError

    def reset(self):
        """Returns initial observations and states."""
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        pass

    def seed(self):
        raise NotImplementedError

    def save_replay(self):
        """Save a replay."""
        raise NotImplementedError

    def get_env_info(self):
        env_info = {
            "obs_shape": self.get_obs_size(),
            "shared_obs_shape": self.get_shared_obs_size(),
            "state_shape": self.get_state_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": getattr(self, 'episode_limit', None),
        }
        return env_info

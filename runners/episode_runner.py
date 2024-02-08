import os.path as osp
import time

import torch as th
from torch import Tensor

from components.misc import *
from learners import DETERMINISTIC_POLICY_GRADIENT_ALGOS
from runners.base_runner import BaseRunner


class EpisodeRunner(BaseRunner):
    """Episode runner

    Each call of `collect` method runs an episode to its maximum length.
    """

    def __init__(self, env_fn, test_env_fn, args):
        super(EpisodeRunner, self).__init__(env_fn, test_env_fn, args)
        self.max_episode_steps = self.env.get_env_info()['episode_limit']
        print(f"`max_episode_steps` of EpisodeRunner is set to `episode_limit` as {self.max_episode_steps}.")
        assert self.max_episode_steps is not None, "Maximum episode length is absent for `EpisodeRunner`."

        self.curr_episode = []

    def warmup(self):
        """Prepares for training."""
        # Reset variables.
        self.t_env = 0  # Reset timestep counter.

        # Reset components.
        self.env.reset()  # Reset env.
        self.learner.reset()  # Reset learner.
        rnn_states = self.get_init_hidden()  # Initial RNN states

        # Let agents select random actions to fill replay buffer.
        self.t_warmup = 0  # Reset warm-up step counter.
        while (self.t_warmup < self.args.warmup_steps) or not self.buffer.can_sample(self.args.batch_size):
            rnn_states = self.collect(rnn_states, frozen=True)  # Random interaction

        return rnn_states

    def collect(self, rnn_states: dict[str, Tensor], frozen: bool = False) -> dict[str, Tensor]:
        """Explores a fixed number of timesteps to collect experiences."""
        act_mode = 'rand' if frozen else 'explore'
        self.policy.eval()  # Set policy to eval mode.
        self.learner.eval()  # Set modules held by learner to eval mode.

        terminated, filled = False, True
        # print(f"\nEpisode begin. Call reset.")
        self.env.reset()
        # print(f"Complete reset. agent = {self.env.agent}")

        self.curr_episode = []  # Clear current episode
        for t in range(self.max_episode_steps + 1):
            if not terminated:  # Episode continues.
                if frozen:
                    self.t_warmup += 1
                else:
                    self.t_env += 1
            else:  # Fill the episode to maximum length after termination.
                filled = False

            # Build pre-decision data.
            pre_decision_data = dict(**self.get_inputs_from_env(self.env), **rnn_states)

            # Select actions following epsilon-greedy strategy.
            actions, h = self.policy.act(pre_decision_data['obs'], pre_decision_data['h'],
                                         pre_decision_data['avail_actions'], self.t_env, mode=act_mode)

            if filled:  # Call environment step.
                # print(f"t = {t}, step with action {actions.squeeze()}")
                rewards, terminated, info = self.env.step(self.get_actions_to_env(actions, self.env))
            else:  # Pseudo-transition
                # print(f"t = {t}, pseudo-step with action {actions.squeeze()}")
                rewards, terminated, info = np.zeros(self.env.n_agents, dtype=np.float32), True, dict()

            # When deterministic policy gradient is used, apply one-hot encoding to discrete actions.
            if (self.args.learner in DETERMINISTIC_POLICY_GRADIENT_ALGOS) and self.args.is_discrete:
                n_classes = self.args.nvec if self.args.is_multi_discrete else self.args.act_size
                actions = onehot_from_actions(actions, n_classes)

            # Call learner step to get required data (e.g., critic RNN states for off-policy AC algos).
            rnn_states = dict(h=h, **self.learner.step(pre_decision_data, actions))  # Next RNN states

            # Collect post-decision data.
            post_decision_data = {
                'actions': actions, 'rewards': rewards,
                'terminated': terminated != info.get('truncated', False),
            }
            # Save transition to replay buffer.
            self.cache(**pre_decision_data, **post_decision_data, filled=filled)

            if filled and not frozen:  # Avoid recurring call of following functions.
                # if terminated:  # End of episode handling.
                #     if 'truncated' in info:  # Drop indicator of episode limit.
                #         del info['truncated']
                #     self.log_metrics(info)

                # Regularly save checkpoints.
                if (self.t_env % self.args.save_interval == 0) or (self.t_env >= self.args.total_env_steps):
                    save_path = osp.join(self.model_dir, f'checkpoint_t{self.t_env}.pt')
                    self.learner.save_checkpoint(save_path)

                # Test the performance of trained models.
                if self.t_env % self.args.test_interval == 0:
                    self.test_agent()

                # Handle the end of session.
                if self.t_env % self.args.steps_per_session == 0:
                    session = self.t_env // self.args.steps_per_session  # Index of session
                    print(f"Finish session {session} at step {self.t_env}/{self.args.total_env_steps} "
                          f"after {(time.time() - self.start_time):.1f}s.")
                    # Call lr scheduler(s) if enabled.
                    if self.args.anneal_lr:
                        self.learner.schedule_lr()

        # if getattr(self.args, 'retrace_rewards', False):
        #     # print(f"Final agent route = {self.env.agent}.")
        #     # Reshape rewards at the end of episode.
        #     after_rewards = self.env.retrace_onward_bottleneck_rate()
        #     # print(f"connected = {self.env.agent.is_connected}, after_rewards = {after_rewards}, per_hop_rate = {self.env.get_per_hop_rate(self.env.agent)}")
        #     # if self.env.agent.is_connected:
        #     #     self.env.save_replay(show_img=True)
        #     #     raise ValueError
        #     for t, r in enumerate(after_rewards):
        #         if t < len(self.curr_episode):
        #             self.curr_episode[t]["rewards"] = th.tensor(r, dtype=th.float, device=self.device).reshape(1, self.env.n_agents)
        #         else:
        #             print("Number of hops > `max_episode_steps`.")
        #             print(f"agent = {self.env.agent}, n_hops = {self.env.agent.n_hops}")
        #             print(f"after_rewards = {after_rewards}")
        #             self.env.save_replay(show_img=True)
        #             raise ValueError

        # Insert transitions to replay buffer.
        for transition in self.curr_episode:
            self.buffer.insert(transition)

        return rnn_states

    def cache(self, obs, actions, rewards, terminated, avail_actions, filled=True, **kwargs):
        # Reshape entries and hold them with a dict.
        transition = dict(obs=obs, avail_actions=avail_actions,
                          actions=actions.view(self.env.n_agents, self.args.act_shape),
                          rewards=th.tensor(rewards, dtype=th.float, device=self.device).reshape(1, self.env.n_agents),
                          terminated=th.tensor(terminated, dtype=th.int, device=self.device).reshape(1, 1),
                          filled=th.tensor(filled, dtype=th.int, device=self.device).reshape(1, 1))
        transition.update(**kwargs)
        self.curr_episode.append(transition)


from copy import deepcopy
import torch as th
import torch.nn as nn

from components.misc import *
from learners.base_learner import BaseLearner
from modules.mixers import REGISTRY as mix_REGISTRY


class QLearner(BaseLearner):
    """Multi-agent Q learning with recurrent models"""

    def __init__(self, env_info, policy, args) -> None:

        self.device = args.device
        self.online_policy = policy
        print(self.online_policy)
        self.params = list(self.online_policy.parameters())
        self.target_policy = deepcopy(self.online_policy)

        self.args = args  # Arguments
        self.n_agents = policy.n_agents  # Number of agents
        self.n_updates = None  # Number of completed updates

        # Set mixer to combine individual state-action values to global ones.
        self.mixer = None
        if hasattr(args, 'mixer'):  # Mixer is specified.
            self.mixer = mix_REGISTRY[args.mixer](env_info['state_shape'], self.n_agents, args).to(self.device)
            print(f"Mixer: \n{self.mixer}")
            self.params += list(self.mixer.parameters())
            self.target_mixer = deepcopy(self.mixer)

        # Define optimizer.
        self._use_huber_loss = args.use_huber_loss  # Whether Huber loss is used.
        self.optimizer = th.optim.Adam(self.params, lr=args.lr, eps=args.optim_eps)

        # Set learning rate scheduler.
        if self.args.anneal_lr:
            lr_lam = get_clipped_linear_decay(total_steps=10, threshold=0.4)
            self.lr_scheduler = th.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lam, verbose=True)

        self._use_double_q = args.use_double_q  # Whether double Q-learning is used

    def reset(self):
        """Resets the learner."""
        self.n_updates = 0  # Reset number of updates

    def schedule_lr(self):
        if self.args.anneal_lr:
            self.lr_scheduler.step()

    def update(self, buffer, batch_size: int):
        """Updates parameters of recurrent models via BPTT."""
        self.online_policy.train()  # Set policy to train mode.
        self.target_policy.train()

        batch = buffer.recall(batch_size)  # Batched samples from reply buffer

        obs = [batch['obs'][t].to(self.device) for t in range(len(batch['obs']))]
        avail_actions = th.stack(batch['avail_actions']).to(self.device) if 'avail_actions' in batch else None
        actions = th.stack(batch['actions']).to(self.device)  # Shape (data_chunk_len, batch_size * n_agents, 1)
        rewards = th.stack(batch['rewards']).to(self.device)  # Shape (data_chunk_len, batch_size, n_agents)
        terminated = th.stack(batch['terminated']).to(self.device)  # Shape (data_chunk_len, batch_size, 1)
        mask = th.stack(batch['filled']).to(self.device)  # Shape (data_chunk_len, batch_size, 1)
        h, h_targ = batch['h'][0].to(self.device), batch['h'][1].to(self.device)  # Get initial hidden states.

        # print(f"rewards.size() = {rewards.size()}, mask.size() = {mask.size()}")
        # for t in range(mask.size(0)):
        #     print(f"t = {t}, rewards = {rewards[t].squeeze()}, terminated = {terminated[t].squeeze()},  mask = {mask[t].squeeze()}")

        assert self.args.data_chunk_len == len(obs) - 1, "Improper length of sequences found in mini-batch."
        # Get maximum filled steps.
        horizon = 0  # Time horizon of forward computation
        for t in range(self.args.data_chunk_len):
            if mask[t].any():
                horizon = t + 1
        # Truncate batch sequences to horizon.
        obs, avail_actions = obs[:horizon + 1], avail_actions[:horizon + 1]
        actions, rewards, terminated, mask = actions[:horizon], rewards[:horizon], terminated[:horizon], mask[:horizon]

        # print(f"len(obs) = {len(obs)}")
        # print(f"actions.size() = {actions.size()}")
        # print(f"After truncation,")
        # print(f"rewards = \n{rewards.squeeze(-1)}")
        # print(f"terminated = \n{terminated.squeeze(-1)}")
        # print(f"mask = \n{mask.squeeze(-1)}")

        online_agent_out, target_agent_out = [], []
        for t in range(horizon):
            # Policy network predicts the Q(o_{t},a).
            logits, h = self.online_policy.forward(obs[t], h)
            online_agent_out.append(logits)
            # Reset RNN states of policy network when termination of episode is encountered.
            h = h * (1 - terminated[t]).expand(batch_size, self.n_agents).reshape(-1, 1)

            # Target network predicts Q(o_{t+1}, a).
            with th.no_grad():
                next_logits, h_targ = self.target_policy.forward(obs[t + 1], h_targ)
                target_agent_out.append(next_logits)
                if t + 1 < horizon:
                    # Reset RNN states of target network when termination of episode is encountered.
                    h_targ = h_targ * (1 - terminated[t + 1]).expand(batch_size, self.n_agents).reshape(-1, 1)

        # Let policy network make predictions for next obs of the last timestep.
        logits, h = self.online_policy.forward(obs[horizon], h)
        online_agent_out.append(logits)

        # Stack outputs of policy/target networks over time.
        online_logits, target_logits = th.stack(online_agent_out), th.stack(target_agent_out)

        # Get Q(o_{t}, a_{t}) from online logits.
        q_vals = online_logits[:-1].gather(2, actions).view(horizon, batch_size, self.n_agents)

        # Get V(o_{t+1}) from target logits.
        if not self._use_double_q:  # Greedy selection of a_{t+1}
            if avail_actions is not None:
                target_logits[avail_actions[1:] == 0] = -1e10  # Mask unavailable actions.
            # Pick the largest Q values from target network.
            next_v = target_logits.max(2, keepdim=True)[0]
        else:  # Double Q-learning is used.
            online_logits_detach = online_logits.clone().detach()  # Duplicate output from policy network.
            if avail_actions is not None:
                online_logits_detach[avail_actions == 0] = -1e10  # Mask unavailable actions.
            next_actions = th.argmax(online_logits_detach[1:], 2, keepdim=True)  # Next actions selected by policy network
            next_v = target_logits.gather(2, next_actions)  # Pick Q values of specified by next actions.

        next_v = next_v.view(horizon, batch_size, self.n_agents)  # Reshaped V(o_{t+1}).

        # When mixer is used, combine individual Q values from agents.
        if self.mixer is not None:
            # Aggregate individual rewards to global ones.
            rewards = rewards.mean(axis=-1, keepdims=True)
            # Get env states.
            states = th.stack(batch['state']).to(self.device)
            # Mix individual values v(o_{t}^{1}),...,v(o_{t}^{n}) and states s_{t} to global ones v(s_{t}).
            q_vals = self.mixer(q_vals, states[:-1])
            with th.no_grad():
                # Likewise, get v(s_{t+1}).
                next_v = self.target_mixer(next_v, states[1:])

        # Obtain one-step target of Q-learning as r_{t} + gamma * (1 - d) * v(s_{t+1}).
        q_targets = rewards + self.args.gamma * (1 - terminated) * next_v

        # Compute masked MSE loss.
        td_error = q_vals - q_targets  # TD error
        loss = huber_loss(td_error, mask) if self._use_huber_loss else mse_loss(td_error, mask)  # Q loss

        # Call one step of gradient descent.
        self.optimizer.zero_grad()
        loss.backward()  # Back propagation
        grad_norm = nn.utils.clip_grad_norm_(self.params, max_norm=self.args.max_grad_norm)  # Gradient-clipping
        self.optimizer.step()  # Call update.

        self.n_updates += 1  # Finish one step of policy update.
        # Sync parameters of target networks.
        if self.args.target_update_mode == "soft":
            self.soft_target_sync()
        elif self.args.target_update_mode == "hard":
            if self.n_updates % self.args.target_update_interval == 0:
                self.hard_target_sync()

        update_info = dict(LossQ=loss.item(), QVals=q_vals.detach().cpu().numpy(), GradNorm=grad_norm)
        return update_info

    @th.no_grad()
    def soft_target_sync(self) -> None:
        """Applies soft update of target networks via polyak averaging."""
        soft_target_update(self.online_policy.model, self.target_policy.model, self.args.polyak)
        if self.mixer is not None:
            soft_target_update(self.mixer, self.target_mixer, self.args.polyak)

    @th.no_grad()
    def hard_target_sync(self) -> None:
        """Copies parameters of policy networks to target networks."""
        self.target_policy.load_state(self.online_policy)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def save_checkpoint(self, path) -> None:
        """Saves states of components."""
        checkpoint = dict()  # Current timestep
        checkpoint['model'] = self.online_policy.model.state_dict()  # Parameters of policy network
        if self.mixer is not None:
            checkpoint['mixer'] = self.mixer.state_dict()  # Parameters of mixer
        checkpoint['opt'] = self.optimizer.state_dict()  # State of optimizer
        if self.args.anneal_lr:
            checkpoint['scheduler'] = self.lr_scheduler.state_dict()  # State of learning rate scheduler
        # Save to path.
        th.save(checkpoint, path)

    def load_checkpoint(self, path):
        """Loads states of components to resume training or test."""
        # Load checkpoint.
        checkpoint = th.load(path, map_location=self.device)
        # Load parameters of policy and target networks.
        self.online_policy.model.load_state(checkpoint['model'])
        self.target_policy.model.load_state(checkpoint['model'])
        # Load parameters of mixers.
        if self.mixer is not None:
            self.mixer.load_state(checkpoint['mixer'])
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        # Load state of optimizer.
        self.optimizer.load_state_dict(checkpoint['opt'])
        # Load state of learning rate scheduler.
        if self.args.anneal_lr:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler'])

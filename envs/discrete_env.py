from typing import List
import torch
from matplotlib import pyplot as plt

from envs.base_env import BaseEnv
from utils.lookups import name_to_distribution


class DiscreteShiftedProximityEnv(BaseEnv):
    def __init__(
            self,
            n_state: int,
            n_action: int,
            opt_shift: float,
            decay_rate: float,
    ):
        super().__init__(prior='categorical', prior_params={'probs': torch.ones((n_state,))})

        self.n_state = n_state
        self.n_action = n_action
        self.opt_shift = opt_shift
        self.decay_rate = decay_rate

    def step(self, actions_n: torch.Tensor, fix_state_between_actions: bool):
        if actions_n.ndim == 0:
            actions_n = actions_n.unsqueeze(-1)

        assert actions_n.ndim == 1

        rews_n = torch.zeros_like(actions_n, dtype=torch.float)
        states_n = torch.zeros_like(actions_n, dtype=torch.long)
        shifts_n = torch.zeros_like(actions_n, dtype=torch.float)
        for i, action in enumerate(actions_n):
            # Find the reward
            ss = (self.state + self.opt_shift) % self.n_action  # Shifted state
            dist = torch.minimum((action - ss) % self.n_action, (ss - action) % self.n_action)
            rews_n[i] = self.decay_rate**dist

            # Store
            states_n[i] = self.state
            shifts_n[i] = self.opt_shift

            # Update the state
            if not fix_state_between_actions:
                self.reset()

        if fix_state_between_actions:
            self.reset()

        return states_n, rews_n, shifts_n


class DiscreteMultiShiftedProximityEnv(BaseEnv):
    def __init__(
            self,
            n_state: int,
            n_action: int,
            opt_shifts: List[float],
            decay_rate: float,
    ):
        super().__init__(prior='categorical', prior_params={'probs': torch.ones((n_state,))})

        self.n_state = n_state
        self.n_action = n_action
        self.opt_shift = opt_shifts[0]  # Default init. Has no effect
        self.decay_rate = decay_rate

        self._env = DiscreteShiftedProximityEnv(n_state, n_action, self.opt_shift, decay_rate)
        self.state = self._env.state

        self.opt_shifts = opt_shifts
        self.opt_shift_prior = name_to_distribution('categorical')(probs=torch.ones((len(opt_shifts),)))

    def step(self, actions_n: torch.Tensor, fix_state_between_actions: bool):
        # A fixed opt_shift will be used for every call

        self.opt_shift = self.opt_shifts[self.opt_shift_prior.sample().item()]
        self._env.opt_shift = self.opt_shift

        step_output = self._env.step(actions_n, fix_state_between_actions)

        self.state = self._env.state

        return step_output


if __name__ == '__main__':
    # For debugging
    # env = DiscreteShiftedProximityEnv(5, 5, -1, 0.5)
    env = DiscreteMultiShiftedProximityEnv(9, 9, [-2, 2], 0.9)
    rew_sa = torch.zeros((env.n_state, env.n_action))
    cnt_sa = torch.zeros((env.n_state, env.n_action))
    ac_dist = torch.distributions.Categorical(probs=torch.ones((env.n_action,)))

    for _ in range(100000):
        a = ac_dist.sample()
        s, r, _ = env.step(a, fix_state_between_actions=True)
        s, r = s[0], r[0]

        cnt_sa[s, a] += 1
        rew_sa[s, a] += r

    rew_sa = rew_sa / cnt_sa

    plt.imshow(rew_sa)
    plt.colorbar()
    plt.show()

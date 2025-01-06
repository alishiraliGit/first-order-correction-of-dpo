from typing import List, Union
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from envs.base_env import BaseEnv
from utils.lookups import name_to_distribution


class DecayFunc:
    LINEAR = 'linear'
    QUADRATIC = 'quadratic'
    EXPONENTIAL = 'exponential'
    GAUSSIAN = 'gaussian'

    def __init__(self, decay_func: str, rate: float):
        self.rate = rate

        self.forward = self.str_to_decay_func(decay_func)

    def linear(self, x):
        return torch.clamp(1 - self.rate*x, min=0.)

    def quadratic(self, x):
        return torch.clamp(1 - self.rate*(x**2), min=0.)

    def exponential(self, x):
        return self.rate**x

    def gaussian(self, x):
        return self.rate**(x**2)

    def str_to_decay_func(self, decay_func_str):
        if decay_func_str == self.LINEAR:
            return self.linear
        elif decay_func_str == self.EXPONENTIAL:
            return self.exponential
        elif decay_func_str == self.QUADRATIC:
            return self.quadratic
        elif decay_func_str == self.GAUSSIAN:
            return self.gaussian
        else:
            raise NotImplementedError


class DiscreteShiftedProximityEnv(BaseEnv):
    def __init__(
            self,
            n_state: int,
            n_action: int,
            shift: float,
            decay_func: str,
            decay_rate: float,
            rew_scale: float,
    ):
        super().__init__(prior='categorical', prior_params={'probs': torch.ones((n_state,))})

        self.n_state = n_state
        self.n_action = n_action
        self.shift = shift
        self.decay_func_str = decay_func
        self.decay_func = DecayFunc(decay_func, rate=decay_rate)
        self.rew_scale = rew_scale

    def reward(self, action: torch.Tensor):
        ss = (self.state + self.shift) % self.n_action  # Shifted state
        dist = torch.minimum((action - ss) % self.n_action, (ss - action) % self.n_action)
        # noinspection PyArgumentList
        return self.rew_scale * self.decay_func.forward(dist)

    def step(self, actions_n: torch.Tensor, fix_state_between_actions: bool):
        if actions_n.ndim == 0:
            actions_n = actions_n.unsqueeze(-1)

        assert actions_n.ndim == 1

        rews_n = torch.zeros_like(actions_n, dtype=torch.float)
        states_n = torch.zeros_like(actions_n, dtype=torch.long)
        shifts_n = torch.zeros_like(actions_n, dtype=torch.float)
        for i, action in enumerate(actions_n):
            # Find the reward
            rews_n[i] = self.reward(action)

            # Store
            states_n[i] = self.state
            shifts_n[i] = self.shift

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
            shifts: List[float],
            decay_func: str,
            decay_rates: Union[float, List[float]],
            rew_scales: Union[float, List[float]],
    ):
        super().__init__(prior='categorical', prior_params={'probs': torch.ones((n_state,))})

        if isinstance(decay_rates, float):
            decay_rates = [decay_rates]*len(shifts)
        if isinstance(rew_scales, float):
            rew_scales = [rew_scales]*len(shifts)

        self.n_state = n_state
        self.n_action = n_action
        self.shifts = shifts
        self.decay_func_str = decay_func
        self.decay_rates = decay_rates
        self.rew_scales = rew_scales

        self.envs = []
        for shift, decay_rate, rew_scale in zip(shifts, decay_rates, rew_scales):
            env = DiscreteShiftedProximityEnv(
                n_state=n_state,
                n_action=n_action,
                shift=shift,
                decay_func=decay_func,
                decay_rate=decay_rate,
                rew_scale=rew_scale,
            )
            self.envs.append(env)

        self.state = self.envs[0].state
        self.env_prior = name_to_distribution('categorical')(probs=torch.ones((len(shifts),)))

    def update_state(self, state):
        self.state = state
        for env in self.envs:
            env.state = self.state

    def reward(self, action: torch.Tensor):
        rews = [env.reward(action) for env in self.envs]
        mean_rew = 0
        for p, rew in zip(self.env_prior.probs, rews):
            mean_rew += p*rew

        return mean_rew

    def step(self, actions_n: torch.Tensor, fix_state_between_actions: bool):
        # A fixed env will be used for every call
        idx = self.env_prior.sample().item()
        env = self.envs[idx]

        step_output = env.step(actions_n, fix_state_between_actions)

        self.update_state(env.state)

        return step_output


if __name__ == '__main__':
    # For debugging
    e = DiscreteMultiShiftedProximityEnv(1, 40, [-10, 0, 10], 'quadratic', [0.0075, 0.01, 0.0075], [4, 1, 4])
    rew_sa = torch.zeros((e.n_state, e.n_action))
    cnt_sa = torch.zeros((e.n_state, e.n_action))
    ac_dist = torch.distributions.Categorical(probs=torch.ones((e.n_action,)))

    for _ in tqdm(range(30000)):
        a = ac_dist.sample()
        s, r, _ = e.step(a, fix_state_between_actions=True)
        s, r = s[0], r[0]

        cnt_sa[s, a] += 1
        rew_sa[s, a] += r

    rew_sa = rew_sa / cnt_sa

    plt.figure()
    plt.imshow(rew_sa)
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.plot(rew_sa[0])
    plt.show()

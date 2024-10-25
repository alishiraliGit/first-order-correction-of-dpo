import abc
import torch
from tqdm import tqdm

from envs.base_env import BaseEnv
from preferencemodels.base_preference import BasePreference
from solvers.base_solver import BaseSolver
from utils.logger import Logger
from policies.base_policy import BasePolicy


class BaseSimulatorTrainer(abc.ABC):
    def __init__(
            self,
            env: BaseEnv,
            preference_model: BasePreference,
            solver: BaseSolver,
            logger: Logger,
    ):
        self.env = env
        self.preference_model = preference_model
        self.solver = solver
        self.logger = logger

    def sample(self, sampling_policy: BasePolicy, n: int):
        states_n = torch.zeros((n,), dtype=torch.long)
        actions_1_n = torch.zeros((n,), dtype=torch.long)
        actions_2_n = torch.zeros((n,), dtype=torch.long)
        prefs_n = torch.zeros((n,), dtype=torch.long)
        us_n = torch.zeros((n,), dtype=torch.float)

        for i in range(n):
            states = torch.tensor([self.env.state, self.env.state])

            # Get two actions from the ref_policy
            actions = sampling_policy.get_actions(states)

            # Find the preferred action
            _, rews, us = self.env.step(actions, fix_state_between_actions=True)

            prefs = self.preference_model.sample(rews[0:1], rews[1:2])

            # Store
            states_n[i] = states[0]
            actions_1_n[i] = actions[0]
            actions_2_n[i] = actions[1]
            prefs_n[i] = prefs[0]
            us_n[i] = us[0]

        return states_n, actions_1_n, actions_2_n, prefs_n, us_n

    @abc.abstractmethod
    def train_one_step(self):
        pass

    def train(self, steps: int):
        for step in tqdm(range(steps), desc='training'):
            # Train
            training_log = self.train_one_step()

            # Log
            for k, v in training_log.items():
                self.logger.log(k, v, step)

        self.logger.flush()

        # Save
        self.save(self.logger.log_dir)

    def save(self, save_path: str):
        self.solver.save(save_path)

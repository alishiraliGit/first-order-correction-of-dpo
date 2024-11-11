import torch
from torch import nn
from typing import Optional

from policies.base_policy import BasePolicy
from policies.mlp_policy import DiscreteMLPPolicy
from solvers.dpo import DPO
from utils.load_and_save_utils import append_class_to_path, load_class
from utils import pytorch_utils as ptu


class FixedVarCorrectedDPO(DPO):
    def __init__(
            self,
            policy: BasePolicy,
            ref_policy: BasePolicy,
            beta: float,
            lr: float,
            var: float,
    ):
        super().__init__(policy, ref_policy, beta, lr)

        self.var = var

    def update(
            self,
            states_n: torch.Tensor,
            actions_1_n: torch.Tensor,
            actions_2_n: torch.Tensor,
            prefs_n: torch.Tensor,
            us_n: torch.Tensor,
    ):
        # Calc. likelihood
        sigmoid_n = self.calc_likelihood(states_n, actions_1_n, actions_2_n, prefs_n)

        # Calc. loss
        loss = -torch.log(sigmoid_n + 0.5*self.var*(1 - 2*sigmoid_n)/sigmoid_n/(1 - sigmoid_n)).mean()

        # Step the optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'training_loss': loss.item()}

    def save(self, save_path: str):
        assert self.policy.__class__.__name__ != self.ref_policy.__class__.__name__

        self.policy.save(save_path)
        self.ref_policy.save(save_path)

        torch.save(
            {
                'kwargs': {
                    'beta': self.beta,
                    'lr': self.lr,
                    'var': self.var,
                },
                'policies': {
                    'policy_appendix': append_class_to_path('', self.policy.__class__),
                    'ref_policy_appendix': append_class_to_path('', self.ref_policy.__class__),
                },
                'optimizer_state_dict': self.optimizer.state_dict(),
            },
            self._append_to_path(save_path),
        )


class CorrectedDPO(FixedVarCorrectedDPO):
    def __init__(
            self,
            policy: BasePolicy,
            ref_policy: BasePolicy,
            beta: float,
            lr: float,
            var: float,
    ):
        super().__init__(policy, ref_policy, beta, lr, var)

        self.cnt_saau = torch.ones((policy.n_state, policy.n_action, policy.n_action, 2), dtype=torch.long)

    def update(
            self,
            states_n: torch.Tensor,
            actions_1_n: torch.Tensor,
            actions_2_n: torch.Tensor,
            prefs_n: torch.Tensor,
            us_n: torch.Tensor,
    ):
        n = len(states_n)

        # Calc. prob. pref
        dist_na = self.policy(states_n)
        ref_dist_na = self.ref_policy(states_n)

        actions_n = torch.concat([actions_1_n.unsqueeze(-1), actions_2_n.unsqueeze(-1)], dim=1)
        actions_w_n = actions_n[torch.arange(n), prefs_n]  # Winners
        actions_l_n = actions_n[torch.arange(n), 1 - prefs_n]  # Losers

        log_probs_w_n = dist_na.log_prob(actions_w_n)
        log_probs_l_n = dist_na.log_prob(actions_l_n)

        ref_log_probs_w_n = ref_dist_na.log_prob(actions_w_n)
        ref_log_probs_l_n = ref_dist_na.log_prob(actions_l_n)

        sigmoid_n = torch.sigmoid(
            self.beta*(log_probs_w_n - log_probs_l_n) - self.beta*(ref_log_probs_w_n - ref_log_probs_l_n)
        )

        # Update counter
        self.cnt_saau[states_n, actions_w_n, actions_l_n, (us_n > 0)*1] += 1

        # Calc. var
        prob_1_n = self.cnt_saau[states_n, actions_w_n, actions_l_n, 1] / (
            self.cnt_saau[states_n, actions_w_n, actions_l_n, 1] + self.cnt_saau[states_n, actions_l_n, actions_w_n, 1]
        )

        prob_0_n = self.cnt_saau[states_n, actions_w_n, actions_l_n, 0] / (
            self.cnt_saau[states_n, actions_w_n, actions_l_n, 0] + self.cnt_saau[states_n, actions_l_n, actions_w_n, 0]
        )

        prob_n = (prob_1_n + prob_0_n)/2
        var_n = 0.5*(prob_1_n - prob_n)**2 + 0.5*(prob_0_n - prob_n)**2

        # Calc. loss
        loss = -torch.log(sigmoid_n + 0.5*self.var*(1 - 2*sigmoid_n)/sigmoid_n/(1 - sigmoid_n)*var_n).mean()

        # Step the optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'training_loss': loss.item()}


class EstVarCorrectedDPO(DPO):
    def __init__(
            self,
            policy: DiscreteMLPPolicy,
            ref_policy: BasePolicy,
            beta: float,
            lr: float,
            var_multiplier: float,
            step: int = 0,
            start_correction_after_step: int = 0,
            joint_likelihood_mdl: Optional[nn.Module] = None,
            joint_likelihood_params: Optional[dict] = None,
    ):
        super().__init__(policy, ref_policy, beta, lr)

        self.n_state = policy.n_state
        self.n_action = policy.n_action

        assert (joint_likelihood_mdl is not None) or (joint_likelihood_params is not None)

        self.var_multiplier = var_multiplier

        self.step = step
        self.start_correction_after_step = start_correction_after_step

        self.joint_likelihood_params = joint_likelihood_params
        if joint_likelihood_mdl is None:
            self.joint_likelihood_mdl = ptu.build_mlp(
                input_size=(self.n_state + self.n_action*2)*2,
                output_size=1,
                output_activation='sigmoid',
                **joint_likelihood_params,
            )
        else:
            self.joint_likelihood_mdl = joint_likelihood_mdl

        self.joint_likelihood_optimizer = torch.optim.Adam(self.joint_likelihood_mdl.parameters(), lr=self.lr)

    def calc_likelihood_correction(
            self,
            states_n: torch.Tensor,
            actions_1_n: torch.Tensor,
            actions_2_n: torch.Tensor,
            prefs_n: torch.Tensor,
            sigmoid_n: torch.Tensor,
    ):
        # Calc. joint probs
        n = len(states_n)

        actions_n = torch.concat([actions_1_n.unsqueeze(-1), actions_2_n.unsqueeze(-1)], dim=1)
        actions_w_n = actions_n[torch.arange(n), prefs_n]  # Winners
        actions_l_n = actions_n[torch.arange(n), 1 - prefs_n]  # Losers

        states_ns = nn.functional.one_hot(states_n, num_classes=self.n_state).to(torch.float)
        actions_w_na = nn.functional.one_hot(actions_w_n, num_classes=self.n_action).to(torch.float)
        actions_l_na = nn.functional.one_hot(actions_l_n, num_classes=self.n_action).to(torch.float)

        joints_nsaa = torch.concat([states_ns, actions_w_na, actions_l_na], dim=1)

        joint_likelihood_n = self.joint_likelihood_mdl(torch.concat([joints_nsaa, joints_nsaa], dim=1))[:, 0]

        # Calc. correction term directly
        correction_n = torch.clamp(
            (1 - 2*sigmoid_n) * (joint_likelihood_n - sigmoid_n**2)/(4 - 6*sigmoid_n)/sigmoid_n,
            min=-0.1,
            max=0.1,
        )

        return correction_n

    def update_joint_likelihood_mdl(
            self,
            states_n: torch.Tensor,
            actions_1_n: torch.Tensor,
            actions_2_n: torch.Tensor,
            prefs_n: torch.Tensor,
            us_n: torch.Tensor,
    ):
        n = len(states_n)

        actions_n = torch.concat([actions_1_n.unsqueeze(-1), actions_2_n.unsqueeze(-1)], dim=1)
        actions_w_n = actions_n[torch.arange(n), prefs_n]  # Winners
        actions_l_n = actions_n[torch.arange(n), 1 - prefs_n]  # Losers

        # Find similars
        us_n1 = us_n.unsqueeze(-1)
        sim_nn = torch.triu((us_n1 == us_n1.T)*1, diagonal=1)
        filt = torch.any(sim_nn, dim=1)

        # Find first similar per row (if any)
        sim_idx_fn = torch.argmax(sim_nn[filt], dim=1)

        # Extract the first batch
        states_1_fns = nn.functional.one_hot(states_n[filt], num_classes=self.n_state).to(torch.float)
        actions_w_1_fna = nn.functional.one_hot(actions_w_n[filt], num_classes=self.n_action).to(torch.float)
        actions_l_1_fna = nn.functional.one_hot(actions_l_n[filt], num_classes=self.n_action).to(torch.float)

        # Extract the second batch
        states_2_fns = nn.functional.one_hot(states_n[sim_idx_fn], num_classes=self.n_state).to(torch.float)
        actions_w_2_fna = nn.functional.one_hot(actions_w_n[sim_idx_fn], num_classes=self.n_action).to(torch.float)
        actions_l_2_fna = nn.functional.one_hot(actions_l_n[sim_idx_fn], num_classes=self.n_action).to(torch.float)

        # Combine two batches
        joint_inputs = torch.concat(
            [states_1_fns, actions_w_1_fna, actions_l_1_fna, states_2_fns, actions_w_2_fna, actions_l_2_fna],
            dim=1,
        )

        # Calc. likelihood and loss
        joint_likelihood_n = self.joint_likelihood_mdl(joint_inputs)

        loss = -torch.log(joint_likelihood_n).mean()

        # Step the optimizer
        self.joint_likelihood_optimizer.zero_grad()
        loss.backward()
        self.joint_likelihood_optimizer.step()

        return {'joint_likelihood_training_loss': loss.item()}

    def update(
            self,
            states_n: torch.Tensor,
            actions_1_n: torch.Tensor,
            actions_2_n: torch.Tensor,
            prefs_n: torch.Tensor,
            us_n: torch.Tensor,
    ):
        # Calc. likelihood
        sigmoid_n = self.calc_likelihood(states_n, actions_1_n, actions_2_n, prefs_n)

        # Calc. loss
        if self.step >= self.start_correction_after_step:
            # Calc. likelihood correction
            correction_n = self.calc_likelihood_correction(
                states_n, actions_1_n, actions_2_n, prefs_n, sigmoid_n
            ).detach()

            likelihood_n = sigmoid_n + self.var_multiplier*correction_n
        else:
            likelihood_n = sigmoid_n

        loss = -torch.log(likelihood_n).mean()

        # Step the optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step += len(states_n)

        # Update the joint likelihood model
        out_dict = self.update_joint_likelihood_mdl(states_n, actions_1_n, actions_2_n, prefs_n, us_n)

        out_dict['training_loss'] = loss.item()

        return out_dict

    def save(self, save_path: str):
        assert self.policy.__class__.__name__ != self.ref_policy.__class__.__name__

        self.policy.save(save_path)
        self.ref_policy.save(save_path)

        torch.save(
            {
                'kwargs': {
                    'beta': self.beta,
                    'lr': self.lr,
                    'var_multiplier': self.var_multiplier,
                    'step': self.step,
                    'start_correction_after_step': self.start_correction_after_step,
                    'joint_likelihood_params': self.joint_likelihood_params,
                },
                'policies': {
                    'policy_appendix': append_class_to_path('', self.policy.__class__),
                    'ref_policy_appendix': append_class_to_path('', self.ref_policy.__class__),
                },
                'optimizer_state_dict': self.optimizer.state_dict(),
                'joint_likelihood_mdl_state_dict': self.joint_likelihood_mdl.state_dict(),
                'joint_likelihood_optimizer_state_dict': self.joint_likelihood_optimizer.state_dict(),
            },
            self._append_to_path(save_path),
        )

    @classmethod
    def load(cls, load_path: str):
        # Load the solver checkpoint
        solver_load_path = cls._append_to_path(load_path)
        checkpoint = torch.load(solver_load_path)

        # Load the policies
        policy = load_class(load_path + checkpoint['policies']['policy_appendix']).load(load_path)
        ref_policy = load_class(load_path + checkpoint['policies']['ref_policy_appendix']).load(load_path)

        # Load the solver
        solver = cls(policy, ref_policy, **checkpoint['kwargs'])
        solver.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        solver.joint_likelihood_optimizer.load_state_dict(checkpoint['joint_likelihood_optimizer_state_dict'])
        solver.joint_likelihood_mdl.load_state_dict(checkpoint['joint_likelihood_mdl_state_dict'])

        return solver

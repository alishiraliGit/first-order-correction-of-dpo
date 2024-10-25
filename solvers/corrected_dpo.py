import torch

from policies.base_policy import BasePolicy
from solvers.dpo import DPO
from utils.load_and_save_utils import append_class_to_path, load_class


class CorrectedDPO(DPO):
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

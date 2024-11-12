import os
import torch
import numpy as np
from matplotlib import pyplot as plt

from utils.load_and_save_utils import load_class


def sigmoid(x):
    return 1/(1 + np.exp(-x))


if __name__ == '__main__':
    # ===== Settings =====
    opt_shifts = [-1, 1]
    decay_rate = 0.5
    correction_var = 0.

    exp_name = f'offline_size100000_' \
               f'shifts{"_".join(["%g" % s for s in opt_shifts])}_' \
               f'decay{"%g" % decay_rate}_' \
               f'estvarcorrected{"%g" % correction_var}'

    log_dir = os.path.join('..', 'data', exp_name)
    figs_dir = os.path.join('..', 'figs')

    save_figs = False

    # ===== Load =====
    solver = load_class(os.path.join(log_dir, 'solvers')).load(log_dir)

    n_state = solver.policy.n_state
    n_action = solver.policy.n_action

    # ===== Eval. policy =====
    probs_saa = np.zeros((n_state, n_action, n_action))
    probs_sa = np.zeros((n_state, n_action))

    for state in range(n_state):
        for action_1 in range(n_action):
            for action_2 in range(n_action):
                probs_saa[state, action_1, action_2] = solver.joint_likelihood_mdl(
                    torch.tensor([state]), torch.tensor([action_1]), torch.tensor([action_2]),
                    torch.tensor([state]), torch.tensor([action_1]), torch.tensor([action_2]),
                ).detach().numpy()

    for state in range(n_state):
        for action_1 in range(n_action):
            probs_sa[state, action_1] = probs_saa[state, state, action_1]

    # ===== Visualize policy =====
    # Matrix
    plt.figure(figsize=(5, 4))

    plt.imshow(probs_sa)
    plt.colorbar()

    plt.tight_layout()

    # Vector
    plt.figure(figsize=(5, 4))

    aligned_sa = probs_sa.copy()
    for state in range(n_state):
        aligned_sa[state] = np.roll(probs_sa[state], n_state // 2 - state)

    aligned_a = np.mean(aligned_sa, axis=0)

    d = np.arange(-(n_state // 2), -(n_state // 2) + n_state)

    dist_f = lambda a, shift: np.minimum((a - shift) % n_state, (shift - a) % n_state)
    opt_a = np.zeros((n_action,))

    for s in opt_shifts:
        opt_a += 1/len(opt_shifts) * sigmoid(decay_rate**dist_f(d, s) - decay_rate**dist_f(0, s))**2

    plt.plot(d, opt_a, 'k--')
    plt.plot(d, aligned_a, 'b')

    plt.ylabel('Joint likelihood')

    plt.legend(['OPT', 'Est.'])

    plt.tight_layout()

    plt.show()

    if save_figs:
        plt.savefig(os.path.join(figs_dir, f'aligned_joint_likelihood_{exp_name}.pdf'))

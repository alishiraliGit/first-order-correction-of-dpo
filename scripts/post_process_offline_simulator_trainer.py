import os
import torch
import numpy as np
from matplotlib import pyplot as plt

from utils.load_and_save_utils import load_class


if __name__ == '__main__':
    # ===== Settings =====
    opt_shifts = [-2, 0, 2]
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
    solver = load_class(log_dir + '.solvers').load(log_dir)

    n_state = solver.policy.n_state
    n_action = solver.policy.n_action

    # ===== Eval. policy =====
    probs_sa = np.zeros((n_state, n_action))

    for state in range(n_state):
        probs_sa[state] = solver.policy(torch.tensor([state], dtype=torch.long)).probs[0].detach().numpy()

    # ===== Visualize policy =====
    # Matrix
    plt.figure(figsize=(5, 4))

    plt.imshow(probs_sa)
    plt.colorbar()

    plt.tight_layout()

    plt.show()

    if save_figs:
        plt.savefig(os.path.join(figs_dir, f'visualized_policy_{exp_name}.pdf'))

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
        opt_a += np.exp(decay_rate ** dist_f(d, s) / solver.beta)
    opt_a /= opt_a.sum()

    plt.plot(d, opt_a, 'k--')
    plt.plot(d, aligned_a, 'b')

    plt.legend(['OPT', 'DPO'])

    plt.tight_layout()

    plt.show()

    if save_figs:
        plt.savefig(os.path.join(figs_dir, f'aligned_policy_{exp_name}.pdf'))

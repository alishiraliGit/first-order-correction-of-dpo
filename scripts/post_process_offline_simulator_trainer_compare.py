import os
import torch
import numpy as np
from matplotlib import pyplot as plt

from utils.load_and_save_utils import load_class


if __name__ == '__main__':
    # ===== Settings =====
    opt_shifts = [-1, 1]
    decay_rate = 0.5
    correction_vars = [0, 1, 2, 3]

    common_exp_name = \
        f'offline_size100000_' \
        f'shifts{"_".join(["%g" % s for s in opt_shifts])}_' \
        f'decay{"%g" % decay_rate}'

    postfix = 'ceestvar4corrected'

    exp_names = [common_exp_name + '_' + postfix + f'{"%g" % c_var}' for c_var in correction_vars]

    log_dirs = [os.path.join('..', 'data', exp_name) for exp_name in exp_names]
    figs_dir = os.path.join('..', 'figs')

    save_figs = True

    # ===== Load =====
    solvers = [load_class(os.path.join(log_dir, 'solvers')).load(log_dir) for log_dir in log_dirs]

    n_state = solvers[0].policy.n_state
    n_action = solvers[0].policy.n_action

    # ===== Calc. OPT =====
    d = np.arange(-(n_state // 2), -(n_state // 2) + n_state)

    dist_f = lambda a, shift: np.minimum((a - shift) % n_state, (shift - a) % n_state)
    opt_a = np.zeros((n_action,))
    for s in opt_shifts:
        opt_a += np.exp(decay_rate ** dist_f(d, s) / solvers[0].beta)
    opt_a /= opt_a.sum()

    # ===== Eval. policy =====
    plt.figure(figsize=(5, 4))

    plt.plot(d, opt_a, 'k--')

    legends = ['OPT']
    cl = lambda i: [(i + 0.5)/len(solvers), 0, (1 - (i + 0.5)/len(solvers))]

    for i_c, c_var in enumerate(correction_vars):
        probs_sa = np.zeros((n_state, n_action))

        for state in range(n_state):
            probs_sa[state] = solvers[i_c].policy(torch.tensor([state], dtype=torch.long)).probs[0].detach().numpy()

        aligned_sa = probs_sa.copy()
        for state in range(n_state):
            aligned_sa[state] = np.roll(probs_sa[state], n_state // 2 - state)

        aligned_a = np.mean(aligned_sa, axis=0)

        plt.plot(d, aligned_a, color=cl(i_c))

        legends.append(r'DPO($\alpha$=%g)' % c_var)

    plt.legend(legends, loc='lower center')
    plt.ylabel(r'$\pi(y - x)$')
    plt.xlabel(r'$y - x$')

    plt.tight_layout()

    if save_figs:
        plt.savefig(os.path.join(figs_dir, f'aligned_policy_comparison_{common_exp_name}_{postfix}.pdf'))

    plt.show()

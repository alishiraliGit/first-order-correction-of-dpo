import os

from envs.discrete_env import DiscreteMultiShiftedProximityEnv
from preferencemodels.bradley_terry import BradleyTerry
from policies.mlp_policy import DiscreteMLPPolicy
from policies.ordinal_policy import OrdinalPolicy
from policies.ref_policy import UniformPolicy
from solvers.dpo import DPO
from solvers.corrected_dpo import CorrectedDPO, EstVarCorrectedDPO
from solvers.nbc import NBC
from trainers.simulator_trainer import OfflineSimulatorTrainer
from utils.logger import Logger
from utils.pytorch_utils import init_gpu


if __name__ == '__main__':
    # ===== Init. =====
    # GPU
    init_gpu()

    # Env
    env = DiscreteMultiShiftedProximityEnv(
        n_state=20,
        n_action=20,
        shifts=[-5, 0, 5],
        decay_func='linear',
        decay_rates=[0.15, 0.2, 0.15],
        rew_scales=[4, 1.5, 4],
    )

    # Preference
    pref_mdl = BradleyTerry()

    # Solver class
    solver_class = DPO

    # Polices
    if solver_class in (DPO, CorrectedDPO, EstVarCorrectedDPO):
        pi = DiscreteMLPPolicy(
            n_state=env.n_state,
            n_action=env.n_action,
            n_layer=2,
            size=15,
        )

    elif solver_class == NBC:
        pi = OrdinalPolicy(
            n_state=env.n_state,
            n_action=env.n_action,
            use_raw_score=True,
        )

    else:
        raise NotImplementedError

    ref_pi = UniformPolicy(n_action=env.n_action)

    # Solver
    if solver_class == DPO:
        solver = DPO(
            policy=pi,
            ref_policy=ref_pi,
            beta=1.,
            lr=1e-3,
        )

    elif solver_class == CorrectedDPO:
        solver = CorrectedDPO(
            policy=pi,
            ref_policy=ref_pi,
            beta=1.,
            lr=1e-3,
            var_multiplier=1.,
        )

    elif solver_class == EstVarCorrectedDPO:
        solver = EstVarCorrectedDPO(
            policy=pi,
            ref_policy=ref_pi,
            beta=1.,
            lr=1e-3,
            var_multiplier=1.,
            start_correction_after_step=8000,
            joint_likelihood_params={'n_layer': 2, 'size': 15, 'latent_dim': len(set(env.shifts))},
            joint_likelihood_lr=1e-2,
            joint_likelihood_lr_scheduler_gamma=1.,
            joint_likelihood_lr_scheduler_step_size=5000,
            use_general_joint_likelihood_mdl=False,
            correction_method=4,
            loss_fn='ce',
            latent_disc_loss_weight=0.,
        )

    elif solver_class == NBC:
        solver = NBC(
            policy=pi,
            ref_policy=ref_pi,
        )

    else:
        raise NotImplementedError

    # Logger
    exp_name = \
        f'offline_size300000_' \
        f'{env.n_state}s{env.n_action}a_' \
        f'shifts{"_".join(["%g" % s for s in env.shifts])}_' \
        f'decay{env.decay_func_str}{"_".join(["%g" % dr for dr in env.decay_rates])}_' \
        f'rscale{"_".join(["%g" % rs for rs in env.rew_scales])}_' \
        f'{str(solver)}'

    logger = Logger(
        log_dir=os.path.join('..', 'data', exp_name),
        logging_freq=1,
    )

    # Trainer
    trainer = OfflineSimulatorTrainer(
        env=env,
        preference_model=pref_mdl,
        solver=solver,
        logger=logger,
        dataset_size=300000,
        batch_size=1024,
    )

    # ===== Train =====
    if solver_class in (DPO, CorrectedDPO, EstVarCorrectedDPO):
        n_step = 10000
    elif solver_class == NBC:
        n_step = 100000
    else:
        raise NotImplementedError

    trainer.train(steps=n_step)

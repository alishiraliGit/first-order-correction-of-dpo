import os

from envs.discrete_env import DiscreteMultiShiftedProximityEnv
from preferencemodels.bradley_terry import BradleyTerry
from policies.mlp_policy import DiscreteMLPPolicy
from policies.ref_policy import UniformPolicy
from solvers.corrected_dpo import CorrectedDPO, EstVarCorrectedDPO
from trainers.simulator_trainer import OfflineSimulatorTrainer
from utils.logger import Logger
from utils.pytorch_utils import init_gpu


if __name__ == '__main__':
    # ===== Init. =====
    # GPU
    init_gpu()

    # Env
    env = DiscreteMultiShiftedProximityEnv(
        n_state=9,
        n_action=9,
        opt_shifts=[-2, 0, 2],
        decay_rate=0.5,
    )

    # Preference
    pref_mdl = BradleyTerry()

    # Polices
    pi = DiscreteMLPPolicy(
        n_state=env.n_state,
        n_action=env.n_action,
        n_layer=2,
        size=15,
    )

    ref_pi = UniformPolicy(n_action=env.n_action)

    # Solver
    solver = EstVarCorrectedDPO(
        policy=pi,
        ref_policy=ref_pi,
        beta=1.,
        lr=1e-3,
        var_multiplier=0.05,
        start_correction_after_step=10000,
        joint_likelihood_params={'n_layer': 3, 'size': 15, 'latent_dim': len(env.opt_shifts)},
        use_general_joint_likelihood_mdl=False,
    )

    # solver = CorrectedDPO(
    #     policy=pi,
    #     ref_policy=ref_pi,
    #     beta=1.,
    #     lr=1e-3,
    #     var_multiplier=0.2,
    # )

    # Logger
    exp_name = f'offline_size100000_' \
               f'shifts{"_".join(["%g" % s for s in env.opt_shifts])}_' \
               f'decay{"%g" % env.decay_rate}_' \
               f'estvarcorrected{"%g" % solver.var_multiplier}'

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
        dataset_size=100000,
        batch_size=512,
    )

    # ===== Train =====
    trainer.train(steps=15000)

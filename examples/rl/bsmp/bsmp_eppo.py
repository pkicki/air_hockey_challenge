from copy import copy
import os
from time import perf_counter
import matplotlib.pyplot as plt

import torch
import numpy as np
from scipy.interpolate import interp1d

from mushroom_rl.algorithms.policy_search import ePPO
from mushroom_rl.utils.minibatches import minibatch_generator
from differentiable_robot_model.robot_model import DifferentiableRobotModel

from bsmp.utils import equality_loss, limit_loss



class BSMPePPO(ePPO):
    """
    Episodic adaptation of the Proximal Policy Optimization algorithm
    with B-spline actions and differentiable constraints.
    "Proximal Policy Optimization Algorithms".
    Schulman J. et al.. 2017.
    "Fast Kinodynamic Planning on the Constraint Manifold With Deep Neural Networks"
    Kicki P. et al. 2023.
    """

    def __init__(self, mdp_info, distribution, policy, optimizer, #mu_optimizer, sigma_optimizer,
                 robot_constraints, constraint_lr,
                 n_epochs_policy, batch_size, eps_ppo, ent_coeff=0.0, context_builder=None): 
                 #mdp_info, robot_constraints, dt, n_q_cps, n_t_cps, n_pts_fixed_begin, n_pts_fixed_end,
                 #n_dim, sigma_init, sigma_eps, mu_lr, constraint_lr, **kwargs):
        self.robot_constraints = robot_constraints
        self.alphas = np.array([0.] * 18)
        self.violation_limits = np.array([1e-3] * 7 + [1e-4] * 7 + [1e-3] * 4)
        self.constraint_lr = constraint_lr
        self.constraint_losses = []
        self.constraint_losses_log = []

        self.urdf_path = os.path.join(os.path.dirname(__file__), "iiwa_striker.urdf")
        self.load_robot()
        self._epoch_no = 0

        super().__init__(mdp_info, distribution, policy, optimizer, n_epochs_policy,
                         batch_size, eps_ppo, ent_coeff, context_builder)

        self._add_save_attr(
            alphas='numpy',
            violation_limits='numpy',
            constraint_lr='primitive',
            sigma_optimizer='torch',
            mu_approximator='mushroom',
            mu_optimizer='torch',
            robot_constraints='pickle',
            urdf_path='primitive',
            constraint_losses='pickle',
            constraint_losses_log='pickle',
        )

    def episode_start(self, initial_state, episode_info):
        policy_state, theta = super().episode_start(initial_state, episode_info)
        return self.policy.reset(initial_state), theta
        #return self.policy.reset(initial_state), theta[0]

    def load_robot(self):
        self.robot = DifferentiableRobotModel(urdf_path=self.urdf_path, name="iiwa", device="cpu")

    def _unpack_qt(self, qt, trainable=False):
        n_q_pts = self._n_trainable_q_pts if trainable else self._n_q_pts
        q = qt[..., :self._n_dim * n_q_pts]
        t = qt[..., self._n_dim * n_q_pts:]
        return q, t

    def update_alphas(self):
        constraint_losses = torch.mean(torch.cat(self.constraint_losses, dim=0), dim=0)
        alphas_update = self.constraint_lr * np.log(
            (constraint_losses.detach().numpy() + self.violation_limits * 1e-1) / self.violation_limits)
        self.alphas += alphas_update
        self.constraint_losses_log = constraint_losses
        self.constraint_losses = []

    def _update(self, Jep, theta, context):
        # Prepare the constrint limits tensors
        q_dot_limits = torch.Tensor(self.robot_constraints['q_dot'])[None, None]
        q_ddot_limits = torch.Tensor(self.robot_constraints['q_ddot'])[None, None]

        # All constraint losses computation organized in a single function
        def compute_constraint_losses(context):
            #mu = self.distribution.estimate_mu(context)
            mu = self.distribution._mu
            q, q_dot, q_ddot, t, dt, duration = self.policy.compute_trajectory_from_theta(mu, context)

            dt_ = dt[..., None]
            q_dot_loss = limit_loss(torch.abs(q_dot), dt_, q_dot_limits)
            q_ddot_loss = limit_loss(torch.abs(q_ddot), dt_, q_ddot_limits)

            q_ = q.reshape((-1, q.shape[-1]))
            q_ = torch.cat([q_, torch.zeros((q_.shape[0], 9 - q_.shape[1]))], dim=-1)
            ee_pos, ee_quat = self.robot.compute_forward_kinematics(q_, "F_striker_tip")
            ee_pos = ee_pos.reshape((q.shape[0], q.shape[1], 3))
            ee_quat = ee_quat.reshape((q.shape[0], q.shape[1], 4))

            x_ee_loss_low = limit_loss(self.robot_constraints["x_ee_lb"], dt, ee_pos[..., 0])[..., None]
            y_ee_loss_low = limit_loss(self.robot_constraints["y_ee_lb"], dt, ee_pos[..., 1])[..., None]
            y_ee_loss_high = limit_loss(ee_pos[..., 1], dt, self.robot_constraints["y_ee_ub"])[..., None]
            z_ee_loss = equality_loss(ee_pos[..., 2], dt, self.robot_constraints["z_ee"])[..., None]

            constraint_losses = torch.cat([q_dot_loss, q_ddot_loss, x_ee_loss_low, y_ee_loss_low,
                                           y_ee_loss_high, z_ee_loss], dim=-1)
            return constraint_losses

        Jep = torch.tensor(Jep)
        J_mean = torch.mean(Jep)
        J_std = torch.std(Jep)

        Jep = (Jep - J_mean) / (J_std + 1e-8)

        old_dist = self.distribution.log_pdf(theta).detach()

        if self.distribution.is_contextual:
            full_batch = (theta, Jep, old_dist, context)
        else:
            full_batch = (theta, Jep, old_dist)

        for epoch in range(self._n_epochs_policy()):
            for minibatch in minibatch_generator(self._batch_size(), *full_batch):
                theta_i, context_i, Jep_i, old_dist_i = self._unpack(minibatch)

                self._optimizer.zero_grad()
                # ePPO loss
                prob_ratio = torch.exp(self.distribution.log_pdf(theta_i) - old_dist_i)
                clipped_ratio = torch.clamp(prob_ratio, 1 - self._eps_ppo(), 1 + self._eps_ppo.get_value())
                loss = -torch.mean(torch.min(prob_ratio * Jep_i, clipped_ratio * Jep_i))
                loss -= torch.mean(self._ent_coeff() * self.distribution.entropy(context_i))

                # constraint loss
                constraint_losses = compute_constraint_losses(context_i)
                self.constraint_losses.append(constraint_losses)
                constraint_loss = torch.exp(torch.Tensor(self.alphas))[None] * constraint_losses
                constraint_loss = torch.sum(constraint_loss, dim=-1)
                loss += torch.mean(constraint_loss)
                loss.backward()
                self._optimizer.step()
            self.update_alphas()
            mu = self.distribution._mu
            #mu = self.distribution.estimate_mu(context)
            q, q_dot, q_ddot, t, dt, duration = self.policy.compute_trajectory_from_theta(mu, context)
            q_ = q.detach().numpy()[0]
            q_dot_ = q_dot.detach().numpy()[0]
            q_ddot_ = q_ddot.detach().numpy()[0]
            t_ = t.detach().numpy()[0]
            qdl = self.robot_constraints['q_dot']
            qddl = self.robot_constraints['q_ddot']
            q_fk = q.reshape((-1, q.shape[-1]))
            q_fk = torch.cat([q_fk, torch.zeros((q_fk.shape[0], 9 - q_fk.shape[1]))], dim=-1)
            ee_pos, ee_quat = self.robot.compute_forward_kinematics(q_fk, "F_striker_tip")
            ee_pos = ee_pos.reshape((q.shape[0], q.shape[1], 3)).detach().numpy()
            ee_quat = ee_quat.reshape((q.shape[0], q.shape[1], 4)).detach().numpy()

            plt.subplot(121)
            plt.plot(ee_pos[0, :, 0], ee_pos[0, :, 1])
            plt.subplot(122)
            plt.plot(t_, ee_pos[0, :, 2])
            plt.savefig(os.path.join(os.path.dirname(__file__), "..", f"imgs/xyz_{self._epoch_no}.png"))
            plt.clf()

            n_dim = 7
            for i in range(n_dim):
                plt.subplot(3, 7, 1+i)
                plt.plot(t_, q_[:, i])
                plt.subplot(3, 7, 1+i+n_dim)
                plt.plot(t_, q_dot_[:, i])
                plt.plot([t_[0], t_[-1]], [qdl[i], qdl[i]], 'r--')
                plt.plot([t_[0], t_[-1]], [-qdl[i], -qdl[i]], 'r--')
                plt.subplot(3, 7, 1+i+2*n_dim)
                plt.plot(t_, q_ddot_[:, i])
                plt.plot([t_[0], t_[-1]], [qddl[i], qddl[i]], 'r--')
                plt.plot([t_[0], t_[-1]], [-qddl[i], -qddl[i]], 'r--')
            plt.savefig(os.path.join(os.path.dirname(__file__), "..", f"imgs/mean_traj_{self._epoch_no}.png"))
            plt.clf()
            self._epoch_no += 1
        print("SIGMA: ", torch.exp(self.distribution._chol_sigma))
        #print("SIGMA: ", torch.exp(self.distribution._log_sigma))
        #print("SIGMA: ", torch.exp(self.distribution._log_sigma_approximator(context_i[:1])))
        print("ENTROPY: ", torch.mean(self.distribution.entropy(context)))
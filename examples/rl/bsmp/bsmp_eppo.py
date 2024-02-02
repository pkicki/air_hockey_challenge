from copy import copy
import os
from time import perf_counter
#import matplotlib.pyplot as plt

import torch
import numpy as np
from scipy.interpolate import interp1d

from mushroom_rl.algorithms.policy_search import ePPO
from mushroom_rl.utils.minibatches import minibatch_generator
#from differentiable_robot_model.robot_model import DifferentiableRobotModel
from storm_kit.differentiable_robot_model import DifferentiableRobotModel

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

    def __init__(self, mdp_info, distribution, policy, optimizer, value_function, value_function_optimizer, #mu_optimizer, sigma_optimizer,
                 robot_constraints, constraint_lr,
                 n_epochs_policy, batch_size, eps_ppo, target_entropy, entropy_lr, initial_entropy_bonus, ent_coeff=0.0, context_builder=None): 
                 #mdp_info, robot_constraints, dt, n_q_cps, n_t_cps, n_pts_fixed_begin, n_pts_fixed_end,
                 #n_dim, sigma_init, sigma_eps, mu_lr, constraint_lr, **kwargs):
        self.robot_constraints = robot_constraints
        self.alphas = np.array([0.] * 18)
        self.violation_limits = np.array([1e-4] * 7 + [1e-5] * 7 + [5e-6] * 4)
        self.constraint_lr = constraint_lr
        self.constraint_losses = []
        self.constraint_losses_log = []

        self.value_function = value_function
        self.value_function_optimizer = value_function_optimizer

        self.urdf_path = os.path.join(os.path.dirname(__file__), "iiwa_striker.urdf")
        self.load_robot()
        self._epoch_no = 0

        self._q = None
        self._q_dot = None
        self._q_ddot = None
        self._t = None
        self._ee_pos = None

        super().__init__(mdp_info, distribution, policy, optimizer, n_epochs_policy,
                         batch_size, eps_ppo, ent_coeff, context_builder)
        
        self._log_entropy_bonus = torch.tensor(np.log(initial_entropy_bonus), dtype=torch.float32, requires_grad=True)
        self._entropy_optimizer = torch.optim.Adam([self._log_entropy_bonus], lr=entropy_lr)
        self._target_entropy = target_entropy

        self._add_save_attr(
            alphas='numpy',
            violation_limits='numpy',
            constraint_lr='primitive',
            sigma_optimizer='torch',
            mu_approximator='mushroom',
            mu_optimizer='torch',
            value_function='torch',
            value_function_optimizer='torch',
            robot_constraints='pickle',
            urdf_path='primitive',
            constraint_losses='pickle',
            constraint_losses_log='pickle',
            _log_entropy_bonus='torch',
            _entropy_optimizer='torch',
            _target_entropy='primitive',
            _epoch_no='primitive',
        )

    def set_target_entropy(self, target_entropy):
        self._target_entropy = target_entropy
    
    def get_target_entropy(self):
        return self._target_entropy

    def episode_start(self, initial_state, episode_info):
        _, theta = super().episode_start(initial_state, episode_info)
        return self._convert_to_env_backend(self.policy.reset(initial_state)), theta
        #return self.policy.reset(initial_state), theta[0]

    def load_robot(self):
        self.robot = DifferentiableRobotModel(urdf_path=self.urdf_path, name="iiwa")
        #self.robot = DifferentiableRobotModel(urdf_path=self.urdf_path, name="iiwa", device="cpu")

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
        self.alphas = np.clip(self.alphas, -7., None)
        self.constraint_losses_log = constraint_losses
        self.constraint_losses = []

    def update_entropy_bonus(self, log_prob):
        entropy_loss = - (self._log_entropy_bonus.exp() * (log_prob + self._target_entropy)).mean()
        self._entropy_optimizer.zero_grad()
        entropy_loss.backward()
        self._entropy_optimizer.step()

    def compute_forward_kinematics(self, q, q_dot):
        q_ = q.reshape((-1, q.shape[-1]))
        q_ = torch.cat([q_, torch.zeros((q_.shape[0], 9 - q_.shape[1]))], dim=-1)
        q_dot_ = q_dot.reshape((-1, q_dot.shape[-1]))
        q_dot_ = torch.cat([q_dot_, torch.zeros((q_dot_.shape[0], 9 - q_dot_.shape[1]))], dim=-1)
        ee_pos, ee_rot = self.robot.compute_forward_kinematics(q_, q_dot_, "F_striker_tip")
        #ee_pos, ee_quat = self.robot.compute_forward_kinematics(q_, "F_striker_tip")
        ee_pos = ee_pos.reshape((q.shape[0], q.shape[1], 3))
        ee_rot = ee_rot.reshape((q.shape[0], q.shape[1], 3, 3))
        #ee_quat = ee_quat.reshape((q.shape[0], q.shape[1], 4))
        return ee_pos, ee_rot

    # All constraint losses computation organized in a single function
    def compute_constraint_losses(self, theta, context):
        q, q_dot, q_ddot, t, dt, duration = self.policy.compute_trajectory_from_theta(theta, context)

        dt_ = dt[..., None]
        # Prepare the constrint limits tensors
        q_dot_limits = torch.Tensor(self.robot_constraints['q_dot'])[None, None]
        q_ddot_limits = torch.Tensor(self.robot_constraints['q_ddot'])[None, None]

        q_dot_loss = limit_loss(torch.abs(q_dot), dt_, q_dot_limits)
        q_ddot_loss = limit_loss(torch.abs(q_ddot), dt_, q_ddot_limits)

        ee_pos, ee_rot = self.compute_forward_kinematics(q, q_dot)

        x_ee_loss_low = limit_loss(self.robot_constraints["x_ee_lb"], dt, ee_pos[..., 0])[..., None]
        y_ee_loss_low = limit_loss(self.robot_constraints["y_ee_lb"], dt, ee_pos[..., 1])[..., None]
        y_ee_loss_high = limit_loss(ee_pos[..., 1], dt, self.robot_constraints["y_ee_ub"])[..., None]
        z_ee_loss = equality_loss(ee_pos[..., 2], dt, self.robot_constraints["z_ee"])[..., None]
        #print("Z LOSS: ", torch.mean(z_ee_loss, axis=0))
        #plt.plot(ee_pos.detach().numpy()[0, :, 2])
        #plt.show()

        constraint_losses = torch.cat([q_dot_loss, q_ddot_loss, x_ee_loss_low, y_ee_loss_low,
                                        y_ee_loss_high, z_ee_loss], dim=-1)
        return constraint_losses

    def _update(self, Jep, theta, context):
        if len(theta.shape) == 3:
            theta = theta[:, 0]

        Jep = torch.tensor(Jep)
        #J_mean = torch.mean(Jep)
        #J_std = torch.std(Jep)

        #Jep = (Jep - J_mean) / (J_std + 1e-8)

        with torch.no_grad():
            value = self.value_function(context)[:, 0]
            mean_advantage = torch.mean(Jep - value)

        old_dist = self.distribution.log_pdf(theta, context).detach()

        if self.distribution.is_contextual:
            full_batch = (theta, Jep, old_dist, context)
        else:
            full_batch = (theta, Jep, old_dist)
        
        
        #for i in range(theta.shape[0]):
        #    q, q_dot, q_ddot, t, dt, duration = self.policy.compute_trajectory_from_theta(theta[i], context)
        #    q_ = q.reshape((-1, q.shape[-1]))
        #    q_ = torch.cat([q_, torch.zeros((q_.shape[0], 9 - q_.shape[1]))], dim=-1)
        #    ee_pos, ee_quat = self.robot.compute_forward_kinematics(q_, "F_striker_tip")
        #    ee_pos = ee_pos.reshape((q.shape[0], q.shape[1], 3))
        #    plt.subplot(231)
        #    plt.plot(q.detach().numpy()[0, :, 0])
        #    plt.subplot(232)
        #    plt.plot(t.detach().numpy()[0, :])
        #    plt.subplot(233)
        #    plt.plot(dt.detach().numpy()[0, :])
        #    plt.subplot(234)
        #    plt.plot(ee_pos.detach().numpy()[0, :, 0], ee_pos.detach().numpy()[0, :, 1])
        #    plt.subplot(235)
        #    plt.plot(ee_pos.detach().numpy()[0, :, 2])
        #    plt.subplot(236)
        #    plt.plot(t.detach().numpy()[0, :], q_dot.detach().numpy()[0, :, 0])
        #plt.show()

        prob_ratios = []
        clipped_prob_ratios = []
        for epoch in range(self._n_epochs_policy()):
            for minibatch in minibatch_generator(self._batch_size(), *full_batch):
                self._optimizer.zero_grad()
                theta_i, context_i, Jep_i, old_dist_i = self._unpack(minibatch)
                #theta_i, context_i, Jep_i, old_dist_i, value_i = self._unpack(minibatch)

                # ePPO loss
                lp = self.distribution.log_pdf(theta_i, context_i)
                prob_ratio = torch.exp(lp - old_dist_i)
                prob_ratios.append(prob_ratio)
                clipped_ratio = torch.clamp(prob_ratio, 1 - self._eps_ppo(), 1 + self._eps_ppo.get_value())
                clipped_prob_ratios.append(clipped_ratio)
                value_i = self.value_function(context_i)[:, 0]
                A = Jep_i - value_i
                A_unbiased = A - mean_advantage
                loss = -torch.mean(torch.min(prob_ratio * A_unbiased, clipped_ratio * A_unbiased))
                #loss -= torch.mean(self._ent_coeff() * self.distribution.entropy(context_i))
                #loss -= torch.mean(self._log_entropy_bonus.exp() * self.distribution.entropy(context_i))

                # constraint loss
                mu = self.distribution.estimate_mu(context_i)
                #mu = self.distribution._mu
                constraint_losses = self.compute_constraint_losses(mu, context_i)
                self.constraint_losses.append(constraint_losses)
                constraint_loss = torch.exp(torch.Tensor(self.alphas))[None] * constraint_losses
                constraint_loss = torch.sum(constraint_loss, dim=-1)
                loss += torch.mean(constraint_loss)

                value_loss = torch.mean(A**2)
                #print("VALUE LOSS: ", value_loss)
                #print("J: ", Jep_i)
                #print("V: ", value_i)
                loss.backward(retain_graph=True)
                self._optimizer.step()
                self.value_function_optimizer.zero_grad()
                value_loss.backward()
                self.value_function_optimizer.step()
            self.update_alphas()
            #self.update_entropy_bonus(self.distribution.log_pdf(theta, context))
            self._epoch_no += 1
            #mu = self.distribution._mu
        with torch.no_grad():
            mu = self.distribution.estimate_mu(context)
            q, q_dot, q_ddot, t, dt, duration = self.policy.compute_trajectory_from_theta(mu, context)
            q_ = q.detach().numpy()[0]
            q_dot_ = q_dot.detach().numpy()[0]
            q_ddot_ = q_ddot.detach().numpy()[0]
            t_ = t.detach().numpy()[0]
            qdl = self.robot_constraints['q_dot']
            qddl = self.robot_constraints['q_ddot']
            ee_pos, ee_rot = self.compute_forward_kinematics(q, q_dot)
            ee_pos = ee_pos.detach().numpy()
            ee_rot = ee_rot.detach().numpy()

            self._q = q_
            self._q_dot = q_dot_
            self._q_ddot = q_ddot_
            self._t = t_
            self._ee_pos = ee_pos

            #plt.subplot(121)
            #plt.plot(ee_pos[0, :, 0], ee_pos[0, :, 1])
            #plt.plot(context[0, 0], context[0, 1], 'ro')
            #plt.subplot(122)
            #plt.plot(t_, ee_pos[0, :, 2])
            #plt.savefig(os.path.join(os.path.dirname(__file__), "..", f"imgs/xyz_{self._epoch_no}.png"))
            #plt.clf()

            #n_dim = 7
            #for i in range(n_dim):
            #    plt.subplot(3, 7, 1+i)
            #    plt.plot(t_, q_[:, i])
            #    plt.subplot(3, 7, 1+i+n_dim)
            #    plt.plot(t_, q_dot_[:, i])
            #    plt.plot([t_[0], t_[-1]], [qdl[i], qdl[i]], 'r--')
            #    plt.plot([t_[0], t_[-1]], [-qdl[i], -qdl[i]], 'r--')
            #    plt.subplot(3, 7, 1+i+2*n_dim)
            #    plt.plot(t_, q_ddot_[:, i])
            #    plt.plot([t_[0], t_[-1]], [qddl[i], qddl[i]], 'r--')
            #    plt.plot([t_[0], t_[-1]], [-qddl[i], -qddl[i]], 'r--')
            #plt.savefig(os.path.join(os.path.dirname(__file__), "..", f"imgs/mean_traj_{self._epoch_no}.png"))
            #plt.clf()

            #prob_ratios = torch.stack(prob_ratios, dim=0)
            #prob_ratios = prob_ratios.sort(dim=1)[0]
            #prob_ratios = prob_ratios.detach().numpy()
            #for i in range(prob_ratios.shape[1]):
            #    plt.plot(prob_ratios[:, i])
            #plt.show()
            #clipped_prob_ratios = torch.stack(clipped_prob_ratios, dim=0)
            #clipped_prob_ratios = clipped_prob_ratios.sort(dim=1)[0]
            #clipped_prob_ratios = clipped_prob_ratios.detach().numpy()
            #for i in range(clipped_prob_ratios.shape[1]):
            #    plt.plot(clipped_prob_ratios[:, i])
            #plt.show()

            #print("SIGMA: ", self.distribution._chol_sigma)
            #print("SIGMA: ", torch.exp(self.distribution._log_sigma))
            #print("SIGMA: ", torch.exp(self.distribution._log_sigma_approximator(context_i[:1])))
            #print("ENTROPY: ", torch.mean(self.distribution.entropy(context)))
            #print("VALUE NETWORK SCALE: ", torch.exp(self.value_function.log_scale))
            #print("VALUE NETWORK BIAS: ", self.value_function.bias)
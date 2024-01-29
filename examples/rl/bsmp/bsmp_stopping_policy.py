import os
import numpy as np
import torch
from air_hockey_challenge.utils.kinematics import forward_kinematics, jacobian
from baseline.baseline_agent.optimizer import TrajectoryOptimizer
from examples.rl.bsmp.bsmp_policy import BSMPPolicy
from examples.rl.bsmp.bspline import BSpline
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from mushroom_rl.policy import Policy

from examples.rl.bsmp.utils import unpack_data_airhockey


class BSMPStoppingPolicy(BSMPPolicy):
    def __init__(self, env_info, dt, n_q_pts, n_dim, n_t_pts, n_pts_fixed_begin=1, n_pts_fixed_end=1, robot_constraints=None):
        super().__init__(env_info, dt, n_q_pts, n_dim, n_t_pts, n_pts_fixed_begin, n_pts_fixed_end, robot_constraints)
        self._n_trainable_q_stop_pts = self._n_q_pts - 6
        self._n_trainable_t_stop_pts = self._n_t_pts
        self._mallet_radius = env_info['mallet']['radius']
        self._puck_radius = env_info['puck']['radius']
        self._add_save_attr(
            _n_trainable_q_stop_pts='primitive',
            _n_trainable_t_stop_pts='primitive',
            _mallet_radius='primitive',
            _puck_radius='primitive',
        )

    def compute_trajectory_from_theta(self, theta, context):
        q_0, q_d, q_dot_0, q_dot_d, q_ddot_0, q_ddot_d, puck = self.unpack_context(context)
        trainable_q_hit_cps, trainable_t_hit_cps, trainable_q_stop_cps, trainable_t_stop_cps, xy_stop = self.extract_qt(theta)

        middle_trainable_q_hit_pts = torch.tanh(trainable_q_hit_cps[:, :-3]/10.) * np.pi
        middle_trainable_q_stop_pts = torch.tanh(trainable_q_stop_cps / 10.) * np.pi
        trainable_q_d = torch.tanh(trainable_q_hit_cps[:, -1:]/10.) * np.pi
        trainable_q_ddot_d = torch.tanh(trainable_q_hit_cps[:, -3:-2]) * torch.tensor(self.joint_acc_limit)
        trainable_delta_angle = torch.tanh(trainable_q_hit_cps[:, -2:-1, -1]/10.) * np.pi/2.
        #trainable_scale = torch.sigmoid(trainable_q_hit_cps[:, -2, -2])[:, None, None]
        delta_xy_stop = torch.tanh(xy_stop) * 0.4


        #x_cur = forward_kinematics(self.env_info['robot']['robot_model'], self.env_info['robot']['robot_data'], q_0[0, 0])[0]

        puck_pos = puck.detach().numpy()
        goal = np.array([2.484, 0., 0.])
        # Compute the vector that shoot the puck directly to the goal
        vec_puck_goal = (goal - puck_pos) / np.linalg.norm(goal - puck_pos)
        r1 = torch.cat([torch.cos(trainable_delta_angle), -torch.sin(trainable_delta_angle), torch.zeros_like(trainable_delta_angle)], axis=-1)
        r2 = torch.cat([torch.sin(trainable_delta_angle), torch.cos(trainable_delta_angle), torch.zeros_like(trainable_delta_angle)], axis=-1)
        r3 = torch.cat([torch.zeros_like(trainable_delta_angle), torch.zeros_like(trainable_delta_angle), torch.ones_like(trainable_delta_angle)], axis=-1)
        R = torch.stack([r1, r2, r3], axis=-2)
        v_des = (R @ torch.tensor(vec_puck_goal)[..., None])[..., 0]

        x_des = puck_pos - (self._mallet_radius + self._puck_radius) * v_des.detach().numpy()
        #x_des = puck_pos# - (self.env_info['mallet']['radius'] + self.env_info['puck']['radius'] - 0.01) * v_des
        x_des[:, -1] = self.desired_ee_z# - 0.03# - self.env_info['robot']['universal_height']

        x_stop = x_des.copy()
        x_stop[:, :2] = x_stop[:, :2] + delta_xy_stop.detach().numpy()

        q_d_s = []
        for k in range(q_0.shape[0]):
            success, q_d = self.optimizer.solve_hit_config(x_des[k], v_des.detach().numpy()[k], q_0.detach().numpy()[k, 0])
            q_d_s.append(q_d)
        q_d_bias = torch.tensor(q_d_s)[:, None]
        q_d = trainable_q_d + q_d_bias
        q_dot_d_s = []
        for k in range(q_0.shape[0]):
            q_dot_d = (torch.linalg.pinv(torch.tensor(self.optimizer.jacobian(q_d.detach().numpy()[k, 0])))[:, :3] @ v_des.T)[..., 0]
            q_dot_d_s.append(q_dot_d)
        q_dot_d_bias = torch.stack(q_dot_d_s, dim=0)[:, None]
        scale = 1. / torch.max(torch.abs(q_dot_d_bias) / torch.tensor(self.joint_vel_limit), axis=-1, keepdim=True)[0]
        q_dot_d = q_dot_d_bias * scale# * trainable_scale
        #q_d = trainable_q_cps[:, -1:] + q_d_bias
        #q_dot_d = trainable_q_cps[:, -2:-1] + q_dot_d_bias

        # hax
        #q_d = trainable_q_cps[:, -1:] + q_0
        #q_dot_d = trainable_q_cps[:, -2:-1]
        #q_ddot_d = trainable_q_cps[:, -3:-2]
        #trainable_q_cps = trainable_q_cps[:, :-3]
        q_ddot_d = trainable_q_ddot_d
        q1, q2, qm2, qm1 = self.compute_boundary_control_points_exp(trainable_t_hit_cps, q_0, q_dot_0, q_ddot_0,
                                                                    q_d, q_dot_d, q_ddot_d)
        q_begin = [q_0, q1, q2]
        q_end = [q_d, qm1, qm2]
        #q_cps = torch.cat(q_begin[:self._n_pts_fixed_begin] + [q_0 + torch.pi * trainable_q_cps] + q_end[::-1], axis=-2)

        #s = torch.linspace(0., 1., middle_trainable_q_pts.shape[1]+2)[None, 1:-1, None]
        #q_b = qm2 * (1 - s) + qm1 * s
        s = torch.linspace(0., 1., middle_trainable_q_hit_pts.shape[1]+6)[None, 3:-3, None]
        q_b = q_0 * (1 - s) + q_d * s
        q_cps = torch.cat(q_begin[:self._n_pts_fixed_begin] + [q_b + middle_trainable_q_hit_pts] + q_end[::-1], axis=-2)

        q, q_dot, q_ddot, t, dt, duration = self.compute_trajectory(q_cps.to(torch.float32), trainable_t_hit_cps.to(torch.float32), differentiable=True)
        #q_dot_scale = (torch.abs(q_dot) / torch.tensor(self.joint_vel_limit))
        #q_ddot_scale = (torch.abs(q_ddot) / torch.tensor(self.joint_acc_limit))
        #q_dot_scale_max = torch.amax(q_dot_scale, (-2, -1), keepdim=True)
        #q_ddot_scale_max = torch.amax(q_ddot_scale, (-2, -1), keepdim=True)
        #scale_max = torch.maximum(q_dot_scale_max, q_ddot_scale_max**(1./2))
        #trainable_t_cps -= torch.log(scale_max)
        #q, q_dot, q_ddot, t, dt, duration = self.compute_trajectory(q_cps.to(torch.float32), trainable_t_cps.to(torch.float32), differentiable=True)

        #q_ = q.detach().numpy()[0]
        #q_dot_ = q_dot.detach().numpy()[0]
        #q_ddot_ = q_ddot.detach().numpy()[0]
        #t_ = t.detach().numpy()[0]
        #qdl = self.joint_vel_limit
        #qddl = self.joint_acc_limit
        #for i in range(self.n_dim):
        #    plt.subplot(3, 7, 1+i)
        #    plt.plot(t_, q_[:, i])
        #    plt.subplot(3, 7, 1+i+self.n_dim)
        #    plt.plot(t_, q_dot_[:, i])
        #    plt.plot([t_[0], t_[-1]], [qdl[i], qdl[i]], 'r--')
        #    plt.plot([t_[0], t_[-1]], [-qdl[i], -qdl[i]], 'r--')
        #    plt.subplot(3, 7, 1+i+2*self.n_dim)
        #    plt.plot(t_, q_ddot_[:, i])
        #    plt.plot([t_[0], t_[-1]], [qddl[i], qddl[i]], 'r--')
        #    plt.plot([t_[0], t_[-1]], [-qddl[i], -qddl[i]], 'r--')
        #plt.show()

        q_stop = []
        for k in range(q_d.shape[0]):
            success, q_stop_ = self.optimizer.inverse_kinematics(x_stop[k], q_d.detach().numpy()[k, 0])
            q_stop.append(q_stop_)
        q_stop = torch.tensor(q_stop)[:, None]
        z = torch.zeros_like(q_0) if isinstance(q_0, torch.Tensor) else np.zeros_like(q_0)
        q1, q2, qm2, qm1 = self.compute_boundary_control_points_exp(trainable_t_stop_cps, q_d, q_dot_d, q_ddot_d,
                                                                    q_stop, z, z)
        q_begin = [q_d, q1, q2]
        q_end = [q_stop, qm1, qm2]
        s = torch.linspace(0., 1., middle_trainable_q_hit_pts.shape[1]+6)[None, 3:-3, None]
        q_b = q_d * (1 - s) + q_stop * s
        q_stop_cps = torch.cat(q_begin[:self._n_pts_fixed_begin] + [q_b + middle_trainable_q_stop_pts] + q_end[::-1], axis=-2)

        q_stop, q_stop_dot, q_stop_ddot, t_stop, dt_stop, duration_stop = self.compute_trajectory(q_stop_cps.to(torch.float32), trainable_t_stop_cps.to(torch.float32), differentiable=True)

        #q_ = q_stop.detach().numpy()[0]
        #q_dot_ = q_stop_dot.detach().numpy()[0]
        #q_ddot_ = q_stop_ddot.detach().numpy()[0]
        #t_ = t_stop.detach().numpy()[0] + t_[-1]
        #qdl = self.joint_vel_limit
        #qddl = self.joint_acc_limit
        #for i in range(self.n_dim):
        #    plt.subplot(3, 7, 1+i)
        #    plt.plot(t_, q_[:, i])
        #    plt.subplot(3, 7, 1+i+self.n_dim)
        #    plt.plot(t_, q_dot_[:, i])
        #    plt.plot([t_[0], t_[-1]], [qdl[i], qdl[i]], 'r--')
        #    plt.plot([t_[0], t_[-1]], [-qdl[i], -qdl[i]], 'r--')
        #    plt.subplot(3, 7, 1+i+2*self.n_dim)
        #    plt.plot(t_, q_ddot_[:, i])
        #    plt.plot([t_[0], t_[-1]], [qddl[i], qddl[i]], 'r--')
        #    plt.plot([t_[0], t_[-1]], [-qddl[i], -qddl[i]], 'r--')
        #plt.show()

        #xyz = []
        #for k in range(q.shape[1]):
        #    xyz_ = self.optimizer.forward_kinematics(q.detach().numpy()[0, k])
        #    xyz.append(xyz_)
        #xyz = np.array(xyz)
        #xyz_stop = []
        #for k in range(q_stop.shape[1]):
        #    xyz_ = self.optimizer.forward_kinematics(q_stop.detach().numpy()[0, k])
        #    xyz_stop.append(xyz_)
        #xyz_stop = np.array(xyz_stop)

        q = torch.cat([q, q_stop[:, 1:]], axis=-2)
        q_dot = torch.cat([q_dot, q_stop_dot[:, 1:]], axis=-2)
        q_ddot = torch.cat([q_ddot, q_stop_ddot[:, 1:]], axis=-2)
        t = torch.cat([t, t_stop[:, 1:] + t[:, -1:]], axis=-1)
        dt = torch.cat([dt, dt_stop[:, 1:]], axis=-1)
        duration = duration + duration_stop

        self._traj_no += 1
        return q, q_dot, q_ddot, t, dt, duration

    def extract_qt(self, x):
        n_q_hit_cps = self._n_trainable_q_pts * self.n_dim
        n_q_stop_cps = self._n_trainable_q_stop_pts * self.n_dim
        n_t_hit_cps = self._n_trainable_t_pts
        n_t_stop_cps = self._n_trainable_t_stop_pts
        q_hit_cps = x[:, :n_q_hit_cps]
        q_stop_cps = x[:, n_q_hit_cps:n_q_hit_cps + n_q_stop_cps] 
        t_hit_cps = x[:, n_q_hit_cps + n_q_stop_cps:n_q_hit_cps + n_q_stop_cps + n_t_hit_cps]
        t_stop_cps = x[:, n_q_hit_cps + n_q_stop_cps + n_t_hit_cps:n_q_hit_cps + n_q_stop_cps + n_t_hit_cps + n_t_stop_cps]
        xy_stop = x[:, n_q_hit_cps + n_q_stop_cps + n_t_hit_cps + n_t_stop_cps:]
        q_hit_cps = q_hit_cps.reshape(-1, self._n_trainable_q_pts, self.n_dim)
        t_hit_cps = t_hit_cps.reshape(-1, self._n_trainable_t_pts, 1)
        q_stop_cps = q_stop_cps.reshape(-1, self._n_trainable_q_stop_pts, self.n_dim)
        t_stop_cps = t_stop_cps.reshape(-1, self._n_trainable_t_stop_pts, 1)
        return q_hit_cps, t_hit_cps, q_stop_cps, t_stop_cps, xy_stop
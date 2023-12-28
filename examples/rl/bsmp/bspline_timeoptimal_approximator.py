import torch
from bsmp.utils import unpack_data_airhockey, unpack_data_ndof, unpack_data_obstacles2D
from examples.rl.bsmp.bspline_approximator import BSplineApproximator


class BSplineFastApproximatorNDoF(BSplineApproximator):
    def __init__(self, input_shape, output_shape, q_bsp, t_bsp, n_dim, input_space, q_dot_limit, q_ddot_limit,
                 n_pts_fixed_begin=1, n_pts_fixed_end=1, **kwargs):
        super(BSplineFastApproximatorNDoF, self).__init__(input_shape, output_shape, q_bsp, t_bsp, n_dim,
                                                          n_pts_fixed_begin, n_pts_fixed_end)

        self.input_space = input_space
        self.q_dot_limit = torch.tensor(q_dot_limit if len(q_dot_limit.shape) == 2 else q_dot_limit[None], dtype=torch.float32)
        self.q_ddot_limit = torch.tensor(q_ddot_limit if len(q_ddot_limit.shape) == 2 else q_ddot_limit[None], dtype=torch.float32)

        output_q_dim = self.n_q_bsp_control_points * self.n_dim
        output_t_dim = self.n_t_bsp_control_points

        activation = torch.nn.Tanh()
        W = 2048
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, W), activation,
            torch.nn.Linear(W, W), activation,
            torch.nn.Linear(W, W), activation,
        )

        self.q_est = torch.nn.Sequential(
            torch.nn.Linear(W, W), activation,
            torch.nn.Linear(W, output_q_dim), activation,
        )

        self.t_est = torch.nn.Sequential(
            torch.nn.Linear(W, output_t_dim),
        )

    def normalize_input(self, x):
        low = torch.Tensor(self.input_space.low)
        high = torch.Tensor(self.input_space.high)
        normalized = (x - low) / (high - low)
        normalized = 2 * normalized - 1
        normalized[:, 0] = (x[:, 0] - 1.51) / (1.948 / 2. - 0.03165)
        normalized[:, 1] = x[:, 1] / (1.038 / 2. - 0.03165)
        return normalized

    def prepare_data(self, x):
        q0, qd, dq0, dqd, ddq0, ddqd = unpack_data_ndof(x)
        x = self.normalize_input(x)
        return x, q0, qd, dq0, dqd, ddq0, ddqd

    def __call__(self, x):
        x, q0, qd, dq0, dqd, ddq0, ddqd = self.prepare_data(x)

        x = self.fc(x)
        q_est = self.q_est(x)
        q = torch.pi * torch.reshape(q_est, (-1, self.n_q_bsp_control_points, self.n_dim))
        log_ds_dt_prototype = self.t_est(x)
        ds_dt_prototype = torch.exp(log_ds_dt_prototype)
        ds_dt_mul = ds_dt_prototype[:, self.n_pts_fixed_begin:self.n_t_bsp_control_points-self.n_pts_fixed_end]

        #if self.n_pts_fixed_end == 0:
        #    delta_q = torch.abs(q[:, -1])
        #else:
        #    delta_q = torch.abs(qd - q0)

        #estimated_trajectory_duration = torch.max(delta_q / self.q_dot_limit, dim=-1)[0]
        #estimated_trajectory_duration = torch.ones_like(estimated_trajectory_duration)
        #average_ds_dt = 1. / estimated_trajectory_duration 
        #fake_ds_dt = (average_ds_dt[:, None]).repeat((1, 2))
        # TODO: think if the ds_dt_prototype should be somehowe ferreferd to the delta_q
        q1, q2, qm2, qm1 = self.compute_boundary_control_points(ds_dt_prototype, q0, dq0, ddq0, qd, dqd, ddqd)

        s = torch.linspace(0., 1., q.shape[1] + 2)[None, 1:-1, None].to(q.device)


        q0 = q0[:, None]
        q1 = q1[:, None]
        q2 = q2[:, None]
        qm2 = qm2[:, None]
        qm1 = qm1[:, None]
        qd = qd[:, None]

        q_begin = [q0]
        if self.n_pts_fixed_begin > 1:
            q_begin.append(q1)
        if self.n_pts_fixed_begin > 2:
            q_begin.append(q2)
        q_end = []
        if self.n_pts_fixed_end > 0:
            q_end = [qd]
        if self.n_pts_fixed_end > 1:
            q_end.append(qm1)
        if self.n_pts_fixed_end > 2:
            q_end.append(qm2)

        # todo check which one works better
        qb = q0
        if q_begin and q_end:
            qb = q_begin[0] * (1 - s) + q_end[0] * s

        q_cps = torch.cat(q_begin + [q + qb] + q_end[::-1], axis=-2)

        #from time import perf_counter
        #t0 = perf_counter()
        q_dot = torch.tensor(self.q_bsp.dN, dtype=torch.float32) @ q_cps
        q_ddot = torch.tensor(self.q_bsp.ddN, dtype=torch.float32) @ q_cps

        ds_dt_vel = 1. / (torch.max(torch.abs(q_dot / self.q_dot_limit), dim=-1)[0] + 1e-5)
        ds_dt_acc = (1. / (torch.max(torch.abs(q_ddot / self.q_ddot_limit), dim=-1)[0])**(1./2) + 1e-5)
        ds_dt_base = torch.minimum(ds_dt_vel, ds_dt_acc)

        #from torchaudio.functional import lowpass_biquad

        #a = torch.rand((1000,)) - 0.5
        #b = lowpass_biquad(a, ds_dt_base.shape[-1], 100.) 
        #c = lowpass_biquad(a, ds_dt_base.shape[-1], 10.) 
        #d = lowpass_biquad(a, ds_dt_base.shape[-1], 1.) 

        #import matplotlib.pyplot as plt
        #plt.plot(a.detach().numpy())
        #plt.plot(b.detach().numpy())   
        #plt.plot(c.detach().numpy())   
        #plt.plot(d.detach().numpy())   
        #plt.legend(["a", "b", "c", "d"])
        #plt.show()

        #scale = torch.max(ds_dt_base)
        #ds_dt_base_ = lowpass_biquad(ds_dt_base / scale, ds_dt_base.shape[-1], 1.) * scale
        #ds_dt_base_ = lowpass_biquad(ds_dt_base / scale, 100, 1.) * scale

        Nt = torch.tensor(self.t_bsp.N[..., self.n_pts_fixed_begin:self.t_bsp.N.shape[-1]-self.n_pts_fixed_end], dtype=torch.float32)
        ds_dt_cps_base = (torch.linalg.pinv(Nt) @ ds_dt_base[..., None])[..., 0]
        #ds_dt_cps_base = torch.maximum(ds_dt_cps_base, torch.ones_like(ds_dt_cps_base) * 1e-3)

        #ds_dt_ = (Nt @ ds_dt_cps_base[..., None])[..., 0]
        #plt.subplot(231)
        #plt.plot(q_dot[0].detach().numpy())
        #plt.subplot(232)
        #plt.plot(q_ddot[0].detach().numpy())
        #plt.subplot(233)
        #plt.plot(ds_dt_base[0].detach().numpy())
        #plt.plot(ds_dt_base_[0].detach().numpy(), 'tab:green')
        #plt.plot(ds_dt_[0].detach().numpy(), 'r--')
        #plt.subplot(234)
        #plt.plot(ds_dt_cps_base[0].detach().numpy())
        #plt.show()

        ds_dt_cps = ds_dt_cps_base * ds_dt_mul
        
        if self.n_pts_fixed_begin > 0:
            ds_dt_cps = torch.cat([ds_dt_prototype[:, :self.n_pts_fixed_begin], ds_dt_cps], axis=-1)
        if self.n_pts_fixed_end > 0:
            ds_dt_cps = torch.cat([ds_dt_cps, ds_dt_prototype[:, ds_dt_prototype.shape[1] - self.n_pts_fixed_end:]], axis=-1)

        #log_ds_dt_cps = torch.log(ds_dt_cps)
        #t1 = perf_counter()
        #print("Time: ", t1 - t0)
        #assert False

        #qN = torch.tensor(self.q_bsp.N, dtype=torch.float32)
        #qdN = torch.tensor(self.q_bsp.dN, dtype=torch.float32)
        #qddN = torch.tensor(self.q_bsp.ddN, dtype=torch.float32)

        #tN = torch.tensor(self.t_bsp.N, dtype=torch.float32)
        #tdN = torch.tensor(self.t_bsp.dN, dtype=torch.float32)

        #q = qN @ q_cps
        #q_dot_s = qdN @ q_cps
        #q_ddot_s = qddN @ q_cps

        #ds_dt = tN @ ds_dt_cps[..., None]
        #dds_dtt = tdN @ ds_dt_cps[..., None]

        ## todo check if the indexing is general
        #dt = 1. / ds_dt[..., 0] / ds_dt.shape[-2]
        #t = torch.cumsum(dt, dim=-1) - dt[..., :1]
        #duration = t[:, -1]

        #q_dot = q_dot_s * ds_dt
        #q_ddot = q_ddot_s * ds_dt ** 2 + dds_dtt * q_dot_s * ds_dt

        #print("XFFF")
        #t_ = t[0].detach().numpy()
        #for i in range(self.n_dim):
        #    plt.subplot(3, self.n_dim, i+1)
        #    plt.plot(t_, q[0, :, i].detach().numpy())
        #    plt.subplot(3, self.n_dim, i+1 + self.n_dim)
        #    qdl = self.q_dot_limit[0, i].detach().numpy()
        #    plt.plot([t_[0], t_[-1]], [qdl, qdl], 'r--')
        #    plt.plot([t_[0], t_[-1]], [-qdl, -qdl], 'r--')
        #    plt.plot(t_, q_dot[0, :, i].detach().numpy())
        #    plt.subplot(3, self.n_dim, i+1 + 2*self.n_dim)
        #    qddl = self.q_ddot_limit[0, i].detach().numpy()
        #    plt.plot([t_[0], t_[-1]], [qddl, qddl], 'r--')
        #    plt.plot([t_[0], t_[-1]], [-qddl, -qddl], 'r--')
        #    plt.plot(t_, q_ddot[0, :, i].detach().numpy())
        #plt.show()
    
        #return q_cps, log_ds_dt_cps
        #return q_cps, ds_dt_prototype
        return q_cps, ds_dt_cps


class BSplineFastApproximatorAirHockey(BSplineFastApproximatorNDoF):
    def prepare_data(self, x):
        puck, puck_dot, q0, qd, dq0, dqd, ddq0, ddqd, _ = unpack_data_airhockey(x)
        x = self.normalize_input(x)
        return x, q0, qd, dq0, dqd, ddq0, ddqd


class BSplineFastApproximatorAirHockeyWrapper(BSplineFastApproximatorAirHockey):
    def __init__(self, input_shape, output_shape, params, **kwargs):
        super(BSplineFastApproximatorAirHockeyWrapper, self).__init__(input_shape, output_shape, params["q_bsp"], params["t_bsp"], params["n_dim"],
                                                           params["input_space"], params["q_dot_limit"], params["q_ddot_limit"],
                                                           params["n_pts_fixed_begin"], params["n_pts_fixed_end"])
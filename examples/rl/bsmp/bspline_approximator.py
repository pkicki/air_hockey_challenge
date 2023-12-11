import torch
from bsmp.utils import unpack_data_airhockey, unpack_data_ndof, unpack_data_obstacles2D


class BSplineApproximator(torch.nn.Module):
    def __init__(self, input_shape, output_shape, q_bsp, t_bsp, n_dim, n_pts_fixed_begin=1, n_pts_fixed_end=1):
        super(BSplineApproximator, self).__init__()
        self.n_pts_fixed_begin = n_pts_fixed_begin
        self.n_pts_fixed_end = n_pts_fixed_end
        self.n_q_bsp_control_points = q_bsp.N.shape[-1] - (self.n_pts_fixed_begin + self.n_pts_fixed_end)
        self.n_t_bsp_control_points = t_bsp.N.shape[-1]
        self.n_dim = n_dim
        self.input_dim = input_shape[0]

        self.q_bsp = q_bsp
        self.t_bsp = t_bsp
        self.qdd1 = self.q_bsp.ddN[0, 0, 0]
        self.qdd2 = self.q_bsp.ddN[0, 0, 1]
        self.qdd3 = self.q_bsp.ddN[0, 0, 2]
        self.qd1 = self.q_bsp.dN[0, 0, 1]
        self.td1 = self.t_bsp.dN[0, 0, 1]

    def prepare_data(self, x):
        raise NotImplementedError()

    def compute_boundary_control_points(self, dtau_dt, q0, q_dot_0, q_ddot_0, qd, q_dot_d, q_ddot_d):
        q1 = q_dot_0 / dtau_dt[:, :1] / self.qd1 + q0
        qm1 = qd - q_dot_d / dtau_dt[:, -1:] / self.qd1
        q2 = ((q_ddot_0 / dtau_dt[:, :1] -
               self.qd1 * self.td1 * (q1 - q0) * (dtau_dt[:, 1] - dtau_dt[:, 0])[:, None]) / dtau_dt[:, :1]
              - self.qdd1 * q0 - self.qdd2 * q1) / self.qdd3
        qm2 = ((q_ddot_d / dtau_dt[:, -1:] -
                self.qd1 * self.td1 * (qd - qm1) * (dtau_dt[:, -1] - dtau_dt[:, -2])[:, None]) / dtau_dt[:, -1:]
               - self.qdd1 * qd - self.qdd2 * qm1) / self.qdd3
        return q1, q2, qm2, qm1

    def __call__(self, x):
        raise NotImplementedError()


class BSplineApproximatorNDoF(BSplineApproximator):
    def __init__(self, input_shape, output_shape, q_bsp, t_bsp, n_dim, input_space, n_pts_fixed_begin=1, n_pts_fixed_end=1, **kwargs):
        super(BSplineApproximatorNDoF, self).__init__(input_shape, output_shape, q_bsp, t_bsp, n_dim,
                                                      n_pts_fixed_begin, n_pts_fixed_end)

        self.input_space = input_space

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
        return (x - low) / (high - low)

    def prepare_data(self, x):
        q0, qd, dq0, dqd, ddq0, ddqd = unpack_data_ndof(x)
        x = self.normalize_input(x)
        return x, q0, qd, dq0, dqd, ddq0, ddqd

    def __call__(self, x):
        x, q0, qd, dq0, dqd, ddq0, ddqd = self.prepare_data(x)

        x = self.fc(x)
        q_est = self.q_est(x)
        log_dtau_dt = self.t_est(x)
        #dtau_dt = torch.exp(dtau_dt)

        # todo think about the expected time scaling
        log_dtau_dt = log_dtau_dt# / expected_time[:, None]

        q = torch.reshape(q_est, (-1, self.n_q_bsp_control_points, self.n_dim))
        s = torch.linspace(0., 1., q.shape[1] + 2)[None, 1:-1, None].to(q.device)

        q1, q2, qm2, qm1 = self.compute_boundary_control_points(torch.exp(log_dtau_dt), q0, dq0, ddq0, qd, dqd, ddqd)

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

        x = torch.cat(q_begin + [q + qb] + q_end[::-1], axis=-2)

        return x, log_dtau_dt


class BSplineApproximatorAirHockey(BSplineApproximatorNDoF):
    def prepare_data(self, x):
        puck, puck_dot, q0, qd, dq0, dqd, ddq0, ddqd = unpack_data_airhockey(x)
        x = self.normalize_input(x)
        return x, q0, qd, dq0, dqd, ddq0, ddqd


class BSplineApproximatorAirHockeyWrapper(BSplineApproximatorAirHockey):
    def __init__(self, input_shape, output_shape, params, **kwargs):
        super(BSplineApproximatorAirHockey, self).__init__(input_shape, output_shape, params["q_bsp"], params["t_bsp"], params["n_dim"],
                                                           params["input_space"], params["n_pts_fixed_begin"], params["n_pts_fixed_end"])
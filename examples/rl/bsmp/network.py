import torch
from bsmp.utils import unpack_data_airhockey


class ConfigurationTimeNetwork(torch.nn.Module):
    def __init__(self, input_shape, output_shape, input_space):
        super(ConfigurationTimeNetwork, self).__init__()

        self.input_space = input_space

        activation = torch.nn.Tanh()
        W = 2048
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_shape[0], W), activation,
            torch.nn.Linear(W, W), activation,
            torch.nn.Linear(W, W), activation,
        )

        self.q_est = torch.nn.Sequential(
            torch.nn.Linear(W, W), activation,
            torch.nn.Linear(W, output_shape[0]), activation,
        )

        self.t_est = torch.nn.Sequential(
            torch.nn.Linear(W, output_shape[1]),
        )

    def normalize_input(self, x):
        low = torch.Tensor(self.input_space.low)[None]
        high = torch.Tensor(self.input_space.high)[None]
        normalized = (x - low) / (high - low)
        normalized = 2 * normalized - 1
        normalized[:, 0] = (x[:, 0] - 1.51) / (1.948 / 2. - 0.03165)
        normalized[:, 1] = x[:, 1] / (1.038 / 2. - 0.03165)
        return normalized

    def prepare_data(self, x):
        puck, puck_dot, q0, qd, dq0, dqd, ddq0, ddqd, _ = unpack_data_airhockey(x)
        x = self.normalize_input(x)
        return x, q0, qd, dq0, dqd, ddq0, ddqd

    def __call__(self, x):
        x, q0, qd, dq0, dqd, ddq0, ddqd = self.prepare_data(x)

        x = self.fc(x)
        q_prototype = self.q_est(x)
        ds_dt_prototype = self.t_est(x)

        return q_prototype, ds_dt_prototype

class ConfigurationTimeNetworkWrapper(ConfigurationTimeNetwork):
    def __init__(self, input_shape, output_shape, params, **kwargs):
        super(ConfigurationTimeNetworkWrapper, self).__init__(input_shape, output_shape, params["input_space"])
import torch
from examples.rl.bsmp.network import AirHockeyNetwork


class ValueNetwork(AirHockeyNetwork):
    def __init__(self, input_space):
        super().__init__(input_space)
        W = 128

        activation = torch.nn.Tanh()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_space.shape[0], W), activation,
            torch.nn.Linear(W, 1),
            #torch.nn.Linear(1, 1)
        )
        #self.fc._modules['6'].bias = torch.nn.Parameter(torch.tensor([-20.]), requires_grad=True)
        #self.fc._modules['3'].bias = torch.nn.Parameter(torch.tensor([-20.], dtype=torch.float64), requires_grad=True)
        #self.fc._modules['3'].weight = torch.nn.Parameter(torch.tensor([[10.]], dtype=torch.float64), requires_grad=True)
        self.log_scale = torch.nn.Parameter(torch.log(torch.tensor([10.], dtype=torch.float64)), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.tensor([-20.], dtype=torch.float64), requires_grad=True)
    
    def __call__(self, x):
        x, q0, qd, dq0, dqd, ddq0, ddqd = self.prepare_data(x)
        return torch.exp(self.log_scale) * self.fc(x) + self.bias
        #return x#10. * self.fc(x) - 20.
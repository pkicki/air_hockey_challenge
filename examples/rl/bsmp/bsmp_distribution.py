import torch
import numpy as np

from mushroom_rl.distributions import AbstractGaussianTorchDistribution
from mushroom_rl.utils.torch import TorchUtils


class DiagonalGaussianBSMPDistribution(AbstractGaussianTorchDistribution):
    def __init__(self, mu_approximator, sigma):
        self._mu_approximator = mu_approximator
        self._log_sigma = torch.nn.Parameter(torch.log(sigma))

        super().__init__(context_shape=self._mu_approximator.input_shape)

        self._add_save_attr(
            _mu_approximator='torch',
            _log_sigma='torch'
        )

    def parameters(self):
        return list(self._mu_approximator.model.network.parameters()) + [self._log_sigma]

    def estimate_mu(self, context):
        if context is None:
            context = np.zeros(self._mu_approximator.input_shape, dtype=np.float32)[None]
        if len(context.shape) == 1:
            context = context[None]
        #q_cps_mu, t_cps_mu = self._mu_approximator.model.network(
        #    torch.from_numpy(context).to(TorchUtils.get_device()).to(torch.float32))
        if isinstance(context, np.ndarray):
            context = torch.from_numpy(context)
        q_cps_mu, t_cps_mu = self._mu_approximator.model.network(context.to(TorchUtils.get_device()))
        # TODO: probably will cause troubles if used with batch size > 1
        #mu = torch.cat((q_cps_mu.flatten(), t_cps_mu.flatten()))
        mu = torch.cat((q_cps_mu, t_cps_mu), dim=-1)
        return mu

    def _get_mean_and_chol(self, context):
        mu = self.estimate_mu(context)
        return mu, torch.diag(torch.exp(self._log_sigma))
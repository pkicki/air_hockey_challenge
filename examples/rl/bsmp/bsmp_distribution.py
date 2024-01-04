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
        if isinstance(context, np.ndarray):
            context = torch.from_numpy(context)
        mu = self._mu_approximator(context)
        return mu

    def _get_mean_and_chol(self, context):
        mu = self.estimate_mu(context)
        return mu, torch.diag(torch.exp(self._log_sigma))


class DiagonalGaussianBSMPSigmaDistribution(AbstractGaussianTorchDistribution):
    def __init__(self, mu_approximator, log_sigma_approximator):
        self._mu_approximator = mu_approximator
        self._log_sigma_approximator = log_sigma_approximator

        super().__init__(context_shape=self._mu_approximator.input_shape)

        self._add_save_attr(
            _mu_approximator='torch',
            _log_sigma_approximator='torch'
        )

    def parameters(self):
        return list(self._mu_approximator.model.network.parameters()) + list(self._log_sigma_approximator.model.network.parameters())

    def estimate_mu(self, context):
        if context is None:
            context = np.zeros(self._mu_approximator.input_shape, dtype=np.float32)[None]
        if len(context.shape) == 1:
            context = context[None]
        if isinstance(context, np.ndarray):
            context = torch.from_numpy(context)
        mu = self._mu_approximator(context)
        return mu

    def estimate_log_sigma(self, context):
        if context is None:
            context = np.zeros(self._log_sigma_approximator.input_shape, dtype=np.float32)[None]
        if len(context.shape) == 1:
            context = context[None]
        if isinstance(context, np.ndarray):
            context = torch.from_numpy(context)
        log_sigma = self._log_sigma_approximator(context)
        return log_sigma

    def _get_mean_and_chol(self, context):
        mu = self.estimate_mu(context)
        log_sigma = self.estimate_log_sigma(context)
        return mu, torch.diag_embed(torch.exp(log_sigma), dim1=-2, dim2=-1)
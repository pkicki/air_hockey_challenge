import torch
import numpy as np

from mushroom_rl.distributions import AbstractGaussianTorchDistribution
from mushroom_rl.utils.torch import TorchUtils

from examples.rl.bsmp.multinormal_distribution import MultiNormalDistribution
from examples.rl.bsmp.utils import project_entropy, project_entropy_independently


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

class DiagonalGaussianBSMPSigmaDistribution(AbstractGaussianTorchDistribution):
    def __init__(self, mu_approximator, log_sigma_approximator, e_lb=None):
        self._mu_approximator = mu_approximator
        self._log_sigma_approximator = log_sigma_approximator
        self._e_lb = e_lb

        super().__init__(context_shape=self._mu_approximator.input_shape)

        self._add_save_attr(
            _mu_approximator='torch',
            _log_sigma_approximator='torch',
            _e_lb='primitive',
        )

    def set_e_lb(self, e_lb):
        self._e_lb = e_lb

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
        chol = torch.diag_embed(torch.exp(log_sigma), dim1=-2, dim2=-1)
        if self._e_lb is not None:
            #chol = project_entropy(chol, self._e_lb)
            chol = project_entropy_independently(chol, self._e_lb)
        return mu, chol


class DiagonalMultiGaussianBSMPSigmaDistribution(AbstractGaussianTorchDistribution):
    def __init__(self, mu_approximator, log_sigma_approximator, entropy_weights, splits):
        self._mu_approximator = mu_approximator
        self._log_sigma_approximator = log_sigma_approximator
        self._entropy_weights = entropy_weights
        self._splits = splits

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

    def _get_mean_and_log_sigma(self, context):
        mu = self.estimate_mu(context)
        log_sigma = self.estimate_log_sigma(context)
        return mu, log_sigma

    def distribution(self, context=None):
        mu, log_sigma = self._get_mean_and_log_sigma(context)
        splits = [0] + self._splits + [mu.shape[-1]]
        mus = [mu[:, splits[i]:splits[i+1]] for i in range(len(splits)-1)]
        chol_sigmas = [torch.diag_embed(torch.exp(log_sigma[:, splits[i]:splits[i+1]]), dim1=-2, dim2=-1) for i in range(len(splits)-1)]
        return MultiNormalDistribution(mus, chol_sigmas, self._entropy_weights)

    def log_pdf(self, theta, context=None):
        dist = self.distribution(context)
        splits = [0] + self._splits + [theta.shape[-1]]
        theta = [theta[..., splits[i]:splits[i+1]] for i in range(len(splits)-1)]
        return dist.log_prob(theta)

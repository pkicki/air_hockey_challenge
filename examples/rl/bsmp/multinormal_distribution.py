import torch


class MultiNormalDistribution:
    def __init__(self, means, covs, entropy_weights):
        self.entropy_weights = entropy_weights
        self.distributions = [torch.distributions.MultivariateNormal(loc=mu, scale_tril=chol_sigma, validate_args=False) for mu, chol_sigma in zip(means, covs)]
        assert len(self.distributions) == len(self.entropy_weights)

    def sample(self):
        return torch.cat([dist.sample() for dist in self.distributions], axis=-1)
    
    def log_prob(self, theta):
        return torch.sum(torch.stack([dist.log_prob(theta[i]) for i, dist in enumerate(self.distributions)], axis=-1), axis=-1)

    def entropy(self):
        return torch.sum(torch.stack([dist.entropy() for dist in self.distributions], axis=-1) * self.entropy_weights[None], axis=-1)
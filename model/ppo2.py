import torch
import torch.nn as nn
import torch.distributions as D


__all__ = ['PPO2', 'ppo2']


class PPO2(nn.Module):
    def __init__(self, input_dim, hidden_dim,  action_space):
        """
        ppo2 model
        :param input_dim: observation dimension
        :param hidden_dim: hidden state dimension
        :param action_space: action space
        """
        super(PPO2, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )


        self.policy_head = nn.Linear(hidden_dim, action_space)
        self.value_head = nn.Linear(hidden_dim, 1)



    def forward(self, x, action=None, distribution=D.categorical.Categorical):
        """
        compute latent state representation.
        such state representation is then used to execute the policy and estimate the value function
        :param x: input observation
        :param action: if None, execute the new policy. Otherwise, compute the negative log-likelihood of the new policy
        :param distribution: distribution type for the given action. Have to be a torch.distributions
        :return:
        """

        if not issubclass(distribution, torch.distributions.distribution.Distribution):
            raise NotImplementedError("distribution type have to be a valid torch.distribution class (logits of each action are used instead of probabilities).")

        latent_state = self.network(x)
        action_logit = self.policy_head(latent_state)   # might change during training. thus we recompute the neg_log_prob
        action_dist = distribution(logits=action_logit)

        if action is None:
            action = action_dist.sample()
            neg_log_prob = action_dist.log_prob(action) * -1.
            entropy = action_dist.entropy()
        else:
            neg_log_prob = action_dist.log_prob(action) * -1.
            entropy = action_dist.entropy()

        value_f = self.value_head(latent_state)
        # value_f = torch.squeeze(value_f)

        return value_f, action, neg_log_prob, entropy

    def loss(self, reward, value_f, neg_log_prob, entropy, advantages, old_value_f, old_neg_log_prob, clip_range, ent_coef, vf_coef):
        """
        compute loss
        :param reward: total reward obtained
        :param value_f: estimated value function
        :param neg_log_prob: negative log-likelihood of each action
        :param entropy: entropy of the action distribution
        :param advantages: estimated advantage of each action
        :param old_value_f: estimated value function using old parameters
        :param old_neg_log_prob: negative log-likelihood of each action using old parametres
        :param clip_range: policy clip value
        :param ent_coef: entropy discount coefficient
        :param vf_coef: value discount coefficient
        :return: total loss, policy loss, value loss, entropy, approximated KL-div between new and old action distribution
        """
        entropy_mean = entropy.mean()
        value_f_clip = old_value_f + torch.clamp(value_f - old_value_f, min=-clip_range, max=clip_range)

        value_loss1 = (value_f - reward)**2
        value_loss2 = (value_f_clip - reward)**2

        # value_loss = F.smooth_l1_loss(value_f, reward)
        value_loss = .5 * torch.mean(torch.max(value_loss1, value_loss2))

        ratio = torch.exp(old_neg_log_prob - neg_log_prob)

        pg_losses1 = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)

        pg_loss = torch.mean(torch.max(pg_losses1, pg_losses2))
        approx_kl = 0.5 * torch.mean((neg_log_prob - old_neg_log_prob)**2)
        # clip_frac = (torch.abs(ratio - 1.0) > clip_range).float().mean()

        loss = pg_loss - (entropy_mean * ent_coef) + (value_loss * vf_coef)

        return loss, pg_loss, value_loss, entropy_mean, approx_kl


    def reset_parameters(self):
        """
        randomly reset the model's parameters
        :return:
        """
        for p in self.parameters():
            if len(p.data.shape) == 2:
                # hidden layer
                nn.init.orthogonal_(p, gain=2**0.5)
            elif len(p.data.shape) == 1:
                # bias
                nn.init.constant_(p, 0.0)


def ppo2(reset_param=True, **kwargs):
    """
    ppo2 model based on the implementation proposed at https://github.com/openai/baselines/tree/master/baselines/ppo2.
    Paper: https://arxiv.org/abs/1707.06347
    :param reset_param: it True, randomly reset the initial parameters according using a (semi) orthogonal matrix.
    """
    model = PPO2(**kwargs)
    if reset_param:
        model.reset_parameters()

    return model
import numpy as np
import torch

class MemoryCollector(object):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner
    run():
    - Make a mini batch
    """
    def __init__(self, env, model, n_step, gamma, lam, device):
        self.env = env
        self.model = model

        self.n_step = n_step # number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where nenv is number of environment copies simulated in parallel)
        self.n_env = n_env = env.num_envs if hasattr(env, 'num_envs') else 1
        self.batch_ob_shape = (n_env * n_step,) + env.observation_space.shape

        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        self.device = device


    def eval_fn(self, obs):
        self.model.eval()
        with torch.set_grad_enabled(False):
            obs = torch.tensor(obs).float().to(self.device)
            value_f, action, neg_log_prob, entropy = self.model(obs)
            return value_f, action, neg_log_prob


    def run(self):
        obs = np.expand_dims(self.env.reset(), 0)
        done = False
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_done, mb_neg_log_prob = [],[],[],[],[],[]
        ep_infos = []
        # For n in range number of steps
        for _ in range(self.n_step):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            value_f, action, neg_log_prob = self.eval_fn(obs)
            action = action.cpu().item()
            neg_log_prob = neg_log_prob.cpu().item()
            value_f = value_f.cpu().item()

            mb_obs.append(obs.copy())
            mb_actions.append(action)
            mb_values.append(value_f)
            mb_neg_log_prob.append(neg_log_prob)
            mb_done.append(done)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            obs, rewards, done, info = self.env.step(action)
            obs = np.expand_dims(obs, 0)
            if 'episode' in info.keys():
                ep_infos.append(info.get('episode'))
            mb_rewards.append(rewards)



        #batch of steps to batch of rollouts
        mb_obs = np.concatenate(mb_obs, 0).astype(np.float32)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neg_log_prob = np.asarray(mb_neg_log_prob, dtype=np.float32)
        mb_done = np.asarray(mb_done, dtype=np.bool)

        last_values, _, _ = self.eval_fn(obs)
        last_values = last_values.cpu().item()

        # discount/bootstrap off value fn
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.n_step)):
            if t == self.n_step - 1:
                next_non_terminal = 1.0 - done
                next_values = last_values
            else:
                next_non_terminal = 1.0 - mb_done[t+1]
                next_values = mb_values[t+1]

            delta = mb_rewards[t] + self.gamma * next_values * next_non_terminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * next_non_terminal * lastgaelam

        mb_target_v = mb_advs + mb_values
        return mb_obs, mb_target_v, mb_done, mb_actions, mb_values, mb_neg_log_prob, ep_infos

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
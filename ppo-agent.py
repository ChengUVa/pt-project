import os
from pickle import FALSE
import gym
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from models import Actor, Critic
from experience import ExperienceSource
from utils import RewardTracker
from tensorboardX import SummaryWriter
import environ
import data

CHECKPOINT_EVERY_STEP = 100_000
TRAJECTORY_SIZE = 4096
BATCH_SIZE = 64
PPO_EPOCHES = 8
ENTROPY_BONUS = 0.005
LR = 0.00005
LR_RATIO = 5  # crt_lr / act_lr
MAX_STEPS = 20_000_000

PPO_EPS = 0.2
GAE_LAMBDA = 0.95
GAMMA = 0.999
GPU = False
U = 1.5
FEE = 0.05
MAX_EPISODE_LENGTH = 1000
SEED = 42


class Agent:
    """
    An Actor-Critic RL agent
    """

    def __init__(self, obs_dim, act_dim, device="cpu"):
        """
        Initialize the actor and the critic networks
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        self.actor = Actor(self.obs_dim, self.act_dim).to(self.device)
        self.critic = Critic(self.obs_dim).to(self.device)

    def initial_state(self):
        """
        Should create initial empty state for the agent. It will be called for the start of the episode
        :return: Anything agent want to remember
        """
        return None

    def __call__(self, states, agent_states):
        """
        Generate actioins using the actor network
        """
        assert isinstance(states, list)
        assert isinstance(agent_states, list)
        assert len(agent_states) == len(states)
        states = np.array(states, dtype=np.float32)
        states = torch.tensor(states).to(self.device)
        action_probs = self.actor(states).data.cpu()
        dist = Categorical(action_probs)
        actions = dist.sample().numpy()
        return actions, agent_states


class PPO:
    """
    Implements the Proximal Policy Optimization Algorithm (PPO)
    """

    def __init__(self, env, save_path, device="cpu"):
        """
        Initialize environment, agent, experience source, optimizers, etc.
        """
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        self.env = env
        self.save_path = save_path
        self.device = device
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.n
        self.agent = Agent(obs_dim, act_dim, self.device)
        self.exp_source = ExperienceSource(self.env, self.agent, steps_count=1)
        self.opt_crt = optim.Adam(self.agent.critic.parameters(), lr=LR)
        self.opt_act = optim.Adam(self.agent.actor.parameters(), lr=LR / LR_RATIO)

        self.writer = SummaryWriter(
            comment="-ppo_"
            + f"L{LR}R{LR_RATIO}_T{TRAJECTORY_SIZE}"
            + f"B{BATCH_SIZE}_E{PPO_EPOCHES}_EB{ENTROPY_BONUS}"
        )
        self.trajectory = []
        self.best_reward = None

    def calc_adv_ref(self, trajectory, net_crt, states, device="cpu"):
        """
        By trajectory calculate advantage and 1-step ref value
        :param trajectory: trajectory list
        :param net_crt: critic network
        :param states: states tensor
        :return: tuple with advantage numpy array and reference values
        """
        values = net_crt(states)
        values = values.squeeze().data.cpu().numpy()
        last_gae = 0.0
        result_adv = []
        result_ref = []
        for val, next_val, (exp,) in zip(
            reversed(values[:-1]), reversed(values[1:]), reversed(trajectory[:-1])
        ):
            if exp.done:
                delta = exp.reward - val
                last_gae = delta
            else:
                delta = exp.reward + GAMMA * next_val - val
                last_gae = delta + GAMMA * GAE_LAMBDA * last_gae
            result_adv.append(last_gae)
            result_ref.append(last_gae + val)  # advantage + value = q_value

        adv = torch.FloatTensor(list(reversed(result_adv)))
        ref = torch.FloatTensor(list(reversed(result_ref)))
        return adv.to(device), ref.to(device)

    def eval(self, step_idx, episodes=5, stochastic=False):
        """
        Evaluate the agent's policy (actor)
        """
        with torch.no_grad():
            rewards = 0.0
            steps = 0
            for _ in range(episodes):
                obs = self.env.reset()
                while True:
                    obs_v = np.array([obs], dtype=np.float32)
                    obs_v = torch.tensor(obs_v).to(self.device)
                    action_probs = self.agent.actor(obs_v).data.cpu()
                    dist = torch.distributions.Categorical(action_probs)
                    if stochastic:
                        action = dist.sample().numpy()[0]
                    else:
                        action = dist.probs.argmax(dim=1).numpy()[0]  # deterministic
                    obs, reward, done, _ = self.test_env.step(action)
                    rewards += reward
                    steps += 1
                    if done:
                        break
            rewards /= episodes
            steps /= episodes
        print(
            "Tested on %d episodes, mean reward %.2f, mean steps %d"
            % (episodes, rewards, steps)
        )
        self.writer.add_scalar("test_reward", rewards, step_idx)
        self.writer.add_scalar("test_steps", steps, step_idx)
        return rewards

    def train_epoch(self, traj_states, traj_actions, traj_adv, traj_ref, old_logprob):
        """
        Train the actor and critic networks for an epoch
        """
        for batch_ofs in range(0, len(self.trajectory), BATCH_SIZE):
            batch_l = batch_ofs + BATCH_SIZE
            states = traj_states[batch_ofs:batch_l]
            actions = traj_actions[batch_ofs:batch_l]
            batch_adv = traj_adv[batch_ofs:batch_l].unsqueeze(-1)
            batch_ref = traj_ref[batch_ofs:batch_l]
            batch_old_logprob = old_logprob[batch_ofs:batch_l]

            self.opt_crt.zero_grad()
            value = self.agent.critic(states)
            loss_value = F.mse_loss(value.squeeze(-1), batch_ref)
            loss_value.backward()
            self.opt_crt.step()

            self.opt_act.zero_grad()
            action_probs = self.agent.actor(states)
            dist = Categorical(action_probs)
            logprob_pi = dist.log_prob(actions)
            entropy = dist.entropy()

            ratio = torch.exp(logprob_pi - batch_old_logprob)
            surr_obj = batch_adv * ratio
            c_ratio = torch.clamp(ratio, 1.0 - PPO_EPS, 1.0 + PPO_EPS)
            clipped_surr = batch_adv * c_ratio

            loss_policy = (
                -torch.min(surr_obj, clipped_surr).mean()
                - ENTROPY_BONUS * entropy.mean()
            )
            loss_policy.backward()
            self.opt_act.step()

    def train(self):
        """
        Main training loop
        """
        with RewardTracker(self.writer) as tracker:
            for step_idx, exp in enumerate(self.exp_source):
                if step_idx > MAX_STEPS:
                    print(f"Training Stopped after {MAX_STEPS}!")
                    break

                rewards_steps = self.exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    self.writer.add_scalar("episode_steps", np.mean(steps), step_idx)
                    tracker.reward(np.mean(rewards), step_idx)

                # if step_idx % EVAL_INT == 0:
                #     rewards = self.eval(step_idx, episodes=EVAL_EPISODES, stochastic=EVAL_STOCHASTIC)
                #     if self.best_reward is None or self.best_reward < rewards:
                #         if self.best_reward is not None:
                #             print(
                #                 "Best reward updated: %.2f -> %.2f" % (self.best_reward, rewards)
                #             )
                #             #name = "best_%+.2f_%d.dat" % (rewards, step_idx)
                #             name = "best.dat"
                #             fname = os.path.join(SAVE_PATH, name)
                #             # torch.save(net_act.state_dict(), fname)
                #         self.best_reward = rewards
                #         if self.best_reward > STOP_REWARD:
                #             print("Solved!")
                #             break

                if step_idx % CHECKPOINT_EVERY_STEP == 0:
                    idx = step_idx // CHECKPOINT_EVERY_STEP
                    torch.save(
                        self.agent.actor.state_dict(),
                        os.path.join(self.save_path, "checkpoint-%3d.data" % idx),
                    )

                self.trajectory.append(exp)
                if len(self.trajectory) < TRAJECTORY_SIZE + 1:
                    continue

                traj_states = [t[0].state for t in self.trajectory]
                traj_actions = [t[0].action for t in self.trajectory]
                traj_states = torch.FloatTensor(np.array(traj_states)).to(self.device)
                traj_actions = torch.FloatTensor(np.array(traj_actions)).to(self.device)

                traj_adv, traj_ref = self.calc_adv_ref(
                    self.trajectory, self.agent.critic, traj_states, self.device
                )

                action_probs = self.agent.actor(traj_states)
                dist = Categorical(action_probs)
                old_logprob = dist.log_prob(traj_actions)

                # normalize advantages (mean should be zero)
                traj_adv = traj_adv - torch.mean(traj_adv)
                traj_adv /= torch.std(traj_adv)

                # drop last entry from the trajectory, as our adv and ref value calculated without it
                self.trajectory = self.trajectory[:-1]
                old_logprob = old_logprob[:-1].detach()

                for epoch in range(PPO_EPOCHES):
                    self.train_epoch(
                        traj_states, traj_actions, traj_adv, traj_ref, old_logprob
                    )

                self.trajectory.clear()


if __name__ == "__main__":
    beta = 0.02
    year = 2009

    save_path = os.path.join(
        "saves", f"{year}-L{LR}-T{TRAJECTORY_SIZE}-B{BATCH_SIZE}-N{PPO_EPOCHES}-E{ENTROPY_BONUS}-b{beta}"
    )
    os.makedirs(save_path, exist_ok=True)

    formation_start = "{}-01-01".format(year)
    formation_end = "{}-12-31".format(year)
    trade_start = "{}-01-01".format(year + 1)
    trade_end = "{}-12-31".format(year + 1)
    pairs = pd.read_csv(
        "pairs/{}_{}.csv".format(formation_start, formation_end), index_col=0
    )
    pairs = pairs[["Stock1", "Stock2"]].values

    spreads = {}
    for pair in pairs:
        ric1, ric2 = pair
        spread_t = "data/spread_t_{}_{}_{}_{}_{}.csv".format(
            ric1, ric2, "30min", trade_start, trade_end
        )
        spreads["{}_{}".format(ric1, ric2)] = data.load_spreads(spread_t)

    env = environ.SpreadEnv(
        spreads,
        beta=beta,
        upper=U,
        lower=-U,
        commission=FEE,
        reset_on_close=False,
        reward_on_close=False,
        reward_scale=5,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_EPISODE_LENGTH)

    device = "cuda" if GPU else "cpu"
    ppo = PPO(env, save_path, device)
    ppo.train()

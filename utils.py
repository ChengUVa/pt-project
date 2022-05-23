import numpy as np
import time
import sys

class RewardTracker:
    def __init__(self, writer, min_ts_diff=1.0):
        """
        Constructs RewardTracker
        :param writer: writer to use for writing stats
        :param min_ts_diff: minimal time difference to track speed
        """
        self.writer = writer
        self.min_ts_diff = min_ts_diff

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, frame, epsilon=None):
        self.total_rewards.append(reward)
        mean_reward = np.mean(self.total_rewards[-100:])
        ts_diff = time.time() - self.ts
        if ts_diff > self.min_ts_diff:
            speed = (frame - self.ts_frame) / ts_diff
            self.ts_frame = frame
            self.ts = time.time()
            epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
            print("%d: done %d episodes, mean reward %.3f, speed %.2f f/s%s" % (
                frame, len(self.total_rewards), mean_reward, speed, epsilon_str
            ))
            sys.stdout.flush()
            #self.writer.add_scalar("speed", speed, frame)
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        #self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        return mean_reward if len(self.total_rewards) > 30 else None


# def test_net(net, env,  device="cpu", count=3,):
#     rewards = 0.0
#     steps = 0
#     for _ in range(count):
#         obs = env.reset()
#         while True:
#             #obs_v = float32_preprocessor([obs]).to(device)
#             obs_v = np.array([obs], dtype=np.float32)
#             obs_v = torch.tensor(obs_v).to(device)
#             action_probs = net(obs_v).data.cpu()
#             dist = torch.distributions.Categorical(action_probs)
#             #action = dist.sample().numpy()[0]
#             action = dist.probs.argmax(dim=1).numpy()[0] # no random actions when testing
#             obs, reward, done, _ = env.step(action)
#             rewards += reward
#             steps += 1
#             if done:
#                 break
#     return rewards / count, steps / count


# def float32_preprocessor(states):
#     np_states = np.array(states, dtype=np.float32)
#     return torch.tensor(np_states)
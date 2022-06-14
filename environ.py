import gym
import gym.spaces
from gym.utils import seeding
import enum
import numpy as np
import data

DEFAULT_BARS_COUNT = 20  # 10
DEFAULT_COMMISSION = 0.005


class Actions(enum.Enum):
    Short = 0
    Neutral = 1
    Long = 2


class SpreadEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        prices,
        beta=1,
        upper=1,
        lower=-1,
        bars_count=DEFAULT_BARS_COUNT,
        commission=DEFAULT_COMMISSION,
        holding_cost=0.0,
        reset_on_close=False,
        random_ofs_on_reset=True,
        reward_on_close=False,
        volumes=False,
    ):
        assert isinstance(prices, dict)
        self._prices = prices
        self._state = State(
            beta,
            upper,
            lower,
            bars_count,
            commission,
            holding_cost=holding_cost,
            reset_on_close=reset_on_close,
            reward_on_close=reward_on_close,
            volumes=volumes,
        )
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=self._state.shape, dtype=np.float32
        )
        self.random_ofs_on_reset = random_ofs_on_reset
        self.seed(0)
        # print('r4')

    def reset(self):
        # make selection of the instrument and it's offset. Then reset the state
        self._instrument = self.np_random.choice(list(self._prices.keys()))
        # print(self._instrument)
        prices = self._prices[self._instrument]
        bars = self._state.bars_count
        if self.random_ofs_on_reset:
            offset = (
                self.np_random.choice(
                    prices.spread.shape[0] - bars * DEFAULT_BARS_COUNT
                )
                + bars
            )
        else:
            offset = bars
        self._state.reset(prices, offset)
        return self._state.encode()

    def step(self, action_idx):
        action = Actions(action_idx)
        reward, done = self._state.step(action)
        obs = self._state.encode()
        info = {"instrument": self._instrument, "offset": self._state._offset}
        return obs, reward, done, info

    def render(self, mode="human", close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        return [seed1, seed2]


class State:
    def __init__(
        self,
        beta,
        upper,
        lower,
        bars_count,
        commission,
        holding_cost=0.0,
        reset_on_close=False,
        reward_on_close=False,
        volumes=True,
    ):
        assert isinstance(bars_count, int)
        assert bars_count > 0
        assert isinstance(commission, float)
        assert commission >= 0.0
        assert isinstance(reset_on_close, bool)
        assert isinstance(reward_on_close, bool)
        self.bars_count = bars_count
        self.commission = commission
        self.holding_cost = holding_cost
        self.reset_on_close = reset_on_close
        self.reward_on_close = reward_on_close
        self.volumes = volumes
        self.beta = beta
        self.upper = upper
        self.lower = lower

    def reset(self, prices, offset):
        # assert isinstance(prices, Spreads)
        assert isinstance(prices, data.Spreads)
        assert offset >= self.bars_count - 1
        self.position = 0
        self.open_price = 0.0
        self.time_since_open = 0.0
        self._prices = prices
        self._offset = offset

    @property
    def shape(self):
        # [h, l, c] * bars + position_flag + rel_profit (since open) + holding period
        if self.volumes:
            return (2 * self.bars_count + 1 + 1,)
        else:
            return (self.bars_count + 1 + 1 + 1 ,)

    def encode(self):
        """
        Convert current state into numpy array.
        """
        res = np.ndarray(shape=self.shape, dtype=np.float32)
        shift = 0
        for bar_idx in range(-self.bars_count + 1, 1):
            res[shift] = self._prices.spread[self._offset + bar_idx]
            shift += 1
            if self.volumes:
                res[shift] = self._prices.volume[self._offset + bar_idx]
                shift += 1
        res[shift] = float(self.position)
        shift += 1
        if not self.position:
            res[shift] = 0.0  # profit is zero if no position is opened
            self.time_since_open = 0.0
        else:
            res[shift] = (self._cur_close() - self.open_price) * self.position * 10
            self.time_since_open += 1
        res[shift+1] = self.time_since_open * 0.01
        return res

    def _cur_close(self):
        """
        Calculate real close price for the current bar
        """
        # open = self._prices.open[self._offset]
        # rel_close = self._prices.close[self._offset]
        # return open * (1.0 + rel_close)
        return self._prices.spread[self._offset]

    def step(self, action):
        """
        Perform one step in our price, adjust offset, check for the end of prices
        and handle position change
        :param action:
        :return: reward, done
        """
        assert isinstance(action, Actions)
        reward = 0.0
        done = False
        close = self._cur_close()
        if self.beta > 0:
            # only apply reward shaping to entry/exit point
            if self.position == 0 and action != Actions.Neutral:
                if close > self.lower and close < self.upper:  # open a position
                    reward -= self.beta
            elif self.position == 1 and action != Actions.Long:
                if close < 0:  # close a long position
                    reward -= self.beta
            elif self.position == -1 and action != Actions.Short:
                if close > 0:  # close a short position
                    reward -= self.beta
        # if self.beta>0:
        #     if self.position == 0:
        #         if close < self.lower:
        #             suggested_action = Actions.Long
        #         elif close > self.upper:
        #             suggested_action = Actions.Short
        #         else:
        #             suggested_action = Actions.Neutral
        #     elif self.position == 1:
        #         if close < 0:
        #             suggested_action = Actions.Long
        #         else:
        #             suggested_action = Actions.Neutral
        #     elif self.position == -1:
        #         if close > 0:
        #             suggested_action = Actions.Short
        #         else:
        #             suggested_action = Actions.Neutral
                    
        #     if action != suggested_action:
        #         reward -= self.beta*abs(action.value - suggested_action.value)


        if self.position == 0 and action != Actions.Neutral:  # Open a position
            self.open_price = close
            reward -= self.commission
            if action == Actions.Long:
                self.position = 1
            elif action == Actions.Short:
                self.position = -1
        elif (self.position == 1) and (
            action != Actions.Long
        ):  # Close the long position
            reward -= self.commission
            done |= self.reset_on_close
            self.position = 0
            self.open_price = 0.0
            self.time_since_open = 0
            if self.reward_on_close:
                reward += close - self.open_price
        elif (self.position == -1) and (
            action != Actions.Short
        ):  # Close the short position
            reward -= self.commission
            done |= self.reset_on_close
            self.position = 0
            self.open_price = 0.0
            self.time_since_open = 0
            if self.reward_on_close:
                reward += self.open_price - close
        elif self.position != 0:
            # Keep the position
            reward -= self.holding_cost

        self._offset += 1
        prev_close = close
        close = self._cur_close()
        done |= self._offset >= self._prices.spread.shape[0] - 1

        if (self.position != 0) and not self.reward_on_close:
            reward += (close - prev_close) * self.position

        return reward, done


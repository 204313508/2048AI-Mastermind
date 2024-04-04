import random
from collections import deque
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import gym
from gym import spaces
from gym.utils import seeding

import numpy as np

from PIL import Image, ImageDraw, ImageFont

import itertools
import logging
from six import StringIO
import sys

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

class IllegalMove(Exception):
    pass

def stack(flat, layers=16):
    """Convert an [4, 4] representation into [4, 4, layers] with one layers for each value."""
    # representation is what each layer represents
    representation = 2 ** (np.arange(layers, dtype=int) + 1)

    # layered is the flat board repeated layers times
    layered = np.repeat(flat[:,:,np.newaxis], layers, axis=-1)

    # Now set the values in the board to 1 or zero depending whether they match representation.
    # Representation is broadcast across a number of axes
    layered = np.where(layered == representation, 1, 0)

    return layered

class Game2048Env(gym.Env):

    def __init__(self):
        # Definitions for game. Board must be square.
        self.size = 4
        self.w = self.size
        self.h = self.size
        self.squares = self.size * self.size

        # Maintain own idea of game score, separate from rewards
        self.score = 0

        # Members for gym implementation
        self.action_space = spaces.Discrete(4)
        # Suppose that the maximum tile is as if you have powers of 2 across the board.
        layers = self.squares
        self.observation_space = spaces.Box(0, 1, (self.w, self.h, layers), dtype=int)
        self.set_illegal_move_reward(-20.)
        self.set_max_tile(2048)

        # Initialise seed
        self.seed()

        # Reset ready for a game
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_illegal_move_reward(self, reward):
        """Define the reward/penalty for performing an illegal move. Also need
            to update the reward range for this."""
        # Guess that the maximum reward is also 2**squares though you'll probably never get that.
        # (assume that illegal move reward is the lowest value that can be returned
        self.illegal_move_reward = reward
        self.reward_range = (self.illegal_move_reward, float(2**self.squares))

    def set_max_tile(self, max_tile):
        """Define the maximum tile that will end the game (e.g. 2048). None means no limit.
           This does not affect the state returned."""
        assert max_tile is None or isinstance(max_tile, int)
        self.max_tile = max_tile

    # Implement gym interface
    def step(self, action):
        """Perform one step of the game. This involves moving and adding a new tile."""
        logging.debug("Action {}".format(action))
        score = 0
        done = None
        info = {
            'illegal_move': False,
        }
        try:
            score = float(self.move(action))
            self.score += score
            assert score <= 2**(self.w*self.h)
            self.add_tile()
            done = self.isend()
            reward = float(score)
        except IllegalMove:
            logging.debug("Illegal move")
            info['illegal_move'] = True
            done = True
            reward = self.illegal_move_reward

        #print("Am I done? {}".format(done))
        info['highest'] = self.highest()

        # Return observation (board state), reward, done and info dict
        return stack(self.Matrix), reward, done, info

    def reset(self):
        self.Matrix = np.zeros((self.h, self.w), int)
        self.score = 0

        logging.debug("Adding tiles")
        self.add_tile()
        self.add_tile()

        return stack(self.Matrix)

    # Implement 2048 game
    def add_tile(self):
        """Add a tile, probably a 2 but maybe a 4"""
        possible_tiles = np.array([2, 4, 8, 16, 32])
        tile_probabilities = np.array([0.6, 0.3, 0.07, 0.02, 0.01])
        val = self.np_random.choice(possible_tiles, 1, p=tile_probabilities)[0]
        empties = self.empties()
        assert empties.shape[0]
        empty_idx = self.np_random.choice(empties.shape[0])
        empty = empties[empty_idx]
        logging.debug("Adding %s at %s", val, (empty[0], empty[1]))
        self.set(empty[0], empty[1], val)

    def get(self, x, y):
        """Return the value of one square."""
        return self.Matrix[x, y]

    def set(self, x, y, val):
        """Set the value of one square."""
        self.Matrix[x, y] = val

    def empties(self):
        """Return a 2d numpy array with the location of empty squares."""
        return np.argwhere(self.Matrix == 0)

    def highest(self):
        """Report the highest tile on the board."""
        return np.max(self.Matrix)

    def move(self, direction, trial=False):
        """Perform one move of the game. Shift things to one side then,
        combine. directions 0, 1, 2, 3 are up, right, down, left.
        Returns the score that [would have] got."""
        if not trial:
            if direction == 0:
                logging.debug("Up")
            elif direction == 1:
                logging.debug("Right")
            elif direction == 2:
                logging.debug("Down")
            elif direction == 3:
                logging.debug("Left")

        changed = False
        move_score = 0
        dir_div_two = int(direction / 2)
        dir_mod_two = int(direction % 2)
        shift_direction = dir_mod_two ^ dir_div_two # 0 for towards up left, 1 for towards bottom right

        # Construct a range for extracting row/column into a list
        rx = list(range(self.w))
        ry = list(range(self.h))

        if dir_mod_two == 0:
            # Up or down, split into columns
            for y in range(self.h):
                old = [self.get(x, y) for x in rx]
                (new, ms) = self.shift(old, shift_direction)
                move_score += ms
                if old != new:
                    changed = True
                    if not trial:
                        for x in rx:
                            self.set(x, y, new[x])
        else:
            # Left or right, split into rows
            for x in range(self.w):
                old = [self.get(x, y) for y in ry]
                (new, ms) = self.shift(old, shift_direction)
                move_score += ms
                if old != new:
                    changed = True
                    if not trial:
                        for y in ry:
                            self.set(x, y, new[y])
        if changed != True:
            raise IllegalMove

        return move_score

    def combine(self, shifted_row):
        """Combine same tiles when moving to one side. This function always
           shifts towards the left. Also count the score of combined tiles."""
        move_score = 0
        combined_row = [0] * self.size
        skip = False
        output_index = 0
        for p in pairwise(shifted_row):
            if skip:
                skip = False
                continue
            combined_row[output_index] = p[0]
            if p[0] == p[1]:
                combined_row[output_index] += p[1]
                move_score += p[0] + p[1]
                # Skip the next thing in the list.
                skip = True
            output_index += 1
        if shifted_row and not skip:
            combined_row[output_index] = shifted_row[-1]

        return (combined_row, move_score)

    def shift(self, row, direction):
        """Shift one row left (direction == 0) or right (direction == 1), combining if required."""
        length = len(row)
        assert length == self.size
        assert direction == 0 or direction == 1

        # Shift all non-zero digits up
        shifted_row = [i for i in row if i != 0]

        # Reverse list to handle shifting to the right
        if direction:
            shifted_row.reverse()

        (combined_row, move_score) = self.combine(shifted_row)

        # Reverse list to handle shifting to the right
        if direction:
            combined_row.reverse()

        assert len(combined_row) == self.size
        return (combined_row, move_score)

    def isend(self):
        """Has the game ended. Game ends if there is a tile equal to the limit
           or there are no legal moves. If there are empty spaces then there
           must be legal moves."""

        if self.max_tile is not None and self.highest() >= self.max_tile:
            return True

        for direction in range(4):
            try:
                self.move(direction, trial=True)
                # Not the end if we can do any move
                return False
            except IllegalMove:
                pass
        return True

    def get_board(self):
        """Retrieve the whole board, useful for testing."""
        return self.Matrix

    def set_board(self, new_board):
        """Retrieve the whole board, useful for testing."""
        self.Matrix = new_board

# 2048 游戏环境
env = Game2048Env()
env.reset()

# 超参数
EPISODES = 5000  # 训练周期数：定义了训练过程中总共进行多少个周期（或者称为回合）。每个周期包含了一系列步骤，直到游戏结束为止。
STEPS = 1000  # 每个周期的最大步数：定义了每个周期内的最大步数。如果游戏在一个周期内达到了最大步数而没有结束，那么这个周期就会结束。它可以用来控制每个周期的时长，避免训练时间过长。
LEARNING_RATE = 0.01  # 学习率：定义了在每次参数更新时，模型沿着梯度方向更新的步长大小。较小的学习率可以使模型更加稳定，但可能会导致收敛速度较慢；较大的学习率可能会导致模型无法收敛或者不稳定。通常需要根据实际情况进行调整。
DISCOUNT_FACTOR = 0.85  # 折扣因子：定义了未来奖励的折现率。它控制着模型对未来奖励的重视程度，较高的折扣因子意味着模型更加重视未来奖励，而较低的折扣因子则意味着模型更加重视当前奖励。通常用于解决长期回报与短期回报之间的权衡问题。
EPSILON = 1.0  # ε-贪心算法中的初始探索率：在ε-贪心算法中，EPSILON定义了Agent在选择动作时进行探索的概率。当EPSILON较大时，Agent更可能选择随机动作来探索环境；当EPSILON较小时，Agent更可能选择当前已知最优的动作。EPSILON的值通常会随着训练的进行逐渐减小，以增加对已知最优动作的利用。
EPSILON_DECAY = 0.995  # ε-贪心算法中的探索率衰减率：EPSILON_DECAY定义了EPSILON在训练过程中的衰减速率。它控制着EPSILON随时间的变化，通常是一个小于1的值。通过逐渐减小EPSILON，可以使Agent在训练早期更多地进行探索，而在训练后期更多地利用已有经验。
MIN_EPSILON = 0.01  # ε-贪心算法中的最小探索率：定义了在训练过程中EPSILON的最小值。当EPSILON减小到这个值以下时，训练过程将不再继续减小EPSILON，以避免过度利用已有经验而导致性能下降。


# 输入和输出形状
STATE_SIZE = np.product(env.observation_space.shape)
ACTION_SIZE = env.action_space.n

# 经验回放
REPLAY_MEMORY_SIZE = 800  # 经验回放缓冲区大小：定义了存储Agent经验的缓冲区的大小。经验回放是一种通过存储和随机抽样Agent与环境交互的经验来训练模型的方法。缓冲区大小决定了可以存储多少条经验，较大的缓冲区可以存储更多的经验，从而使得模型能够更好地学习到环境的动态特性。

BATCH_SIZE = 64  # 批量大小：定义了从经验回放缓冲区中抽样的批量大小。在每次训练时，模型会从经验回放缓冲区中随机抽样一批经验，并用这些经验来进行参数更新。较大的批量大小可以使得模型在每次训练时使用更多的数据来更新参数，从而使得参数更新更加稳定；但同时也会增加计算开销。通常需要根据实际情况进行调整，以平衡训练效果和计算开销。

replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)  # 经验回放缓冲区：使用一个双端队列来存储Agent的经验，其中最大长度为经验回放缓冲区大小。当新的经验被添加到缓冲区时，如果缓冲区已满，则最早的经验会被自动移除，从而保持缓冲区的大小不超过设定的最大值。


# Q-network 模型：定义了用于估计动作价值的神经网络模型。
model = Sequential()  # 创建一个序贯模型，即将层按顺序堆叠在一起的线性堆叠模型。

# 添加输入层和隐藏层：输入层接受状态信息作为输入，隐藏层用于学习状态与动作价值之间的映射关系。
model.add(Dense(128, input_shape=(STATE_SIZE,), activation='relu'))  # 添加一个全连接层，包含128个神经元，使用ReLU激活函数。
model.add(Dense(128, activation='relu'))  # 添加一个全连接层，包含128个神经元，使用ReLU激活函数。

# 添加输出层：输出层用于输出每个动作的估计价值。
model.add(Dense(ACTION_SIZE, activation='linear'))  # 添加一个全连接层，神经元数量等于动作空间的大小，使用线性激活函数。

# 编译模型：定义了模型的损失函数和优化器。
model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))  # 使用均方误差作为损失函数，Adam优化器来更新模型参数，学习率为LEARNING_RATE。



def update_replay_memory(state, action, reward, next_state, done):
    replay_memory.append((state, action, reward, next_state, done))


def get_training_batch():
    batch = random.sample(replay_memory, BATCH_SIZE)
    states, targets = np.zeros((BATCH_SIZE, STATE_SIZE)), np.zeros((BATCH_SIZE, ACTION_SIZE))

    for i, (state, action, reward, next_state, done) in enumerate(batch):
        target = reward
        if not done:
            target = reward + DISCOUNT_FACTOR * np.amax(model.predict(next_state.reshape((1, STATE_SIZE))))

        target_vector = model.predict(state.reshape((1, -1)))
        target_vector[0][action] = target
        states[i] = state.flatten()
        targets[i] = target_vector

    return states, targets


def train_model():
    global EPSILON

    for episode in range(EPISODES):
        state = env.reset()
        for step in range(STEPS):
            if np.random.rand() <= EPSILON:
                legal_actions = [action for action in range(ACTION_SIZE) if env.is_legal_move(action)]
                action = np.random.choice(legal_actions)
            else:
                action_values = model.predict(state.reshape((1, -1)))[0]
                legal_action_values = {action: value for action, value in enumerate(action_values) if env.is_legal_move(action)}
                action = max(legal_action_values, key=legal_action_values.get)

            next_state, reward, done, _ = env.step(action)
            update_replay_memory(state, action, reward, next_state, done)

            if len(replay_memory) >= BATCH_SIZE:
                states, targets = get_training_batch()
                model.fit(states, targets, epochs=1, verbose=0)

            state = next_state

            if done:
                break

        EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
        print(f"Episode: {episode + 1}/{EPISODES}, Score: {env.score}, Epsilon: {EPSILON:.4f}")
        if (episode + 1) % 50 == 0:
            model.save(f"2048_qlearning_model_checkpoint-{episode + 1}.h5")
    model.save("2048_qlearning_model.h5")


if __name__ == "__main__":
    train_model()

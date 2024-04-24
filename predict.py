import numpy as np
from tensorflow.keras.models import load_model
import gym
from gym import spaces
from gym.utils import seeding
import itertools
import logging
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
    def is_legal_move(self, action):
        """Check if the given action is legal."""
        try:
            self.move(action, trial=True)
            return True
        except IllegalMove:
            return False
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

        self.add_tile()
        self.add_tile()

        return stack(self.Matrix)

    # Implement 2048 game
    def add_tile(self):
        """Add a tile"""
        if self.highest() < 32:
            possible_tiles = np.array([2])
            tile_probabilities = np.array([1])
        elif self.highest() < 128:
            possible_tiles = np.array([2, 4])
            tile_probabilities = np.array([0.8, 0.2])
        elif self.highest() < 256:
            possible_tiles = np.array([2, 4])
            tile_probabilities = np.array([0.73, 0.27])
        elif self.highest() < 512:
            possible_tiles = np.array([2, 4, 8, 16])
            tile_probabilities = np.array([0.73, 0.21, 0.059, 0.001])
        else:
            possible_tiles = np.array([2, 4, 8, 16, 32])
            tile_probabilities = np.array([0.73, 0.21, 0.051, 0.008, 0.001])

        val = self.np_random.choice(possible_tiles, 1, p=tile_probabilities)[0]
        empties = self.empties()
        assert empties.shape[0]
        empty_idx = self.np_random.choice(empties.shape[0])
        empty = empties[empty_idx]
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
        # if not trial:
        #     if direction == 0:
        #         logging.debug("Up")
        #     elif direction == 1:
        #         logging.debug("Right")
        #     elif direction == 2:
        #         logging.debug("Down")
        #     elif direction == 3:
        #         logging.debug("Left")

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


# 加载模型
model = load_model("trained_model.h5")

# 2048 游戏环境
env = Game2048Env()
env.reset()


def predict_move(state):
    # 获取当前状态下各个动作的预测值
    action_values = model.predict(state.reshape((1, -1)), verbose=0)[0]

    # 找到具有最大预测值的动作
    legal_actions = [action for action in range(4) if env.is_legal_move(action)]
    legal_action_values = {action: value for action, value in enumerate(action_values) if action in legal_actions}

    # 如果只剩一个位置，选择能够继续游戏且预测值更高的移动方向
    if len(env.empties()) == 1 or len(env.empties()) == 0:
        print("empty:",env.Matrix)
        max_value = -float('inf')
        best_action = None
        for action, value in legal_action_values.items():
            # 模拟执行动作并评估预测值
            temp_env = Game2048Env()
            temp_env.Matrix = np.copy(env.Matrix)
            _, _, done, _ = temp_env.step(action)
            if not done:
                if value > max_value:
                    max_value = value
                    best_action = action
        if best_action is not None:
            return best_action

    # 否则，选择具有最大预测值的动作
    predicted_action = max(legal_action_values, key=legal_action_values.get)
    return predicted_action



if __name__ == "__main__":
    total_scores = []  # 用于储存每一次游戏的分数
    games_ended_max_tile = 0  # 记录因为达到最大方块而结束的游戏次数
    highest_scores_count = {}  # 记录每个最高分数出现的次数

    for _ in range(100):  # 玩游戏100次
        board = env.reset()  # 重置环境
        done = False
        while not done:
            current_state = stack(env.Matrix)  # 获取当前状态
            action = predict_move(current_state)  # 预测最佳动作
            _, reward, done, info = env.step(action)  # 执行动作，获取结果
            if done:
                print(env.Matrix,info['highest'])
                total_scores.append(env.score)  # 记录分数
                if info['highest'] >= env.max_tile:  # 检查是否因为达到最大方块而结束
                    games_ended_max_tile += 1
                highest_score = info['highest']
                if highest_score in highest_scores_count:
                    highest_scores_count[highest_score] += 1
                else:
                    highest_scores_count[highest_score] = 1

    print("每个最高分数出现的次数：")
    for score, count in highest_scores_count.items():
        print(f"{score} 分出现了 {count} 次")

    print(f"因为达到最大方块而结束的游戏次数：{games_ended_max_tile}")






import tkinter as tk
import random
from collections import deque
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import itertools

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

class IllegalMove(Exception):
    pass

def stack(flat, layers=16):
    """Convert an [4, 4] representation into [4, 4, layers] with one layers for each value."""
    representation = 2 ** (np.arange(layers, dtype=int) + 1)
    layered = np.repeat(flat[:,:,np.newaxis], layers, axis=-1)
    layered = np.where(layered == representation, 1, 0)
    return layered
class IllegalMove(Exception):
    pass


def stack(flat, layers=16):
    """Convert an [4, 4] representation into [4, 4, layers] with one layers for each value."""
    representation = 2 ** (np.arange(layers, dtype=int) + 1)
    layered = np.repeat(flat[:, :, np.newaxis], layers, axis=-1)
    layered = np.where(layered == representation, 1, 0)
    return layered


class Game2048Env(gym.Env):
    def __init__(self):
        self.size = 4
        self.w = self.size
        self.h = self.size
        self.squares = self.size * self.size
        self.score = 0

        self.action_space = spaces.Discrete(4)
        layers = self.squares
        self.observation_space = spaces.Box(0, 1, (self.w, self.h, layers), dtype=int)
        self.set_illegal_move_reward(-20.)
        self.set_max_tile(2048)

        self.seed()
        self.reset()

    def is_legal_move(self, action):
        try:
            self.move(action, trial=True)
            return True
        except IllegalMove:
            return False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_illegal_move_reward(self, reward):
        self.illegal_move_reward = reward
        self.reward_range = (self.illegal_move_reward, float(2 ** self.squares))

    def set_max_tile(self, max_tile):
        assert max_tile is None or isinstance(max_tile, int)
        self.max_tile = max_tile

    def step(self, action):
        score = 0
        done = None
        info = {
            'illegal_move': False,
        }
        try:
            score = float(self.move(action))
            self.score += score
            assert score <= 2 ** (self.w * self.h)
            self.add_tile()
            done = self.isend()
            reward = float(score)
        except IllegalMove:
            info['illegal_move'] = True
            done = True
            reward = self.illegal_move_reward

        info['highest'] = self.highest()

        return stack(self.Matrix), reward, done, info

    def reset(self):
        self.Matrix = np.zeros((self.h, self.w), int)
        self.score = 0

        self.add_tile()
        self.add_tile()

        return stack(self.Matrix)

    def add_tile(self):
        possible_tiles = np.array([2, 4, 8, 16, 32])
        tile_probabilities = np.array([0.6, 0.3, 0.07, 0.02, 0.01])
        val = self.np_random.choice(possible_tiles, 1, p=tile_probabilities)[0]
        empties = self.empties()
        assert empties.shape[0]
        empty_idx = self.np_random.choice(empties.shape[0])
        empty = empties[empty_idx]
        self.set(empty[0], empty[1], val)

    def get(self, x, y):
        return self.Matrix[x, y]

    def set(self, x, y, val):
        self.Matrix[x, y] = val

    def empties(self):
        return np.argwhere(self.Matrix == 0)

    def highest(self):
        return np.max(self.Matrix)

    def move(self, direction, trial=False):
        changed = False
        move_score = 0
        dir_div_two = int(direction / 2)
        dir_mod_two = int(direction % 2)
        shift_direction = dir_mod_two ^ dir_div_two

        rx = list(range(self.w))
        ry = list(range(self.h))

        if dir_mod_two == 0:
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
                skip = True
            output_index += 1
        if shifted_row and not skip:
            combined_row[output_index] = shifted_row[-1]

        return (combined_row, move_score)

    def shift(self, row, direction):
        shifted_row = [i for i in row if i != 0]

        if direction:
            shifted_row.reverse()

        (combined_row, move_score) = self.combine(shifted_row)

        if direction:
            combined_row.reverse()

        assert len(combined_row) == self.size
        return (combined_row, move_score)

    def isend(self):
        if self.max_tile is not None and self.highest() >= self.max_tile:
            return True

        for direction in range(4):
            try:
                self.move(direction, trial=True)
                return False
            except IllegalMove:
                pass
        return True

    def get_board(self):
        return self.Matrix

    def set_board(self, new_board):
        self.Matrix = new_board


class GameUI(tk.Tk):
    def __init__(self, env, model):
        super().__init__()
        self.env = env
        self.model = model
        self.title("2048 Game")
        self.geometry("400x400")

        self.grid_frame = tk.Frame(self)
        self.grid_frame.pack()
        self.create_grid()

        self.button_frame = tk.Frame(self)
        self.button_frame.pack()
        self.create_buttons()

        self.step_count = 0

    def create_grid(self):
        self.grid_cells = []
        for i in range(self.env.size):
            row = []
            for j in range(self.env.size):
                cell = tk.Label(self.grid_frame, text='', font=('Helvetica', 16, 'bold'), width=4, height=2,
                                relief='ridge')
                cell.grid(row=i, column=j, padx=5, pady=5)
                row.append(cell)
            self.grid_cells.append(row)
        self.update_grid()

    def update_grid(self):
        for i in range(self.env.size):
            for j in range(self.env.size):
                value = self.env.get(i, j)
                if value == 0:
                    self.grid_cells[i][j].configure(text='', bg='gray')
                else:
                    self.grid_cells[i][j].configure(text=str(value), bg='white')

    def create_buttons(self):
        buttons = ['Up', 'Right', 'Down', 'Left']
        # 创建一个字典，为每个按钮指定其网格位置
        grid_positions = {
            'Up': (0, 1),
            'Left': (1, 0),
            'Down': (1, 1),
            'Right': (1, 2)
        }

        for button_text in buttons:
            button = tk.Button(self.button_frame, text=button_text, command=lambda b=button_text: self.move(b))
            # 使用grid而不是pack来布局按钮
            row, col = grid_positions[button_text]
            button.grid(row=row, column=col, padx=10, pady=10)

    def move(self, direction):
        actions = {'Up': 0, 'Right': 1, 'Down': 2, 'Left': 3}
        action = actions[direction]
        if self.env.is_legal_move(action):
            _, reward, done, _ = self.env.step(action)
            self.update_grid()
            self.train_model()
            if done:
                self.env.reset()
                self.update_grid()
            self.step_count += 1
            if self.step_count % 100 == 0:
                self.model.save(f"2048_checkpoint-{self.step_count}.h5")

    def train_model(self):
        if len(replay_memory) >= BATCH_SIZE:
            states, targets = self.get_training_batch()
            self.model.fit(states, targets, epochs=1, verbose=0)

    def get_training_batch(self):
        batch = random.sample(replay_memory, BATCH_SIZE)
        states, targets = np.zeros((BATCH_SIZE, STATE_SIZE)), np.zeros((BATCH_SIZE, ACTION_SIZE))

        for i, (state, action, reward, next_state, done) in enumerate(batch):
            target = reward
            if not done:
                target = reward + DISCOUNT_FACTOR * np.amax(
                    self.model.predict(next_state.reshape((1, STATE_SIZE)), verbose=0))

            target_vector = self.model.predict(state.reshape((1, -1)), verbose=0)
            target_vector[0][action] = target
            states[i] = state.flatten()
            targets[i] = target_vector

        return states, targets


if __name__ == "__main__":
    env = Game2048Env()

    EPISODES = 5000
    STEPS = 1000
    LEARNING_RATE = 0.01
    DISCOUNT_FACTOR = 0.95
    EPSILON = 1.0
    EPSILON_DECAY = 0.995
    MIN_EPSILON = 0.01
    STATE_SIZE = np.product(env.observation_space.shape)
    ACTION_SIZE = env.action_space.n
    REPLAY_MEMORY_SIZE = 1000
    BATCH_SIZE = 64
    replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
    model = Sequential()
    model.add(Dense(128, input_shape=(STATE_SIZE,), activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(ACTION_SIZE, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))
    model = load_model("2048_human.h5")
    game_ui = GameUI(env, model)
    game_ui.mainloop()

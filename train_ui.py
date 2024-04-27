import tkinter as tk
from tkinter import messagebox, scrolledtext
import threading
import time
import math
import re
import random
from collections import deque
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import itertools
import os


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
        if self.highest() < 32:
            possible_tiles = np.array([2])
            tile_probabilities = np.array([1])
        elif self.highest() < 128:
            possible_tiles = np.array([2, 4])
            tile_probabilities = np.array([0.9, 0.1])
        elif self.highest() < 256:
            possible_tiles = np.array([2, 4])
            tile_probabilities = np.array([0.8, 0.2])
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



def parse_array(array_str):
    # 使用正则表达式提取数组中的数字部分
    numbers = re.findall(r'\d+', array_str)
    # 将提取的数字转换为整数
    return [int(num) for num in numbers]


def train_model_from_file(learning_rate, episodes, progress_callback):
    # Check if the replay_memory.txt file exists, if not, create it
    if not os.path.exists("replay_memory.txt"):
        messagebox.showwarning("文件错误", "并未找到经验文件，请先双击play_ui.exe进行2048游戏后再进行训练")
        return
    progress_callback(f"训练轮数: {0}/{episodes}\n")
    DISCOUNT_FACTOR = 0.95
    STATE_SIZE = np.product(env.observation_space.shape)
    ACTION_SIZE = env.action_space.n
    # print(STATE_SIZE)
    REPLAY_MEMORY_SIZE = 500000
    BATCH_SIZE = 900
    replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

    model = Sequential()
    model.add(Dense(256, input_shape=(STATE_SIZE,), activation='relu'))  # 增加神经元数量
    model.add(Dense(256, activation='relu'))  # 增加更多层和神经元
    model.add(Dropout(0.2))  # 添加Dropout层以减少过拟合
    model.add(Dense(256, activation='relu'))  # 继续调整层和神经元
    model.add(Dense(256, activation='relu'))  # 继续调整层和神经元
    model.add(Dense(ACTION_SIZE, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate), metrics=['mae'])  # 添加metrics以便更好地观察训练过程

    with open("replay_memory.txt", "r") as file:
        lines = ""
        t = 0
        for line in file:
            t += 1
            lines += line.strip()
            if t % 7 == 0:  # Each experience contains 7 lines
                # print(t)
                state_str, action, reward, next_state_str, done_str = lines.split(",")
                # print(state_str)
                state = stack(np.array(parse_array(state_str), dtype=int).reshape(4, 4))
                action = int(action)
                reward = 2048.0 + float(reward)
                next_state = stack(np.array(parse_array(next_state_str), dtype=int).reshape(4, 4))
                done = bool(done_str)
                experience = (state, action, reward, next_state, done)
                replay_memory.append(experience)
                lines = ""  # Reset lines for the next experience

    if len(replay_memory) >= BATCH_SIZE:
        for i in range(episodes):
            # print("i:", i)
            states, targets = get_training_batch(replay_memory, model, BATCH_SIZE, STATE_SIZE, ACTION_SIZE,
                                                 DISCOUNT_FACTOR)
            model.fit(states, targets, epochs=1, verbose=0)
            tf.keras.backend.clear_session()
            # 更新进度显示
            progress_callback(f"训练轮数: {i+1}/{episodes}\n")
        model.save("2048_base.h5")
        progress_callback(f"训练成功，已保存到同目录下2048_base.h5，把该文件替换2048一键包中的同名文件即可使用")
        return model
    else:
        messagebox.showwarning("文件错误", "经验文件过少，请先双击play_ui.exe进行多次2048游戏后再进行训练")
        return model


def get_training_batch(replay_memory, model, BATCH_SIZE, STATE_SIZE, ACTION_SIZE, DISCOUNT_FACTOR):
    batch = random.sample(replay_memory, BATCH_SIZE)
    states, targets = np.zeros((BATCH_SIZE, STATE_SIZE)), np.zeros((BATCH_SIZE, ACTION_SIZE))

    for i, (state, action, reward, next_state, done) in enumerate(batch):
        target = reward
        if not done:
            target = reward + DISCOUNT_FACTOR * np.amax(model.predict(next_state.reshape((1, STATE_SIZE)), verbose=0))

        # target_vector = model.predict(state.reshape((1, STATE_SIZE)), verbose=0)
        target_vector = model.predict(state.reshape((1, -1)), verbose=0)
        target_vector[0][action] = target
        states[i] = state.flatten()
        targets[i] = target_vector

    return states, targets


# 当点击训练按钮时调用
def on_train_click():
    try:
        learning_rate = float(lr_entry.get())
        episodes = int(episodes_entry.get())
        if learning_rate <= 0 or learning_rate > 1:
            messagebox.showwarning("参数错误", "学习率需要在0到1之间")
            return
        if episodes <= 0:
            messagebox.showwarning("参数错误", "训练轮数需要大于0")
            return
        # 清空进度显示
        progress_text.configure(state='normal')
        progress_text.delete('1.0', tk.END)
        progress_text.configure(state='disabled')

        # 弹出提示信息
        messagebox.showinfo("提示", "已开始训练，请耐心等待，训练进度会在下方框图内显示，请勿多次点击开始训练，否则会多线程同时进行重复训练")
        # 使用线程来避免阻塞UI
        threading.Thread(target=start_training_thread, args=(learning_rate, episodes, update_progress)).start()
    except ValueError:
        messagebox.showwarning("参数错误", "请输入有效的数值")


# 启动训练的线程函数
def start_training_thread(learning_rate, episodes, progress_callback):
    # 在这里替换为您的训练函数
    # 注意将progress_callback用于在GUI中显示进度
    global env
    env = Game2048Env()
    train_model_from_file(learning_rate, episodes, progress_callback)


# 更新进度显示的函数
def update_progress(message):
    progress_text.configure(state='normal')
    progress_text.insert(tk.END, message)
    progress_text.configure(state='disabled')
    # 自动滚动到底部
    progress_text.see(tk.END)

def main():
    global lr_entry, episodes_entry, progress_text
    root = tk.Tk()
    root.title("模型训练")

    tk.Label(root, text="学习率:").grid(row=0, column=0)
    lr_entry = tk.Entry(root)
    lr_entry.grid(row=0, column=1)

    tk.Label(root, text="训练轮数:").grid(row=1, column=0)
    episodes_entry = tk.Entry(root)
    episodes_entry.grid(row=1, column=1)

    train_button = tk.Button(root, text="开始训练", command=on_train_click)
    train_button.grid(row=2, column=0, columnspan=2)

    # 创建一个滚动文本框，用于显示训练进度
    progress_text = scrolledtext.ScrolledText(root, state='disabled', height=10)
    progress_text.grid(row=3, column=0, columnspan=2, sticky='we', padx=10, pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()

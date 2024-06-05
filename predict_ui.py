import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
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
        self.newnumberflag = True
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
            # self.add_tile()
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
        # self.add_tile()
        # self.add_tile()

        return stack(self.Matrix)

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
    def is_legal_move(self, action):
        """Check if the given action is legal."""
        try:
            self.move(action, trial=True)
            return True
        except IllegalMove:
            return False
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



class Game2048Interface:
    def __init__(self, root):
        self.root = root
        self.root.title("2048下一步预测")

        self.mode_var = tk.StringVar()
        self.model_var = tk.StringVar()
        self.password_var = tk.StringVar()

        self.create_widgets()

        self.model = None
        self.env = Game2048Env()
        self.env.reset()
        self.direction_map = {
            'left': self.move_left,
            'right': self.move_right,
            'up': self.move_up,
            'down': self.move_down
        }

    def create_widgets(self):
        tk.Label(self.root, text="请在格子中输入数字，模型会给出下一步最佳滑动方向").pack()

        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # 添加关于菜单
        about_menu = tk.Menu(menubar, tearoff=0)
        about_menu.add_command(label="关于", command=self.show_about)
        menubar.add_cascade(label="关于", menu=about_menu)

        mode_frame = tk.Frame(self.root)
        mode_frame.pack(pady=5)

        tk.Label(mode_frame, text="选择游戏模式:").pack(side="left")
        mode_options = ["普通2048模式"]
        ttk.Combobox(mode_frame, textvariable=self.mode_var, values=mode_options).pack(side="left")

        model_frame = tk.Frame(self.root)
        model_frame.pack(pady=5)

        tk.Label(model_frame, text="选择模型版本:").pack(side="left")
        model_options = ["基础版模型"]#, "高级版模型"]
        model_combobox = ttk.Combobox(model_frame, textvariable=self.model_var, values=model_options)
        # model_combobox.pack(side="left")

        # password_frame = tk.Frame(self.root)
        # password_frame.pack(pady=5)

        # tk.Label(password_frame, text="输入解锁口令(仅限高级版模型):").pack(side="left")
        # ttk.Entry(password_frame, textvariable=self.password_var, show="*").pack(side="left")

        self.grid_frame = tk.Frame(self.root)
        self.grid_frame.pack(pady=10)

        self.entry_grid = [[None] * 4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                self.entry_grid[i][j] = tk.Entry(self.grid_frame, width=5)
                self.entry_grid[i][j].grid(row=i, column=j)

        # Add directional buttons
        direction_frame = tk.Frame(self.root)
        direction_frame.pack(pady=5)

        tk.Button(direction_frame, text="↑", command=lambda: self.slide_and_update_grid('up')).pack(side="top")
        tk.Button(direction_frame, text="←", command=lambda: self.slide_and_update_grid('left')).pack(side="left")
        tk.Button(direction_frame, text="→", command=lambda: self.slide_and_update_grid('right')).pack(side="right")
        tk.Button(direction_frame, text="↓", command=lambda: self.slide_and_update_grid('down')).pack(side="bottom")

        # Add clear button
        tk.Button(self.root, text="清空", command=self.clear_grid).pack(pady=10)

        tk.Button(self.root, text="预测下一步最佳滑动方向", command=self.predict_next_move).pack(pady=10)

    def predict_next_move(self):
        mode = "普通2048模式"
        model_version = "基础版模型"

        if mode == "" or model_version == "":
            messagebox.showerror("错误", "请选择游戏模式和模型版本")
            return

        if model_version == "高级版模型":
            password = self.password_var.get()
            if password != "伐姐菜菜带带":
                messagebox.showerror("错误", "口令不正确")
                return

        if model_version == "基础版模型":
            model_path = "2048_base.h5"
        else:
            model_path = "2048_vip.h5"

        try:
            board = np.zeros((4, 4))

            for i in range(4):
                for j in range(4):
                    if self.entry_grid[i][j].get():
                        entry_value = int(self.entry_grid[i][j].get())
                        if mode == "普通2048模式":
                            if entry_value not in [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
                                messagebox.showerror("错误", f"输入的数字{entry_value}不合法")
                                return

                        board[i][j] = entry_value

            for i in range(4):
                for j in range(4):
                    if board[i][j] == 1:
                        board[i][j] = 0
            self.env.Matrix = board.reshape((4, 4))
            # print(self.env.Matrix)
            current_state = stack(np.array(board.reshape(1, 16)).reshape((4, 4, 1)))
            if self.model is None:
                self.model = load_model(model_path)

            action = self.predict_move(current_state)

            if action == 0:
                info1 = "上"
            elif action == 1:
                info1 = "右"
            elif action == 2:
                info1 = "下"
            elif action == 3:
                info1 = "左"

            messagebox.showinfo("预测结果", f"下一步建议移动方向为：{info1}")
        except Exception as e:
            messagebox.showerror("错误", f"预测失败: {str(e)}")

    def is_legal_move(self, action):
        """Check if the given action is legal."""
        try:
            self.env.move(action, trial=True)
            return True
        except IllegalMove:
            return False

    def predict_move(self, board):
        action_values = self.model.predict(board.reshape((1, -1)), verbose=0)
        legal_actions = [action for action in range(4) if self.is_legal_move(action)]
        legal_action_values = {action: value for action, value in enumerate(action_values[0]) if action in legal_actions}

        if len(self.env.empties()) == 1 or len(self.env.empties()) == 0:
            max_value = -float('inf')
            best_action = None
            for action, value in legal_action_values.items():
                # 模拟执行动作并评估预测值
                temp_env = Game2048Env()
                temp_env.Matrix = np.copy(self.env.Matrix)
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


    def move_left(self, grid):
        mode = self.mode_var.get()

        self.env.Matrix = np.array(grid)
        self.env.step(3)
        new_grid = self.env.Matrix

        return new_grid.tolist()

    def move_right(self, grid):
        mode = self.mode_var.get()

        self.env.Matrix = np.array(grid)
        self.env.step(1)
        new_grid = self.env.Matrix

        return new_grid.tolist()

    def move_up(self, grid):
        mode = self.mode_var.get()

        self.env.Matrix = np.array(grid)
        self.env.step(0)
        new_grid = self.env.Matrix

        return new_grid.tolist()

    def move_down(self, grid):
        mode = self.mode_var.get()

        self.env.Matrix = np.array(grid)
        self.env.step(2)
        new_grid = self.env.Matrix

        return new_grid.tolist()

    def slide_and_update_grid(self, direction):
        try:
            mode = self.mode_var.get()
            board = [[int(self.entry_grid[i][j].get()) if self.entry_grid[i][j].get() else 0 for j in range(4)] for i in
                     range(4)]
            if mode == "普通2048模式":
                for row in board:
                    for num in row:
                        if num not in [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
                            messagebox.showerror("错误", f"输入的数字{num}不合法")
                            return

            moved_board = self.direction_map[direction](board)

            for i in range(4):
                for j in range(4):
                    self.entry_grid[i][j].delete(0, 'end')
                    self.entry_grid[i][j].insert(0, str(moved_board[i][j]))
        except Exception as e:
            messagebox.showerror("错误", f"移动失败: {str(e)}")

    def clear_grid(self):
        for i in range(4):
            for j in range(4):
                self.entry_grid[i][j].delete(0, 'end')

    def show_about(self):
        about_text = ("作者b站主页链接 https://space.bilibili.com/1420119869?spm_id_from=333.1007.0.0\n"
                      "作者贴吧主页链接 https://tieba.baidu.com/home/main?id=tb.1.136c2e5b.aAic-optjyrBDdcIJxHDBQ?t=1629621709&fr=index\n"
                      "github开源链接 https://github.com/204313508/2048AI-Mastermind\n"
                      "gitee开源链接 https://gitee.com/dududuck/2048AI-Mastermind\n")
        messagebox.showinfo("关于", about_text)


if __name__ == "__main__":
    root = tk.Tk()
    app = Game2048Interface(root)
    root.mainloop()

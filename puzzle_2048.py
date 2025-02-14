import constants as c
import binary_puzzle as bp
from visual2 import GameVisual
from mcts_ai import MCTSBoard
from tkinter import Button, Scale, Label, Menu, messagebox

class GameGrid(GameVisual):
    def __init__(self):
        super().__init__()
        self.master.bind("<Key>", self.key_down)
        self.board = bp.Board()
        self.mcts_board = MCTSBoard(self.board, simulation_time=1, exploration=0.1)

        self.commands = {
            c.KEY_UP: lambda: self.board.move2("up"),
            c.KEY_DOWN: lambda: self.board.move2("down"),
            c.KEY_LEFT: lambda: self.board.move2("left"),
            c.KEY_RIGHT: lambda: self.board.move2("right"),
            c.KEY_UP_ALT1: lambda: self.board.move2("up"),
            c.KEY_DOWN_ALT1: lambda: self.board.move2("down"),
            c.KEY_LEFT_ALT1: lambda: self.board.move2("left"),
            c.KEY_RIGHT_ALT1: lambda: self.board.move2("right"),
            c.KEY_UP_ALT2: lambda: self.board.move2("up"),
            c.KEY_DOWN_ALT2: lambda: self.board.move2("down"),
            c.KEY_LEFT_ALT2: lambda: self.board.move2("left"),
            c.KEY_RIGHT_ALT2: lambda: self.board.move2("right"),
        }

        self.check_valid_moves = {
            c.KEY_UP: lambda: self.board.can_swipe_up(),
            c.KEY_DOWN: lambda: self.board.can_swipe_down(),
            c.KEY_LEFT: lambda: self.board.can_swipe_left(),
            c.KEY_RIGHT: lambda: self.board.can_swipe_right(),
            c.KEY_UP_ALT1: lambda: self.board.can_swipe_up(),
            c.KEY_DOWN_ALT1: lambda: self.board.can_swipe_down(),
            c.KEY_LEFT_ALT1: lambda: self.board.can_swipe_left(),
            c.KEY_RIGHT_ALT1: lambda: self.board.can_swipe_right(),
            c.KEY_UP_ALT2: lambda: self.board.can_swipe_up(),
            c.KEY_DOWN_ALT2: lambda: self.board.can_swipe_down(),
            c.KEY_LEFT_ALT2: lambda: self.board.can_swipe_left(),
            c.KEY_RIGHT_ALT2: lambda: self.board.can_swipe_right(),
        }

        # 添加一个按钮，用于预测下一步最佳移动方向
        self.predict_button = Button(self, text="预测下一步", command=self.predict_next_move, bg="#f9f6f2", fg="#776e65", font=("Verdana", 12, "bold"), relief="flat")
        self.predict_button.grid(row=c.GRID_LEN, column=0, columnspan=c.GRID_LEN, sticky="nsew")

        # 添加滑动条和标签
        self.simulation_time_label = Label(self, text="模拟时间 (秒)", bg="#bbada0", fg="#776e65", font=("Verdana", 12, "bold"))
        self.simulation_time_label.grid(row=c.GRID_LEN + 1, column=0, columnspan=2, sticky="nsew")
        self.simulation_time_scale = Scale(self, from_=0.1, to=5.0, resolution=0.1, orient="horizontal", command=self.update_simulation_time, bg="#bbada0", fg="#776e65", highlightthickness=0, sliderrelief="flat")
        self.simulation_time_scale.set(0.5)
        self.simulation_time_scale.grid(row=c.GRID_LEN + 2, column=0, columnspan=2, sticky="nsew")

        self.exploration_label = Label(self, text="探索因子", bg="#bbada0", fg="#776e65", font=("Verdana", 12, "bold"))
        self.exploration_label.grid(row=c.GRID_LEN + 1, column=2, columnspan=2, sticky="nsew")
        self.exploration_scale = Scale(self, from_=0.0, to=1.0, resolution=0.01, orient="horizontal", command=self.update_exploration, bg="#bbada0", fg="#776e65", highlightthickness=0, sliderrelief="flat")
        self.exploration_scale.set(0.1)
        self.exploration_scale.grid(row=c.GRID_LEN + 2, column=2, columnspan=2, sticky="nsew")

        # 添加菜单栏
        self.menu_bar = Menu(self.master)
        self.master.config(menu=self.menu_bar)

        self.file_menu = Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label="关于", command=self.show_about)
        self.file_menu.add_command(label="参数说明", command=self.show_parameters_info)
        # 添加操作说明命令
        self.file_menu.add_command(label="操作说明", command=self.show_operation_instructions)
        self.menu_bar.add_cascade(label="帮助", menu=self.file_menu)

        self.update_grid_cells()
        self.mainloop()

    def key_down(self, event):
        key = event.keysym
        print(event)
        if key == c.KEY_QUIT: exit()
        if key in self.commands:
            if self.check_valid_moves[key]():
                self.commands[key]()
                self.update_grid_cells()
                if self.board.is_game_over():
                    self.show_game_over()

    def predict_next_move(self):
        from tkinter.messagebox import showinfo
        best_move = self.mcts_board.get_best_move()
        if best_move:
            move_dict = {"up": "上", "down": "下", "left": "左", "right": "右"}
            showinfo("预测结果", f"下一步最佳移动方向：{move_dict[best_move]}")
        else:
            showinfo("预测结果", "没有有效的移动方向")

    def update_simulation_time(self, value):
        self.mcts_board.simulation_time = float(value)

    def update_exploration(self, value):
        self.mcts_board.exploration = float(value)

    def show_about(self):
        about_text = ("作者b站主页链接 https://space.bilibili.com/1420119869?spm_id_from=333.1007.0.0\n"
                      "作者贴吧主页链接 https://tieba.baidu.com/home/main?id=tb.1.136c2e5b.aAic-optjyrBDdcIJxHDBQ?t=1629621709&fr=index\n"
                      "github开源链接 https://github.com/204313508/2048AI-Mastermind\n"
                      "gitee开源链接 https://gitee.com/dududuck/2048AI-Mastermind\n")
        messagebox.showinfo("关于", about_text)

    def show_parameters_info(self):
        parameters_info = ("模拟时间：单位为秒。模拟时间越长，算法进行的模拟次数越多，决策越准确。推荐值：0.5秒。\n"
                           "探索因子：范围在0到1之间。探索因子越大，随机性越大，算法越倾向于探索未知的路径，而不是选择当前已知的最佳路径。该项过小算法可能会陷入局部最优解，而过大可能会降低算法的决策准确性。推荐值：0.1。")
        messagebox.showinfo("参数说明", parameters_info)

    def show_operation_instructions(self):
        operation_instructions = "点击键盘上的上下左右键即可移动"
        messagebox.showinfo("操作说明", operation_instructions)

game_grid = GameGrid()
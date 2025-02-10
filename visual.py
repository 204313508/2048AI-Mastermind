from tkinter import Frame, Label, CENTER
import constants as c
import binary_puzzle as bp
import numpy as np

class GameVisual(Frame):
    def __init__(self):
        Frame.__init__(self)
        self.grid()
        self.master.title('2048')
        self.grid_cells = []
        self.init_grid()
        self.board = None

    def init_grid(self):
        background = Frame(self, bg=c.BACKGROUND_COLOR_GAME, width=c.SIZE, height=c.SIZE)
        background.grid()

        for i in range(c.GRID_LEN):
            grid_row = []
            for j in range(c.GRID_LEN):
                cell = Frame(
                    background,
                    bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                    width=c.SIZE / c.GRID_LEN,
                    height=c.SIZE / c.GRID_LEN
                )
                cell.grid(
                    row=i,
                    column=j,
                    padx=c.GRID_PADDING,
                    pady=c.GRID_PADDING
                )
                t = Label(
                    master=cell,
                    text="",
                    bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                    justify=CENTER,
                    font=c.FONT,
                    width=5,
                    height=2
                )
                t.grid()
                t.bind("<Button-1>", lambda event, x=i, y=j: self.on_cell_click(event, x, y))
                grid_row.append(t)
            self.grid_cells.append(grid_row)

    def update_grid_cells(self):
        game_board = self.board.get_2048_board()
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                new_number = game_board[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    exponent = int(np.log2(new_number))
                    self.grid_cells[i][j].configure(
                        text=str(exponent),
                        bg=c.BACKGROUND_COLOR_DICT[new_number],
                        fg=c.CELL_COLOR_DICT[new_number]
                    )
        self.update_idletasks()

    def on_cell_click(self, event, x, y):
        from tkinter.simpledialog import askinteger
        value = askinteger("输入数字", "请输入0到11之间的数字:")
        if value is not None and 0 <= value <= 11:
            new_number = 2 ** value if value != 0 else 0
            self.board.set_value((x, y), new_number)  # 修改这里，将参数从三个改为一个元组
            self.update_grid_cells()

    def show_game_over(self):
        self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
        self.grid_cells[1][2].configure(text="Lose!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)

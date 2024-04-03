import numpy as np
from tensorflow.keras.models import load_model
import keras

from gym_2048.envs import Game2048Env

# 加载模型
model = load_model("2048_qlearning_model.h5")

# 2048 游戏环境
env = Game2048Env()
env.reset()
def predict_move(board):
    state = np.array(board, dtype=np.float32).flatten().reshape((1, -1))
    action = np.argmax(model.predict(state))

    return action

if __name__ == "__main__":
    board = env.reset()

    while True:
        # 输入当前 4x4 格子的数量
        board_input = list(map(int, input("请输入当前 4x4 格子的数量（空格分隔）：").split()))
        board = np.array(board_input).reshape((4, 4))
        board = np.expand_dims(board, axis=-1)
        board = np.repeat(board, 16, axis=-1).flatten()

        action = predict_move(board)

        print("下一步最佳移动方案：")
        if action == 0:
            info1 = "左"
        elif action == 1:
            info1 = "下"
        elif action == 2:
            info1 = "右"
        elif action == 3:
            info1 = "上"

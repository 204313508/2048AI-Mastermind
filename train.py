import numpy as np
import gym
import random
from collections import deque
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from gym_2048.envs import game2048_env, Game2048Env

# 创建 2048 游戏环境
env = Game2048Env()
env.reset()

# 设置超参数
EPISODES = 1500  # 总训练轮数
STEPS = 1000  # 每轮最大步数
LEARNING_RATE = 0.001  # 学习率
DISCOUNT_FACTOR = 0.95  # 折扣因子
EPSILON = 1.0  # 探索率
EPSILON_DECAY = 0.995  # 探索率衰减
MIN_EPSILON = 0.01  # 探索率下限

# 输入和输出的形状
STATE_SIZE = np.product(env.observation_space.shape)  # 状态空间大小
ACTION_SIZE = env.action_space.n  # 动作空间大小

# 设置经验回放存储
REPLAY_MEMORY_SIZE = 1000  # 经验回放缓冲区大小
BATCH_SIZE = 32  # 批量训练大小
replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)  # 创建经验回放缓冲区

# 创建 Q-network 模型
model = Sequential()
model.add(Dense(64, input_shape=(STATE_SIZE,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(ACTION_SIZE, activation='linear'))
model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))


def update_replay_memory(state, action, reward, next_state, done):
    """
    更新经验回放缓冲区
    """
    replay_memory.append((state, action, reward, next_state, done))


def get_training_batch():
    """
    获取训练批次数据
    """
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
    """
    训练模型
    """
    global EPSILON

    for episode in range(EPISODES):
        state = env.reset()
        for step in range(STEPS):
            if np.random.rand() <= EPSILON:
                action = env.action_space.sample()
            else:
                action = np.argmax(model.predict(state.reshape((1, -1))))

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

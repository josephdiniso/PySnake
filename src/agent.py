from collections import deque
import random

import torch
import numpy as np

from main import Game, TriDirections
from model import Linear_QNet, QTrainer

MAX_MEM = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self):
        self.memory = deque(maxlen=MAX_MEM)
        self.n_games: int = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory
        for state, action, reward, next_state, done in mini_sample:
            self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


def train():
    record = 0
    agent = Agent()
    game = Game(True)
    counter: int = 0
    game.set_game_time(1)
    while True:
        state_old = game.get_state()
        move = agent.get_action(state_old)
        if move == [1, 0, 0]:
            final_move = TriDirections.STRAIGHT
        elif move == [0, 1, 0]:
            final_move = TriDirections.RIGHT
        else:
            final_move = TriDirections.LEFT
        game.set_direction(final_move)
        reward, done, score = game.game_logic()
        if reward == 0:
            counter += 1
        else:
            counter = 0
        state_new = game.get_state()
        if counter > 100:
            done = True
            reward = -10
        if counter > 100:
            game.set_game_time(5)
        agent.train_short_memory(state_old, move, reward, state_new, done)
        agent.remember(state_old, move, reward, state_new, done)
        if done:
            counter = 0
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            if score > record:
                record = score
                agent.model.save()
            print('Game', agent.n_games, 'Score', score, 'Record:', record)


def main():
    train()


if __name__ == "__main__":
    main()

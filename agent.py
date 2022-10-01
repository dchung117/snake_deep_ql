import random
from collections import deque

import numpy as np
import torch

from snake import SnakeGameEnvironment, Direction, Point


MAX_MEMORY = 100_000 # max buffer size
BATCH_SIZE = 100 # batch size for training deep q-net
lr = 0.001 # learning rate

class Agent(object):
    def __init__(self):
        self.n_games = 0 # number of games
        self.epsilon = 0 # exploration vs. exploitation (epsilon-greedy exploration policy)
        self.gamma = 0 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # memory buffer -> queue w/ max length ( removes old data )

        # TODO: deep q-net

    def get_state(self, game):
        pass

    def remember(self, state, action, reward, next_state, done):
        pass

    def train_long_memory(self):
        pass

    def train_short_memory(self, state, action, reward, next_state, done):
        pass

    def get_action(self, state):
        pass

def train():
    # Keep track of game scores
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    best_score = 0

    # Initialize agent
    agent = Agent()

    # Create game
    game = SnakeGameEnvironment()

    # Begin training loop
    while True:
        # Get current state
        state_old = agent.get_state(game)

        # Get next action
        action = agent.get_action(state_old)

        # Apply action to game
        reward, game_over, score = game.play_step(action)

        # Get new state
        state_new = agent.get_state(game)

        # Train on short memory (on state-action-reward-new_state transition)
        agent.train_short_memory(state_old, action, reward, state_new, game_over)

        # Remember state, action, reward, new state transition
        agent.remember(state_old, action, reward, state_new, game_over)

        if game_over:
            # Reset the game
            game.reset()
            agent.n_games += 1

            # Train the long memory (on all games that in memory)
            agent.train_long_memory()

            # Update high score
            if score > best_score:
                best_score = best_score

                # TODO: Save model for best score

            print("Game: ", agent.n_games, "Score: ", score, "Best Score: ", best_score)

            # TODO: plot agent learning

if __name__ == "__main__":
    train()
import random
from collections import deque
from tkinter import N

import numpy as np
import torch

from snake import SnakeGameEnvironment, Direction, Point, BLOCK_SIZE
from model import LinearQNet, QNetTrainer
from helper import plot

MAX_MEMORY = 100_000 # max buffer size
BATCH_SIZE = 1000 # batch size for training deep q-net
LR = 0.001 # learning rate

class Agent(object):
    def __init__(self):
        self.n_games = 0 # number of games
        self.epsilon = 0 # exploration vs. exploitation (epsilon-greedy exploration policy)
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # memory buffer -> queue w/ max length ( removes old data )

        # initialize model, trainer
        self.model = LinearQNet(input_size=11, hidden_size=256, output_size=3)
        self.trainer = QNetTrainer(model=self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game: SnakeGameEnvironment) -> np.ndarray:
        # Get snake head position, bounds, and directions
        head = game.snake[0]
        head_l = Point(head.x - BLOCK_SIZE, head.y)
        head_r = Point(head.x + BLOCK_SIZE, head.y)
        head_u = Point(head.x, head.y - BLOCK_SIZE)
        head_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Danger states (straight, left, right)
        danger_straight = (dir_l and game.is_collision(head_l) or \
            dir_r and game.is_collision(head_r) or \
            dir_u and game.is_collision(head_u) or \
            dir_d and game.is_collision(head_d)
            )
        danger_left = (dir_l and game.is_collision(head_d) or \
            dir_r and game.is_collision(head_u) or \
            dir_u and game.is_collision(head_l) or \
            dir_d and game.is_collision(head_r)
            )
        danger_right = (dir_l and game.is_collision(head_u) or \
            dir_r and game.is_collision(head_d) or \
            dir_u and game.is_collision(head_r) or \
            dir_d and game.is_collision(head_l)
            )

        # Food location (left, right, up, down)
        food_l = game.food.x < head.x
        food_r = game.food.x > head.x
        food_u = game.food.y < head.y
        food_d = game.food.y > head.y

        # Make state vector
        state = np.array([danger_straight, danger_left, danger_right,
            dir_l, dir_r, dir_u, dir_d,
            food_l, food_r, food_u, food_d], dtype=np.int64)

        return state

    def remember(self, state: np.ndarray, action: list, reward: int, next_state: np.ndarray, done: bool) -> None:
        # Append transition to memory
        # note: deque will pop off top element when length exceeds MAX_MEMORY
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        # Train on one batch of transitions from memory buffer
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, k=BATCH_SIZE)
        else:
            sample = self.memory

        # Train model on mini-batch
        states, actions, rewards, next_states, dones = zip(*sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state: np.ndarray, action: list, reward: int, next_state: np.ndarray, done: bool) -> None:
        # Train Q-Net on current game transition
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state: np.ndarray) -> list:
        # Initialize action_vector
        action = [0, 0, 0]

        # Define epsilon (i.e. more games played, less likely to explore)
        self.epsilon = 80 - self.n_games

        # Sample a random move
        if random.randint(0, 200) < self.epsilon:
            a_idx = random.randint(0, 2)
        else: # Take best action from q-net
            state = torch.tensor(state, dtype=torch.float)
            q = self.model(state)
            a_idx = torch.argmax(q).item()
        action[a_idx] = 1

        return action

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
                agent.model.save()

            print("Game: ", agent.n_games, "Score: ", score, "Best Score: ", best_score)

            # Append game score
            plot_scores.append(score)

            # Get new mean score
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            # plot agent learning
            plot(plot_scores, plot_mean_scores)

if __name__ == "__main__":
    train()
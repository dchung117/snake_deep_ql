import random
from enum import Enum
from collections import namedtuple
from typing import Optional

import numpy as np
import pygame


pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

# size of one block (20 x 20)
BLOCK_SIZE = 20
SPEED = 20

class SnakeGameEnvironment:
    def __init__(self, w=640, h=480):
        # display dimensions
        self.w = w
        self.h = h

        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')

        # set clock
        self.clock = pygame.time.Clock()

        # init game state
        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        # head starts at the center of the map
        self.head = Point(self.w/2, self.h/2)

        # Snake is initially three units long
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()

        # Initialize the frame number for counting
        self.frame = 0

    def _place_food(self):
        # Randomly place food on board
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)

        # Redo if food is on top of snake
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        # Increment frame iteration
        self.frame += 1

        # Initialize reward for frame
        reward = 0

        # 1. collect user input
        for event in pygame.event.get():
            # quit pygame
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head) # place new head at front of snake "list" object

        # 3. check if game over
        game_over = False
        if self.is_collision() or self.frame > 100*len(self.snake): # end game if collision or snake doesn't eat food in time
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop() # pop off old tail position of snake (only if snake didn't eat; otherwise, the snake gets one square larger)

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self, point: Optional[Point] =None):
        # Default point is None
        if point is None:
            point = self.head

        # hits boundary
        if point.x > self.w - BLOCK_SIZE or point.x < 0 or point.y > self.h - BLOCK_SIZE or point.y < 0:
            return True
        # hits itself
        if point in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)
        for pt in self.snake:
            # Draw large blue rectangle
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))

            # Draw small blue recangle
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        # Draw food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # Render score
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip() # update blitted text onto surface

    def _move(self, action):
        # Determine direction based on action and heading [straight, right, left]
        directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]

        # Get the idx of current snake heading
        idx = directions.index(self.direction)

        # Get action idx [straight, right, left]
        action_idx = np.argwhere(action == 1).flatten()[0]

        # Get new direction
        if action_idx == 0: # straight
            new_idx = idx
        elif action_idx == 1: # right
            new_idx = (idx + 1) % 4
        else: # left
            new_idx = (idx - 1) % 4
        self.direction = directions[new_idx]

        # Get head position (x, y)
        x = self.head.x
        y = self.head.y

        # Move head along the current moving direction
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        # Update head position
        self.head = Point(x, y)

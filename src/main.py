#!/usr/bin/env python3
import random
from enum import Enum
import sys

import argparse
import pygame
import math
import random
import numpy as np


# TODO: Modularize so it can work with any size

class Directions(Enum):
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4


class TriDirections(Enum):
    STRAIGHT = 0
    RIGHT = 1
    LEFT = 2


# PyGame Initializations
block_size: int = 20
blocks = 20
size = (blocks * block_size + 4, blocks * block_size + 4)
screen = pygame.display.set_mode(size)
clock = pygame.time.Clock()

pygame.init()
pygame.mixer.init(22050, -8, 16, 65536)
pygame.display.set_caption("PySnake")

# TODO: Figure out a more proper place to store these colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
LIGHT_GREEN = (0, 200, 0)
RED = (255, 0, 0)
LIGHT_RED = (200, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (66, 71, 235)
ORANGE = (247, 165, 0)
LIME = (0, 247, 66)
PURPLE = (155, 46, 201)
TEAL = (0, 128, 128)
DARK_BLUE = (3, 86, 252)


class Snake:
    def __init__(self, x: int, y: int, direction: int = Directions.UP, head: bool = False) -> None:
        """
        Args:
            x (int): x position of piece
            y (int): y position of piece
            direction (int): Enum designation for starting direction of piece
            head (bool): Specifies if this item is the head of the snake

        Returns:
            None
        """
        if direction == Directions.UP:
            self.x = x
            self.y = y + 20
        elif direction == Directions.RIGHT:
            self.x = x - 20
            self.y = y
        elif direction == Directions.DOWN:
            self.x = x
            self.y = y - 20
        elif direction == Directions.LEFT:
            self.x = x + 20
            self.y = y
        self.dir = direction
        self.instruction = None
        self.child = None
        self.head = head

    def move(self, change_dir=False) -> None:
        """
        Moves the position in the correct direction and sends an instruction to a potential child if there is one.

        Returns:
            None
        """
        if self.instruction:
            self.dir = self.instruction
            if self.child:
                self.child.instruction = self.dir
            self.instruction = None
        if change_dir and self.child and self.head:
            self.child.instruction = self.dir
        if self.dir == Directions.UP:
            self.y -= 20
        elif self.dir == Directions.RIGHT:
            self.x += 20
        elif self.dir == Directions.DOWN:
            self.y += 20
        elif self.dir == Directions.LEFT:
            self.x -= 20

    def create(self) -> None:
        """
        Gets the tail member of the snake and adds a new child to it.

        Returns:
            None
        """
        obj: Snake = self
        while obj.child:
            obj = obj.child
        obj.child = Snake(obj.x, obj.y, direction=obj.dir)
        return obj.child


class Fruit:
    def __init__(self, snakes) -> None:
        """
        Generates a new fruit object where it does not collide with any of the snake

        Args:
            snakes (List[Snake]): List of all snake pieces

        Returns:
            None
        """
        self.x = random.randint(1, blocks - 1) * block_size + 10
        self.y = random.randint(1, blocks - 1) * block_size + 10
        snake_pos = [(snake.x, snake.y) for snake in snakes]
        while (self.x, self.y) in snake_pos:
            self.x = random.randint(1, 24) * block_size + 10
            self.y = random.randint(1, 24) * block_size + 10


def pause() -> None:
    """
    Static function to pause the screen is the pause key (p) is pressed

    Returns:
        None
    """
    paused = True
    while paused:
        pos = pygame.mouse.get_pos()
        if (size[0] - 4) // 2 - 50 <= pos[0] <= size[1] // 2 + 50 <= pos[1] <= size[1] // 2 + 90:
            button_color = DARK_BLUE
        else:
            button_color = BLUE
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    paused = False
            elif event.type == pygame.MOUSEBUTTONUP:
                if button_color == DARK_BLUE:
                    paused = False
        pygame.draw.rect(screen, BLACK, [(size[0] - 4) // 2 - 100, (size[1] - 4) // 2 - 100, 200, 200])
        pygame.draw.rect(screen, WHITE, [size[0] // 2 - 100, size[1] // 2 - 100, 196, 196])
        pygame.draw.rect(screen, button_color, [(size[0] - 4) // 2 - 50, size[1] // 2 + 50, 100, 40])
        font = pygame.font.Font('freesansbold.ttf', 15)

        # create a text suface object on which text is drawn on it.
        pause_text = font.render("Paused", True, BLACK, WHITE)
        continue_text = font.render("Continue", True, BLACK, button_color)
        # create a rectangular object for the text surface object
        pause_rect = pause_text.get_rect()
        continue_rect = continue_text.get_rect()

        # set the center of the rectangular object.
        pause_rect.center = ((size[0] - 4) // 2, (size[1] - 4) // 2 - 65)
        continue_rect.center = ((size[0] - 4) // 2, (size[1] - 4) // 2 + 75)
        screen.blit(pause_text, pause_rect)
        screen.blit(continue_text, continue_rect)
        pygame.display.flip()
        pygame.display.update()


class Game:
    def __init__(self, training: bool) -> None:
        """
        Generates a blank game and initializes

        Returns:
            None
        """
        self.training: bool = training
        self.iterations: int = 0
        self.snakes = []
        self.score: int = 0
        self.fruit: Fruit = Fruit(self.snakes)
        self.snakes = []
        self.snake = Snake(blocks // 2 * 20 + 10, blocks // 2 * 20 + 10, head=True)
        self.snake.dir = Directions.UP
        self.snake.child = None
        self.snakes.append(self.snake)
        self.prev_time_first: int = 0
        self.game_time = 1 if training else 60

    def move_snake(self, change: bool) -> None:
        """
        Iterates over snake backwards and has it move

        Args:
            change (bool): If a change in direction occurred
        """
        for piece in reversed(self.snakes):
            piece.move(change_dir=change)

    def check_collisions(self) -> int:
        """
        Checks if snake has collided with borders, flower, or itself. Returns the score for calculating the loss

        Returns:
            (int) Score
        """
        first = True
        for index, snake in enumerate(self.snakes):
            if first:
                if snake.x == self.fruit.x and snake.y == self.fruit.y:
                    self.fruit = Fruit(self.snakes)
                    child = snake.create()
                    self.snakes.append(child)
                    self.score += 1
                    return 10
                if snake.x >= size[0] or snake.y >= size[0] or snake.x <= 0 or snake.y <= 0:
                    self.die()
                    return -10
            for other_index, other_snake in enumerate(self.snakes):
                if index != other_index and snake.x == other_snake.x and snake.y == other_snake.y:
                    self.die()
                    return -10
        return 0

    def animate(self) -> None:
        """
        Draws snake and flower on screen

        Returns:
            None
        """
        screen.fill(WHITE)
        font = pygame.font.Font('freesansbold.ttf', 15)
        pygame.draw.rect(screen, GREEN, [0, 0, size[0], size[1]])
        pygame.draw.rect(screen, WHITE, [2, 2, size[0] - 4, size[1] - 4])
        score_text = font.render(f"Score:{str(self.score)}", False, (0, 0, 0))
        screen.blit(score_text, (15, 15))
        iter_text = font.render(f"Iterations:{str(self.iterations)}", False, (0, 0, 0))
        screen.blit(iter_text, (size[0] - 100, 15))
        pygame.draw.rect(screen, LIGHT_RED, [self.fruit.x - 10, self.fruit.y - 10, 20, 20])
        pygame.draw.rect(screen, RED, [self.fruit.x - 8, self.fruit.y - 8, 16, 16])
        for snake in self.snakes:
            pygame.draw.rect(screen, LIGHT_GREEN, [snake.x - 10, snake.y - 10, 20, 20])
            pygame.draw.rect(screen, GREEN, [snake.x - 8, snake.y - 8, 16, 16])

    def reset(self):
        """
        Reinitialize game

        Returns:
            None
        """
        self.snakes = []
        self.score: int = 0
        self.fruit = Fruit(self.snakes)
        self.snakes = []
        self.snake = Snake(blocks // 2 * 20 + 10, blocks // 2 * 20 + 10, head=True)
        self.snake.dir = Directions.UP
        self.snake.child = None
        self.snakes.append(self.snake)

    def inc_games(self) -> None:
        self.iterations += 1

    def score(self) -> int:
        return self.score

    def set_game_time(self, time: int):
        self.game_time = time

    def die(self) -> None:
        """
        Called if a collision with itself or the wall occurs

        Waits for a user input to play again if not training, otherwise automatically starts next game.

        Returns:
            None
        """
        self.inc_games()
        if self.training:
            self.reset()
            return
        dead = True
        while dead:
            pos = pygame.mouse.get_pos()
            if (size[0] - 4) // 2 - 50 <= pos[0] <= size[1] // 2 + 50 <= pos[1] <= size[1] // 2 + 90:
                button_color = DARK_BLUE
            else:
                button_color = BLUE
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        dead = False
                elif event.type == pygame.MOUSEBUTTONUP:
                    if button_color == DARK_BLUE:
                        dead = False
            pygame.draw.rect(screen, BLACK, [(size[0] - 4) // 2 - 100, (size[1] - 4) // 2 - 100, 200, 200])
            pygame.draw.rect(screen, WHITE, [size[0] // 2 - 100, size[1] // 2 - 100, 196, 196])
            pygame.draw.rect(screen, button_color, [(size[0] - 4) // 2 - 50, size[1] // 2 + 50, 100, 40])
            font = pygame.font.Font('freesansbold.ttf', 15)
            dead_text = font.render("Dead", True, BLACK, WHITE)
            play_text = font.render("Play Again", True, BLACK, button_color)
            dead_rect = dead_text.get_rect()
            play_rect = play_text.get_rect()
            dead_rect.center = ((size[0] - 4) // 2, (size[1] - 4) // 2 - 65)
            play_rect.center = ((size[0] - 4) // 2, (size[1] - 4) // 2 + 75)
            screen.blit(dead_text, dead_rect)
            screen.blit(play_text, play_rect)
            pygame.display.flip()
            pygame.display.update()
        self.reset()

    def get_state(self) -> np.array:
        """
        Returns state of the game as a numpy array

        Returns:
            (np.array): Array of the following form:
            [danger-straight, danger-right, danger-left,
            direction-left, direction-right, direction-up, direction-down,
            food-left, food-right, food-up, food-down]
        """
        grid = np.zeros((blocks, blocks))
        head_coord = ((self.snake.x - 10) // 20, (self.snake.y - 10) // 20)
        for snake in self.snakes:
            x_index = (snake.x - 10) // 20
            y_index = (snake.y - 10) // 20
            if blocks > x_index >= 0 and blocks > y_index >= 0:
                grid[x_index, y_index] = 1
        fields = np.zeros(11, dtype=bool)
        if head_coord[0] < 20 and head_coord[1] < 20:
            if self.snake.dir == Directions.UP:
                fields[5] = True
                if head_coord[1] <= 0 or grid[head_coord[0], head_coord[1] - 1] == 1:
                    fields[0] = True
                if head_coord[0] >= blocks - 1 or grid[head_coord[0] + 1, head_coord[1]] == 1:
                    fields[1] = True
                if head_coord[0] <= 0 or grid[head_coord[0] - 1, head_coord[1]] == 1:
                    fields[2] = True
            if self.snake.dir == Directions.DOWN:
                fields[6] = True
                if head_coord[1] >= blocks - 1 or grid[head_coord[0], head_coord[1] + 1] == 1:
                    fields[0] = True
                if head_coord[0] <= 0 or grid[head_coord[0] - 1, head_coord[1]] == 1:
                    fields[1] = True
                if head_coord[0] >= blocks - 1 or grid[head_coord[0] + 1, head_coord[1]] == 1:
                    fields[2] = True
            if self.snake.dir == Directions.LEFT:
                fields[3] = True
                if head_coord[0] <= 0 or grid[head_coord[0] - 1, head_coord[1]] == 1:
                    fields[0] = True
                if head_coord[1] <= 0 or grid[head_coord[0], head_coord[1] - 1] == 1:
                    fields[1] = True
                if head_coord[1] >= blocks - 1 or grid[head_coord[0], head_coord[1] + 1] == 1:
                    fields[2] = True
            if self.snake.dir == Directions.RIGHT:
                fields[4] = True
                if head_coord[0] >= blocks - 1 or grid[head_coord[0] - 1, head_coord[1]] == 1:
                    fields[0] = True
                if head_coord[1] >= blocks - 1 or grid[head_coord[0], head_coord[1] + 1] == 1:
                    fields[1] = True
                if head_coord[1] >= 0 or grid[head_coord[0], head_coord[1] - 1] == 1:
                    fields[2] = True
        fruit_coords = ((self.fruit.x - 10) // 20, (self.fruit.y - 10) // 20)
        if fruit_coords[0] < head_coord[0]:
            fields[7] = True
        if fruit_coords[0] > head_coord[0]:
            fields[8] = True
        if fruit_coords[1] < head_coord[1]:
            fields[9] = True
        if fruit_coords[1] > head_coord[1]:
            fields[10] = True
        return fields

    def key_input(self) -> None:
        """
        Either takes user input for movement or queries the NN for the next move

        Returns:
            None
        """
        if self.training:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a and self.snake.dir != Directions.RIGHT:
                        self.snake.instruction = Directions.LEFT
                    elif event.key == pygame.K_d and self.snake.dir != Directions.LEFT:
                        self.snake.instruction = Directions.RIGHT
                    elif event.key == pygame.K_w and self.snake.dir != Directions.DOWN:
                        self.snake.instruction = Directions.UP
                    elif event.key == pygame.K_s and self.snake.dir != Directions.UP:
                        self.snake.instruction = Directions.DOWN
                    elif event.key == pygame.K_p:
                        pause()

    def set_direction(self, direction):
        if direction == TriDirections.LEFT:
            if self.snake.dir == Directions.UP:
                self.snake.instruction = Directions.LEFT
            elif self.snake.dir == Directions.RIGHT:
                self.snake.instruction = Directions.UP
            elif self.snake.dir == Directions.LEFT:
                self.snake.instruction = Directions.DOWN
            elif self.snake.dir == Directions.DOWN:
                self.snake.instruction = Directions.RIGHT
        elif direction == TriDirections.RIGHT:
            if self.snake.dir == Directions.UP:
                self.snake.instruction = Directions.RIGHT
            elif self.snake.dir == Directions.RIGHT:
                self.snake.instruction = Directions.DOWN
            elif self.snake.dir == Directions.LEFT:
                self.snake.instruction = Directions.UP
            elif self.snake.dir == Directions.DOWN:
                self.snake.instruction = Directions.LEFT

    def game_logic(self):
        while pygame.time.get_ticks() - self.prev_time_first <= self.game_time:
            pass
        prev_dir = self.snake.dir
        self.prev_time_first = pygame.time.get_ticks()
        if prev_dir != self.snake.dir:
            change = True
        else:
            change = False
        reward: int = self.check_collisions()
        self.get_state()
        self.move_snake(change)
        self.key_input()
        self.animate()
        clock.tick(60)
        pygame.display.flip()
        done = True if reward == -10 else False
        return reward, done, self.score

    def game_loop(self) -> None:
        """
        Game loop to take player input and process it

        Returns:
            None
        """
        while True:
            self.game_logic()


def main():
    game = Game(False)
    game.game_loop()


if __name__ == "__main__":
    main()

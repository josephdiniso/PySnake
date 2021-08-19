#!/usr/bin/env python3
import random
from enum import Enum
import argparse

import pygame
import gym
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


# TODO: Modularize so it can work with any size

class Directions(Enum):
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4


# PyGame Initializations
block_size: int = 20
blocks = 35
size = (blocks * block_size + 4, blocks * block_size + 4)
screen = pygame.display.set_mode(size)
clock = pygame.time.Clock()

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
        text = font.render("Paused", True, BLACK, WHITE)
        text3 = font.render("Continue", True, BLACK, button_color)
        # create a rectangular object for the text surface object
        textRect = text.get_rect()
        textRect3 = text3.get_rect()

        # set the center of the rectangular object.
        textRect.center = ((size[0] - 4) // 2, (size[1] - 4) // 2 - 65)
        textRect3.center = ((size[0] - 4) // 2, (size[1] - 4) // 2 + 75)
        screen.blit(text, textRect)
        screen.blit(text3, textRect3)
        pygame.display.flip()
        pygame.display.update()


class Game:
    def __init__(self) -> None:
        """
        Generates a blank game and initializes

        Returns:
            None
        """
        self.snakes = []
        self.score: int = 0
        self.fruit = Fruit(self.snakes)
        self.snakes = []
        self.snake = Snake(12 * 20 + 10, 12 * 20 + 10, head=True)
        self.snake.x = 12 * 20 + 10
        self.snake.y = 12 * 20 + 10
        self.snake.dir = Directions.UP
        self.snake.child = None
        self.snakes.append(self.snake)

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
        Checks if snake has collided with borders, flower, or itself. Will return the score for calculating the loss

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
        self.snake = Snake(12 * 20 + 10, 12 * 20 + 10, head=True)
        self.snake.x = 12 * 20 + 10
        self.snake.y = 12 * 20 + 10
        self.snake.dir = Directions.UP
        self.snake.child = None
        self.snakes.append(self.snake)

    def die(self) -> None:
        """
        Occurs if a collision with itself or the wall occurs

        Returns:
            None
        """
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
            text = font.render("Dead", True, BLACK, WHITE)
            text3 = font.render("Play Again", True, BLACK, button_color)
            textRect = text.get_rect()
            textRect3 = text3.get_rect()
            textRect.center = ((size[0] - 4) // 2, (size[1] - 4) // 2 - 65)
            textRect3.center = ((size[0] - 4) // 2, (size[1] - 4) // 2 + 75)
            screen.blit(text, textRect)
            screen.blit(text3, textRect3)
            pygame.display.flip()
            pygame.display.update()
        self.reset()

    def game_loop(self) -> None:
        """
        Game loop to take player input and process it

        Returns:
            None
        """
        done = False
        prev_time_first = 0
        prev_time_second = 0
        prev_dir = Directions.UP

        while not done:
            if pygame.time.get_ticks() - prev_time_first > 120:
                prev_dir = self.snake.dir
                prev_time_first = pygame.time.get_ticks()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.KEYDOWN:
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
            screen.fill(WHITE)
            font = pygame.font.Font('freesansbold.ttf', 15)
            pygame.draw.rect(screen, GREEN, [0, 0, size[0], size[1]])
            pygame.draw.rect(screen, WHITE, [2, 2, size[0] - 4, size[1] - 4])
            score_text = font.render(str(self.score), False, (0, 0, 0))
            screen.blit(score_text, (15, 15))
            self.animate()
            clock.tick(30)
            if pygame.time.get_ticks() - prev_time_second > 120:
                if prev_dir != self.snake.dir:
                    change = True
                else:
                    change = False
                score: int = self.check_collisions()
                self.move_snake(change)
                prev_time_second = pygame.time.get_ticks()
            pygame.display.flip()


def main():
    pygame.init()
    pygame.mixer.init(22050, -8, 16, 65536)
    pygame.display.set_caption("PySnake")
    game = Game()
    game.game_loop()


if __name__ == "__main__":
    main()

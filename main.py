#!/usr/bin/env python3
import random
from enum import Enum

import pygame

# PyGame Initializations
pygame.init()
pygame.mixer.init(22050, -8, 16, 65536 )
pygame.display.set_caption("PySnake")
size = (504, 504)
screen = pygame.display.set_mode(size)
clock = pygame.time.Clock()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255,255,0)
BLUE = (66, 71, 235)
ORANGE = (247, 165, 0)
LIME = (0, 247, 66)
PURPLE = (155, 46, 201)
TEAL = (0,128,128)
DARK_BLUE = (3, 86, 252)

block_size = 20
class Directions(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class Fruit:
    def __init__(self):
        self.x = random.randint(1,(size[1]-4)/block_size-1)*block_size
        self.y = random.randint(1,(size[1]-4)/block_size-1)*block_size

class Snake:
    def __init__(self):
        self.x = ((size[1]-4)/2)-10
        self.y = self.x
        self.dir = Directions.UP
    
    
    def move(self):
        print('here')
        if self.dir == Directions.UP:
            self.y -= 20
        elif self.dir == Directions.RIGHT:
            self.x += 20
        elif self.dir == Directions.DOWN:
            self.y += 20
        elif self.dir == Directions.LEFT:
            self.x -= 20

def check_collisions(fruit, snake):
    if snake.x == fruit.x and snake.y == fruit.y:
        fruit = Fruit()
    return fruit

def draw_chars(fruit, snake):
    pygame.draw.rect(screen, RED, [fruit.x-10,fruit.y-10,20,20])
    pygame.draw.rect(screen, GREEN, [snake.x-10,snake.y-10,20,20])

def main():
    fruit = Fruit()
    snake = Snake()
    done = False
    prev_time = 0
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    snake.dir = Directions.LEFT
                elif event.key == pygame.K_d:
                    snake.dir = Directions.RIGHT
                elif event.key == pygame.K_w:
                    snake.dir = Directions.UP
                elif event.key == pygame.K_s:
                    snake.dir = Directions.DOWN
        screen.fill(WHITE)
        pygame.draw.rect(screen, GREEN, [0,0,size[0],size[1]])
        pygame.draw.rect(screen, WHITE, [3,3,size[0]-6,size[1]-6])
        draw_chars(fruit, snake)
        clock.tick(30)
        if pygame.time.get_ticks() - prev_time > 120:
            snake.move()
            fruit = check_collisions(fruit,snake)
            prev_time = pygame.time.get_ticks()
        pygame.display.flip()

if __name__ == "__main__":
    main()
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

snakes = []
block_size = 20
class Directions(Enum):
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4

class Fruit:
    def __init__(self):
        self.x = random.randint(1,24)*block_size+10
        self.y = random.randint(1,24)*block_size+10
        snake_pos = [(snake.x,snake.y) for snake in snakes]
        while (self.x,self.y) in snake_pos:
            self.x = random.randint(1,24)*block_size+10
            self.y = random.randint(1,24)*block_size+10           
            


class Snake:
    def __init__(self, x, y, direction=Directions.UP, head=False):
        if direction == Directions.UP:
            self.x = x
            self.y = y+20
        elif direction == Directions.RIGHT:
            self.x = x-20
            self.y = y
        elif direction == Directions.DOWN:
            self.x = x
            self.y = y-20
        elif direction == Directions.LEFT:
            self.x = x+20
            self.y = y
        self.dir = direction
        self.instruction = None
        self.child = None
        self.head = head
    
    
    def move(self, change_dir = False):
        if self.instruction:
            self.dir = self.instruction
            if self.child:
                self.child.instruction = self.dir
            self.instruction = None
        if self.dir == Directions.UP:
            self.y -= 20
        elif self.dir == Directions.RIGHT:
            self.x += 20
        elif self.dir == Directions.DOWN:
            self.y += 20
        elif self.dir == Directions.LEFT:
            self.x -= 20
        if change_dir and self.child and self.head:
            self.child.instruction = self.dir

    
    def create(self):
        obj = self
        while obj.child:
            obj = obj.child
        obj.child = Snake(obj.x, obj.y, direction=obj.dir)
        snakes.append(obj.child)


def check_collisions(fruit, snake):
    first = True
    for index, snake in enumerate(snakes):
        if first:
            if snake.x == fruit.x and snake.y == fruit.y:
                fruit = Fruit()
                snake.create()
            if snake.x >= size[0] or snake.y >= size[0] or snake.x <= 0 or snake.y <= 0:
                die()
            first = False
        for other_index, other_snake in enumerate(snakes):
            if index != other_index and snake.x == other_snake.x and snake.y == other_snake.y:
                die()
        return fruit

def draw_chars(fruit):
    pygame.draw.rect(screen, RED, [fruit.x-10,fruit.y-10,20,20])
    for snake in snakes:
        pygame.draw.rect(screen, GREEN, [snake.x-10,snake.y-10,20,20])


def die():
    dead = True
    while dead:
        pos = pygame.mouse.get_pos()
        if pos[0] >= 200 and pos[0] <= 300 and pos[1] >= 300 and pos[1] <= 340:
            button_color = DARK_BLUE
        else:
            button_color = BLUE
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    dead=False
            elif event.type == pygame.MOUSEBUTTONUP:
                if button_color == DARK_BLUE:
                    dead = False
        pygame.draw.rect(screen, BLACK, [150,200,200,200])
        pygame.draw.rect(screen, WHITE, [152,202,196,196])
        pygame.draw.rect(screen, button_color, [200,300, 100, 40])
        font = pygame.font.Font('freesansbold.ttf', 15) 

        # create a text suface object, 
        # on which text is drawn on it. 
        text = font.render("Dead", True, BLACK, WHITE)
        text3 = font.render("Play Again", True, BLACK, button_color)
        # create a rectangular object for the 
        # text surface object 
        textRect = text.get_rect()
        textRect3 = text3.get_rect()

        # set the center of the rectangular object. 
        textRect.center = (250, 220) 
        textRect3.center = (250, 320)
        screen.blit(text,textRect)
        screen.blit(text3, textRect3)
        pygame.display.flip()
        pygame.display.update()
        clock.tick(30)
    init_game()


def pause():
    paused = True
    while paused:
        pos = pygame.mouse.get_pos()
        if pos[0] >= 200 and pos[0] <= 300 and pos[1] >= 300 and pos[1] <= 340:
            button_color = DARK_BLUE
        else:
            button_color = BLUE
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    paused=False
            elif event.type == pygame.MOUSEBUTTONUP:
                if button_color == DARK_BLUE:
                    paused = False
        pygame.draw.rect(screen, BLACK, [150,200,200,200])
        pygame.draw.rect(screen, WHITE, [152,202,196,196])
        pygame.draw.rect(screen, button_color, [200,300, 100, 40])
        font = pygame.font.Font('freesansbold.ttf', 15) 

        # create a text suface object, 
        # on which text is drawn on it. 
        text = font.render("Paused", True, BLACK, WHITE)
        text3 = font.render("Continue", True, BLACK, button_color)
        # create a rectangular object for the 
        # text surface object 
        textRect = text.get_rect()
        textRect3 = text3.get_rect()

        # set the center of the rectangular object. 
        textRect.center = (250, 220) 
        textRect3.center = (250, 320)
        screen.blit(text,textRect)
        screen.blit(text3, textRect3)
        pygame.display.flip()
        pygame.display.update()
        clock.tick(30)


def init_game():
    global fruit
    fruit = Fruit()
    global snakes
    snakes = []
    snake.x = 12*20+10
    snake.y = 12*20+10
    snake.dir = Directions.UP
    snake.child = None
    snakes.append(snake)
    global done, prev_time_first, prev_time_second, prev_dir
    done = False
    prev_time_first = 0
    prev_time_second = 0
    prev_dir = Directions.UP


def main():
    fruit = Fruit()
    global snake
    snake = Snake(12*20+10, 12*20+10, head=True)
    snakes.append(snake)
    done = False
    prev_time_first = 0
    prev_time_second = 0
    prev_dir = Directions.UP
    while not done:
        if pygame.time.get_ticks() - prev_time_first > 120: 
            prev_dir = snake.dir
            prev_time_first = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a and snake.dir != Directions.RIGHT:
                    snake.dir = Directions.LEFT
                elif event.key == pygame.K_d and snake.dir != Directions.LEFT:
                    snake.dir = Directions.RIGHT
                elif event.key == pygame.K_w and snake.dir != Directions.DOWN:
                    snake.dir = Directions.UP
                elif event.key == pygame.K_s and snake.dir != Directions.UP:
                    snake.dir = Directions.DOWN
                elif event.key == pygame.K_p:
                    pause()
        screen.fill(WHITE)
        pygame.draw.rect(screen, GREEN, [0,0,size[0],size[1]])
        pygame.draw.rect(screen, WHITE, [2,2,size[0]-4,size[1]-4])
        draw_chars(fruit)
        clock.tick(30)
        if pygame.time.get_ticks() - prev_time_second > 120:
            if prev_dir != snake.dir:
                change = True
            else:
                change = False
            for snake_obj in reversed(snakes):
                snake_obj.move(change_dir=change)
            fruit = check_collisions(fruit,snake)
            prev_time_second = pygame.time.get_ticks()
        pygame.display.flip()

if __name__ == "__main__":
    main()
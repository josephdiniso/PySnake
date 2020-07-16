#!/usr/bin/env python3
import random
from enum import Enum
from collections import namedtuple
import argparse

import pygame
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from network import NeuralNetwork, ReplayMemory

class Directions(Enum):
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# PyGame Initializations
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
block_size: int = 20
score: int = 0
fruit = None
args = None
reward: int = 0
actions = (Directions.LEFT, Directions.RIGHT, Directions.UP, Directions.DOWN, None, None, None, None, None, None, None, None)
actions_predict = (Directions.LEFT, Directions.RIGHT, Directions.UP, Directions.DOWN, None)

# Training variables
GAMMA = 0.999
gamma_rate = 0.05
screen_vals = np.zeros((625))
model = NeuralNetwork().float()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
memory = ReplayMemory(10000)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")




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


    
    def create(self):
        obj = self
        while obj.child:
            obj = obj.child
        obj.child = Snake(obj.x, obj.y, direction=obj.dir)
        snakes.append(obj.child)


def check_collisions(fruit, snake):
    first = True
    global score, reward
    for index, snake in enumerate(snakes):
        if first:
            if snake.x == fruit.x and snake.y == fruit.y:
                fruit = Fruit()
                snake.create()
                score += 1
                reward = 10
            if snake.x >= size[0] or snake.y >= size[0] or snake.x <= 0 or snake.y <= 0:
                reward = -10
                die()
            first = False
        for other_index, other_snake in enumerate(snakes):
            if index != other_index and snake.x == other_snake.x and snake.y == other_snake.y:
                reward = -10
                die()
        return fruit


def draw_chars(fruit):
    pygame.draw.rect(screen, RED, [fruit.x-10,fruit.y-10,20,20])
    for snake in snakes:
        pygame.draw.rect(screen, GREEN, [snake.x-10,snake.y-10,20,20])


def die():
    dead = True
    global GAMMA
    while dead:
        if args.training:
            break
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
        text = font.render("Dead", True, BLACK, WHITE)
        text3 = font.render("Play Again", True, BLACK, button_color)
        textRect = text.get_rect()
        textRect3 = text3.get_rect()
        textRect.center = (250, 220) 
        textRect3.center = (250, 320)
        screen.blit(text,textRect)
        screen.blit(text3, textRect3)
        pygame.display.flip()
        pygame.display.update()
    GAMMA -= 0.1
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


def init_game():
    global fruit
    global snakes
    global score
    # global gamma
    # gamma -= gamma_rate
    fruit = Fruit()
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
    score = 0


def array_screen():
    global screen_vals
    head_index = int((snakes[0].x-10)/20+(snakes[0].y-10)/20*25)
    fruit_index = int((fruit.x-10)/20+(fruit.y-10)/20*25)
    screen_vals[head_index] = 2
    screen_vals[fruit_index] = 3
    for snake in snakes[1:]:
        snake_index = int((snake.x-12)/20+(snake.y-12)/20*25)
        screen_vals[snake_index] = 1
    return torch.from_numpy(screen_vals)


def select_action(state, prev_dir):
    if random.random() < GAMMA:
        direction = random.choice(actions)
        if direction == None:
            direction = prev_dir
        return direction
    else:
        return actions_predict[model(state).max(1)[1].view(1,1)]


def optimize_model():
    global memory
    if len(memory) < 10:
        return
    transitions = memory.sample(10)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)
    state_batch = state_batch.view(state_batch.size(0), -1).float()
    state_action_values = model(state_batch.float())

    next_state_values = torch.zeros(10, device=device)
    next_state_values[non_final_mask] = model(non_final_next_states.float()).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = 2

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def main():
    pygame.init()
    pygame.mixer.init(22050, -8, 16, 65536 )
    pygame.display.set_caption("PySnake")
    global args, fruit

    parser = argparse.ArgumentParser(description="Controls for PySnake")
    parser.add_argument("--training", action="store_true", help="Add if training")
    args = parser.parse_args()

    fruit = Fruit()
    global snake, reward
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
            if args.training:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                snake.dir = select_action(array_screen(), snake.dir)
                if snake.dir == None:
                    snake.dir = prev_dir
                if reward:
                    reward = 0
        if not args.training:
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
        font = pygame.font.Font('freesansbold.ttf', 15) 
        pygame.draw.rect(screen, GREEN, [0,0,size[0],size[1]])
        pygame.draw.rect(screen, WHITE, [2,2,size[0]-4,size[1]-4])
        score_text = font.render(str(score), False, (0, 0, 0))
        screen.blit(score_text,(15,15))
        draw_chars(fruit)
        clock.tick(30)

        if pygame.time.get_ticks() - prev_time_second > 120:
            prev_state = array_screen()
            if prev_dir != snake.dir:
                change = True
            else:
                change = False
            for snake_obj in reversed(snakes):
                snake_obj.move(change_dir=change)
            fruit = check_collisions(fruit,snake)
            prev_time_second = pygame.time.get_ticks()
            if change:
                action = actions_predict.index(snake.dir)
            else:
                action = 4
            memory.push(prev_state, torch.tensor(action), array_screen(), torch.tensor(reward))

            optimize_model()
            
        pygame.display.flip()

if __name__ == "__main__":
    main()
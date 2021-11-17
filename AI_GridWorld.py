# Grid World: AI-controlled play

# Instructions:
#   Move up, down, left, or right to move the character. The 
#   objective is to find the key and get to the door
#
# Control:
#    arrows  : Merge up, down, left, or right
#    s       : Toggle slow play
#    a       : Toggle AI player
#    d       : Toggle rendering 
#    r       : Restart game
#    q / ESC : Quit

from GridWorld import GridWorld
import numpy as np
import pygame
from collections import defaultdict
from random import randint
import time

# Definitions and default settings
actions = ['left', 'right', 'up', 'down']
exit_program = False
action_taken = False
slow = False
runai = True
render = False
done = False

# Game clock
clock = pygame.time.Clock()

# INSERT YOUR CODE HERE (1/2)
# Define data structure for q-table
Q = defaultdict(lambda: [0.,0.,0.,0.])
gammas = [
    0.7,
    0.8,
    0.9]
gammaIdx = 0
previous_reward = 0
previous_x = 0
previous_y = 0

boards = [
    "Mazes/40-maze.txt",
    "Mazes/30-maze.txt",
    "Mazes/20-maze.txt",
    "Mazes/10-maze.txt",
    "Mazes/8-maze.txt",
]
boardIdx = 0

wins_in_a_row = 0
n = 385
iterations = 0
run_iterations = []
runs = 0
run_start = time.time()

results = np.zeros((len(boards), len(gammas), 3))

def prepare_next_run():
    global runs, iterations, boardIdx, gammaIdx, exit_program, results, run_start

    runs += 1
    run_iterations.append(iterations)

    print(f"{boards[boardIdx]} ({gammas[gammaIdx]}) RUN: ", runs)

    if runs == n:
        runs = 0
        
        results[boardIdx, gammaIdx, 0] = np.mean(run_iterations)
        results[boardIdx, gammaIdx, 1] = np.std(run_iterations)
        results[boardIdx, gammaIdx, 2] = (time.time() - run_start)/n
        result_block = results[boardIdx, gammaIdx]
        print(f"{boards[boardIdx]} ({gammas[gammaIdx]}) -->  M = {result_block[0]}, STD: {result_block[1]}, Avg. time: {result_block[2]}")

        run_iterations.clear()
        run_start = time.time()


        if gammaIdx != len(gammas) - 1:
            gammaIdx += 1
        elif boardIdx != len(boards) - 1:
            boardIdx += 1
            gammaIdx = 0
        else:
            exit_program = True
            print_results(results)

    Q.clear()
    iterations = 0
        
def print_results(results):
    print("################################################## DONE ##################################################")
    for bi, board in enumerate(boards):
        for gi, gamma in enumerate(gammas):
            result_block = results[bi,gi]
            print(f"Board: {board}, Gamma: {gamma} --> M = {result_block[0]}, STD: {result_block[1]}, Avg. time: {result_block[2]}")

# END OF YOUR CODE (1/2)

# Initialize the environment
env = GridWorld(boards[boardIdx])
env.reset(boards[boardIdx])
x, y, has_key = env.get_state()

while not exit_program:
    if render:
        env.render()
    
    # Slow down rendering to 5 fps
    if slow and runai:
        clock.tick(5)
        
    # Automatic reset environment in AI mode
    if done and runai:
        iterations += 1

        if env.won(x, y, has_key, env.board):
            wins_in_a_row += 1

            if wins_in_a_row == 5:
                prepare_next_run()
        else:
            wins_in_a_row = 0

        env.reset(boards[boardIdx])
        x, y, has_key = env.get_state()
        previous_x = x
        previous_y = y
        
    # Process game events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit_program = True
        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                exit_program = True
            if event.key == pygame.K_UP:
                action, action_taken = 'up', True
            if event.key == pygame.K_DOWN:
                action, action_taken  = 'down', True
            if event.key == pygame.K_RIGHT:
                action, action_taken  = 'right', True
            if event.key == pygame.K_LEFT:
                action, action_taken  = 'left', True
            if event.key == pygame.K_r:
                env.reset()   
            if event.key == pygame.K_d:
                render = not render
            if event.key == pygame.K_s:
                slow = not slow
            if event.key == pygame.K_a:
                runai = not runai
                clock.tick(5)
    
    # AI controller (enable/disable by pressing 'a')
    if runai:
        # INSERT YOUR CODE HERE (2/2)
        #
        # Implement a Grid World AI (q-learning): Control the person by 
        # learning the optimal actions through trial and error
        #
        # The state of the environment is available in the variables
        #    x, y     : Coordinates of the person (integers 0-9)
        #    has_key  : Has key or not (boolean)
        #
        # To take an action in the environment, use the call
        #    (x, y, has_key), reward, done = env.step(action)
        #
        #    This gives you an updated state and reward as well as a Boolean 
        #    done indicating if the game is finished. When the AI is running, 
        #    the game restarts if done=True

        # 1. choose an action
        q_current = Q[(x,y,has_key)]

        if randint(1,env.board.shape[0]**2) == 1:
            action_num = randint(0,3)
            action = actions[action_num]
        else:
            action_num =  np.argmax(q_current)
            action = actions[action_num]

        # 2. step the environment
        (x, y, has_key), reward, done = env.step(action)

        # 3. update q table
        if x == previous_x and y == previous_y:
            reward = -100
        q_next = Q[(x, y, has_key)]
        q_current[action_num] = reward + gammas[gammaIdx]*np.max(q_next)

        previous_reward = reward
        previous_x = x
        previous_y = y

        # END OF YOUR CODE (2/2)
    
    # Human controller        
    else:
        if action_taken:
            (x, y, has_key), reward, done = env.step(action)
            action_taken = False


env.close()

import pygame
from Graphics import *
from TicTacToe import TicTacToe
from State import State
from Human_Agent import Human_Agent
from Random_Agent import Random_Agent
from DQN_Agent_2 import DQN_Agent

PATH = 'Data\DQN_PARAM_9_20K.pth'

pygame.init()
clock = pygame.time.Clock()
graphics = Graphics()
env = TicTacToe(State())
# player1 = Random_Agent(player=1, env=env, graphics=graphics)
player1 = DQN_Agent(player=1,parametes_path=PATH, train=False, env=env)
# player2 = Random_Agent(player=-1, env=env, graphics=graphics)
player2 = Human_Agent(-1,env=env, graphics=graphics)

def main ():
    player = player1
    run = True
    
    while (run):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
               run = False
        action = player(events, env.state)
        if action:
            env.move(action)
            player = switch_players(player)
            # pygame.time.wait(500)
            if env.state.end_of_game != 0:
                graphics(env.state)
                pygame.time.wait(2000)            
                env.state.reset()
                player = player1
                # run = False
                
        graphics(env.state)
        clock.tick(FPS)
    
def switch_players(player):
    if player == player1:
        return player2
    else:
        return player1

if __name__ == '__main__':
    main()

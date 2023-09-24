from TicTacToe import TicTacToe
from State import State
from Human_Agent import Human_Agent
from Random_Agent import Random_Agent
from DQN_Agent import DQN_Agent

PATH = 'Data\DQN_PARAM_7_10K.pth'
# PATH=None
env = TicTacToe(State())
player1 = DQN_Agent(1, env=env, parametes_path=PATH, train=False)
# player1 = Random_Agent(1, env,graphics=None)
player2 = Random_Agent(-1, env,graphics=None)
num = 1000

def main ():

    x_win = 0
    o_win = 0
    tie = 0
        
    for n in range(num):
        state = State()
        player = player1
        while not env.end_of_game(state):
            action = player.get_action(state=state)
            state, _ = env.next_state(state,action)
            player = switch_players(player)
        if state.end_of_game == 1:
            x_win +=1
        elif state.end_of_game == -1:
            o_win += 1
        else:
            tie +=1
        state.reset()    
        print(n, end = "\r")
    print()
    print(x_win, o_win, tie) 

def switch_players(player):
    if player == player1:
        return player2
    else:
        return player1

if __name__ == '__main__':
    main()
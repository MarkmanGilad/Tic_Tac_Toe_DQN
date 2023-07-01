from TicTacToe import TicTacToe
from State import State
from Human_Agent import Human_Agent
from Random_Agent import Random_Agent
from AI_Agent import AI_Agent

PATH = 'Data/Q_SARSA_4.pth'
env = TicTacToe(State())

player1 = AI_Agent(1, env, graphics=None, Q_table_PATH=None)
Q = player1.Q
get_Q = player1.get_Q
gamma = 0.9
alpha = 0.1

player2 = Random_Agent(-1, env,graphics=None)

def main ():
    player = player1    
    epochs = 100000
    
    for epoch in range(epochs):
        state = State()
        action = player1.get_action(state, epoch)   #using e-greedy
        while not env.end_of_game(state): 
            afterState, reward = env.next_state(state, action)
            if env.end_of_game(afterState):
                Q[(state, action)] =  get_Q(state, action) + alpha * (reward + gamma * 0 - get_Q(state, action))
                # Q[(state, action)] = reward
                break
            afterAction = player2.get_action(state=afterState)
            next_state, reward = env.next_state(afterState, afterAction)
            if env.end_of_game(next_state):
                Q[(state, action)] =  get_Q(state, action) + alpha * (reward + gamma * 0 - get_Q(state, action))
                # Q[(state, action)] = reward
                break
            next_action = player1.get_action(next_state, epoch)   #using e-greedy
            Q[(state, action)] =  get_Q(state, action) + alpha * (reward + gamma * get_Q(next_state, next_action) - get_Q(state, action))
            state = next_state
            action = next_action
        
        print(epoch, end="\r")
    
    player1.save_Q(PATH)
    print(test(100))

def test (num):
    x_win = 0
    o_win = 0
    tie = 0
    player = player1
    player.train=False
    player.load_Q(PATH)
    for n in range(num):
        player = player1
        state = State()
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
    return x_win, o_win, tie

def switch_players(player):
    if player == player1:
        return player2
    else:
        return player1

if __name__ == '__main__':
    main()
    # print(test(100))
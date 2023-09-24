from DQN_2 import DQN
from DQN_Agent_2 import DQN_Agent       #one layer of 64
from Random_Agent import Random_Agent
from TicTacToe import TicTacToe
from ReplayBuffer import ReplayBuffer
from State import State
import torch 

epochs = 30000
C = 1000
batch = 64
learning_rate = 0.1
path = "Data\DQN_PARAM_9_20K.pth"
replay_path = "Data\Replay_9_20K.pth"
def main ():
    env = TicTacToe()
    player1 = DQN_Agent(1, env=env)
    player1_hat = DQN_Agent(1, env=env,train=False)
    Q = player1.DQN
    Q_hat :DQN = Q.copy()
    Q_hat.train = False
    player1_hat.DQN = Q_hat

    player2 = Random_Agent(-1, env=env)
    replay = ReplayBuffer()
    optim = torch.optim.SGD(Q.parameters(), lr=learning_rate)
       
    for epoch in range(epochs):
        print (epoch, end="\r")
        state = State()
        while not env.end_of_game(state):
            action = player1.get_action(state, epoch=epoch)
            after_state, reward = env.next_state(state, action)
            done = env.end_of_game(after_state)
            if done:
                replay.push(state, action, reward, after_state, done)
                break
            after_action = player2.get_action(state=after_state)
            next_state, reward = env.next_state(after_state, after_action)
            replay.push(state, action, reward, next_state, env.end_of_game(next_state))
            state = next_state

            if epoch < batch:
                continue
            states, actions, rewards, next_states, dones = replay.sample(batch)
            Q_values = Q(states, actions)
            next_actions = player1_hat.get_actions(next_states, dones) #fix bug
            with torch.no_grad():
                Q_hat_Values = Q_hat(next_states, next_actions)
            
            loss = Q.loss(Q_values, rewards, Q_hat_Values, dones)
            loss.backward()
            optim.step()
            optim.zero_grad()
        if epoch % C == 0:
            Q_hat.load_state_dict(Q.state_dict())

    player1.save_param(path)
    torch.save(replay, replay_path)
if __name__ == '__main__':
    main()
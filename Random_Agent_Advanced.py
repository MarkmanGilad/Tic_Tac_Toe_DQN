from TicTacToe import TicTacToe
import pygame
import numpy as np
from Graphics import *
import random
import matplotlib.pyplot as plt
import torch

class Random_Agent_Advanced:
    def __init__(self, player, env: TicTacToe, graphics: Graphics = None):
        self.env = env
        self.player = player
        self.graphics = graphics

    def get_action(self, events=None, state = None, epoch=100, train=False, random_step = 5):
        if state is None:
            state = self.env.state
        board = state.board
        win_action = self.find_winning_action(board=board, player=self.player)
        block_action = self.find_winning_action(board=board, player=-self.player)
        if epoch % random_step == 0:
            if win_action:
                return win_action
            if block_action :
                return block_action
        indices = np.where(board == 0)
        actions = list(zip(indices[0], indices[1]))
        action = random.choice(actions)
        return action

    def get_state_action (self, state, epoch = None):
        action = self.get_action(state=state)
        next_state, reward = self.env.next_state(state, action)
        return action, reward, next_state

    def find_winning_action(self, board, player):
        """
        Find a winning move for the given player on a Tic Tac Toe board using NumPy operations.
        
        Args:
            board (numpy.ndarray): A 3x3 numpy array representing the board.
                                0: empty, 1: X, -1: O
            player (int): The player number (1 for X, -1 for O).
        
        Returns:
            tuple: (row, col) of the winning move, or None if no such move exists.
        """
        # Check rows
        row_sums = np.sum(board, axis=1)
        for row in np.where(row_sums == 2 * player)[0]:
            col = np.where(board[row, :] == 0)[0]
            if col.size > 0:
                return row.item(), col[0].item()
        
        # Check columns
        col_sums = np.sum(board, axis=0)
        for col in np.where(col_sums == 2 * player)[0]:
            row = np.where(board[:, col] == 0)[0]
            if row.size > 0:
                return row[0].item(), col.item()

        # Check main diagonal
        main_diag = board.diagonal()
        if np.sum(main_diag) == 2 * player:
            pos = np.where(main_diag == 0)[0]
            if pos.size > 0:
                return pos[0].item(), pos[0].item()

        # Check anti-diagonal
        anti_diag = np.fliplr(board).diagonal()
        if np.sum(anti_diag) == 2 * player:
            pos = np.where(anti_diag == 0)[0]
            if pos.size > 0:
                return pos[0].item(), 2 - pos[0].item()

        # No winning move found
        return None
    
    def plot_normalized_results(self, results, figure=1, alpha  = 0.3):
        # delete first zeros
        results.pop(0)

        # Transpose the data to get columns as separate lists
        columns = list(zip(*results))  # [(1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 6)]

        columns[0] = self.exponential_moving_average(columns[0], alpha)
        columns[1] = self.exponential_moving_average(columns[1], alpha)
        columns[2] = self.exponential_moving_average(columns[2], alpha)

        # Plot each column
        x = range(len(results))  # X-axis values (row indices)
        
        plt.figure(figure)
        plt.plot(x, columns[0], label="wins")
        plt.plot(x, columns[1], label="losses")
        plt.plot(x, columns[2], label="ties")

        # Add legend
        plt.legend()
        
        # plt.plot(results)
        plt.xlabel("Games (100)")
        plt.ylabel("Results")
        plt.title("Results for every 100 Games")
    
    def exponential_moving_average(self, data, alpha=0.3):
        ema = [data[0]]  # Start with the first value
        for value in data[1:]:
            ema.append(alpha * value + (1 - alpha) * ema[-1])
        return ema


    def __call__(self, events=None, state=None):
        return self.get_action(events=events, state = state)

if __name__ == '__main__':
    player = Random_Agent_Advanced(player=-1, env=None)
    # board = np.array([
    #     [1, 1, -1],
    #     [1, -1, -1],
    #     [0, 0, 1]
    # ])

    # print(player.find_winning_action(board))
    results = torch.load(f"Data/results_2.pth")
    player.plot_normalized_results(results,alpha=0.1)
    plt.show()
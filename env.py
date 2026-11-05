import numpy as np

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.is_done = False
        self.winner = None

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.is_done = False
        self.winner = None
        return self.get_state()

    def get_state(self):
        return tuple(self.board.flatten())

    def get_available_actions(self):
        flat_board = self.board.flatten()
        return [i for i, val in enumerate(flat_board) if val == 0]

    def check_winner(self):
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3:
                return self.board[i, 0]
            if abs(sum(self.board[:, i])) == 3:
                return self.board[0, i]

        diag1 = sum([self.board[i, i] for i in range(3)])
        diag2 = sum([self.board[i, 2 - i] for i in range(3)])
        if abs(diag1) == 3 or abs(diag2) == 3:
            return self.board[1, 1]

        if len(self.get_available_actions()) == 0:
            return 0

        return None

    def step(self, action):
        if self.is_done:
            raise ValueError("Game is already over.")

        row, col = divmod(action, 3)

        if self.board[row, col] != 0:
            raise ValueError(f"Action {action} is invalid. Cell already taken.")

        self.board[row, col] = self.current_player
        result = self.check_winner()

        if result is not None:
            self.is_done = True
            self.winner = result
            if self.winner == self.current_player:
                reward = 1.0
            elif self.winner == 0:
                reward = 0.5
            else:
                reward = -1.0
        else:
            reward = 0.0

        self.current_player = -self.current_player
        return self.get_state(), reward, self.is_done, self.winner

    def render(self):
        symbols = {1: 'X', -1: 'O', 0: ' '}
        for row in self.board:
            print('| ' + ' | '.join(symbols[val] for val in row) + ' |')
        print("-" * 13)
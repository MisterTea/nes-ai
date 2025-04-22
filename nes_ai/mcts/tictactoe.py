from nes_ai.mcts.mcts import GameState


class TicTacToeState(GameState):
    def __init__(self, board=None, player=1):
        self.board = board or [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.player = player  # 1 for X, -1 for O

    def get_current_team(self):
        return self.player

    def get_legal_moves(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == 0]

    def make_move(self, move):
        board = [list(row) for row in self.board]  # Make a copy of the board.
        board[move[0]][move[1]] = self.player
        return TicTacToeState(board, -self.player)

    def is_terminal(self):
        for i in range(3):
            if (
                abs(sum(self.board[i])) == 3
                or abs(sum([row[i] for row in self.board])) == 3
            ):
                return True
        if (
            abs(sum(self.board[i][i] for i in range(3))) == 3
            or abs(sum(self.board[i][2 - i] for i in range(3))) == 3
        ):
            return True
        if all(self.board[i][j] != 0 for i in range(3) for j in range(3)):
            return True
        return False

    def get_reward(self):
        win_sum = (
            -self.player * 3
        )  # "negative player" because we're finding the reward relative to the previous player (the one who made the terminal move)
        for i in range(3):
            if (
                sum(self.board[i]) == win_sum
                or sum(row[i] for row in self.board) == win_sum
            ):
                return 1
        if (
            sum(self.board[i][i] for i in range(3)) == win_sum
            or sum(self.board[i][2 - i] for i in range(3)) == win_sum
        ):
            return 1
        return 0

    def __repr__(self):
        return "\n".join(
            "".join({1: "X", -1: "O", 0: "-"}[val] for val in row) for row in self.board
        )

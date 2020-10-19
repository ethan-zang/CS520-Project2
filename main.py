import random
from typing import List


class Minesweeper:

    def __init__(self, d, n):
        self._d = d
        self._n = n
        self.environment = self.make_board()

    def make_board(self) -> List[List[int]]:
        board = [[0 for _ in range(self._d)] for _ in range(self._d)]
        for _ in range(self._n):
            i = random.randint(0, self._d-1)
            j = random.randint(0, self._d-1)
            while board[i][j] == 9:
                i = random.randint(0, self._d - 1)
                j = random.randint(0, self._d - 1)
            board[i][j] = 9
            for x in range(-1, 2):
                for y in range(-1, 2):
                    if x != 0 or y != 0:
                        if 0 <= i+x < self._d and 0 <= j + y < self._d and board[i + x][j + y] != 9:
                            board[i + x][j + y] += 1
        return board

    def print_board(self) -> None:
        for row in range(self._d):
            print(self.environment[row])


def main(d: int, n: int):
    print('Hello world')
    minesweeper = Minesweeper(d, n)
    minesweeper.print_board()


if __name__ == '__main__':
    main(10,12)

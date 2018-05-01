"""
This program solves a randomly generated nxn sliding puzzle using bfs and A* search algorithms and returns the average
time.

Authors: Hannah Gray, Thomas Shogren, Linh Vo
"""


import random
import numpy as np
import time
import signal


class Puzzle:
    """This class represents the puzzles, their characteristics, and their functions.
        n is the dimensions of the puzzle.
        Zero represents the empty spot in the puzzle where neighbor tiles can be moved to."""

    def __init__(self, n):
        self.distance = 0  # distance away from solution
        self.size = n  # size of puzzle
        self.height = 0  # depth of the puzzle in the search algorithm
        self.previous = None  # parent puzzle
        self.matrix = createSolution(n)  # 3x3 array of the matrix
        self.solution = createSolution(n)  # solution matrix

    def randomize(self, numMoves):
        # Function that shuffles a solution to get a random puzzle.
        # As numMoves increases,the difficulty of the puzzles tend to increase.
        # If numMoves is 100, most of the puzzles take too long to solve.

        for i in range(0, numMoves):
            moves = self.getPossibleMoves()
            tile = random.choice(moves)
            self.swapTiles(tile)

    def getPossibleMoves(self):
        # Function that returns the row and column of the possible moves for a puzzle.

        possibleMoves = []
        row, col = self.findValue(0)
        if row > 0:
            possibleMoves.append((row - 1, col))
        if row < (self.size - 1):
            possibleMoves.append((row + 1, col))
        if col > 0:
            possibleMoves.append((row, col - 1))
        if col < (self.size - 1):
            possibleMoves.append((row, col + 1))
        return possibleMoves

    def findValue(self, value):
        # Finds the row and column of the value in the puzzle

        for row in range(self.size):
            for col in range(self.size):
                if self.matrix[row][col] == value:
                    return row, col

    def getValue(self, row, col):
        # Function that returns the value with the specified row and column.
        return self.matrix[row][col]

    def setValue(self, row, col, value):
        # Function that sets a given value at a specified row and column.
        self.matrix[row][col] = value

    def swapTiles(self, tile):
        # Function that swaps two tiles. Tile is the row and column of the tile we are trying to swap.

        zero = self.findValue(0)
        self.setValue(zero[0], zero[1], self.getValue(*tile))
        self.setValue(tile[0], tile[1], 0)

    def findDistance(self, solution):
        # Finds the total distance that a puzzle state is away from the solution.
        # Distance is defined as the sum of the number of rows and columns that a tile is away from its solved state.

        distance = 0
        for i in range(0,self.size):
            for j in range(0,self.size):
                value = solution[i][j]
                if value != 0:
                    row, col = self.findValue(value)
                    score = abs(row-i)+abs(col-j)
                    distance = distance + score
        return distance

    def findPath(self, path):
        # Function that returns the path that solved the puzzle.

        if self.previous is None:
            return path
        else:
            path.append(self)
            return self.previous.findPath(path)

    def copy(self):
        # Function that copies a puzzle.

        puzzle = Puzzle(self.size)
        for i in range(0,self.size):
            puzzle.matrix[i] = self.matrix[i][:]
        return puzzle

    def createMoves(self):
        # Function that returns a dictionary.
        # Keys are the puzzle state and the values are the possible puzzle states that can be obtained by making one move

        moves = self.getPossibleMoves()
        map = {}
        for move in moves:
            map[self.swapAndCopy(move)] = moves
        return map

    def swapAndCopy(self, b):
        # Function that swaps tile b and then updates the height and previous puzzle and returns the new puzzle.

        p = self.copy()
        p.swapTiles(b)
        p.height = self.height + 1
        p.previous = self
        return p

    def a_star_solve(self):
        # Function that solves a puzzle using A* algorithm.

        openList = [self]  # contains all nodes that need to be considered
        closedList = []  # the nodes that have already been visited
        while len(openList) > 0:
            puzzle = openList.pop(0)
            if np.all(puzzle.matrix == puzzle.solution):
                if len(closedList) > 0:
                    return puzzle.findPath([])
                else:
                    return []

            children = puzzle.createMoves()

            for child in children:
                openIndex = index(child, openList)
                closedIndex = index(child, closedList)
                distance = child.findDistance(child.solution)
                value = distance + child.height
                if closedIndex == -1 and openIndex == -1:  # if child is not already in either the closed or open list
                    child.distance = distance
                    openList.append(child)
                elif openIndex > -1:
                    copy = openList[openIndex]
                    if value < copy.distance + copy.height:
                        copy.distance = distance
                        copy.previous = child.previous
                        copy.height = child.height
                elif closedIndex > -1:
                    copy = closedList[closedIndex]
                    if value < copy.distance + copy.height:
                        child.distance = distance
                        closedList.remove(copy)
                        openList.append(child)
            closedList.append(puzzle)
            openList = sorted(openList, key=lambda p: p.distance + p.height)

        return []

    def bfs_solve(self):
        # Function solves puzzle using Breadth First Search

        frontier = [self]  # queue
        visited = []
        while frontier:  # while frontier has items in it
            puzzle = frontier.pop(0)  # the node the frontier is working on
            if np.all(puzzle.matrix == puzzle.solution):
                return puzzle.findPath([])
            else:
                children = puzzle.createMoves()
                for child in children:
                    if child not in visited:
                        visited.append(child)
                        if np.all(child.matrix == puzzle.solution):
                            return child.findPath([])
                        else:
                            frontier.append(child)
        return []


def index(puzzle, list):
    # Returns the index of the puzzle in the open/closed list if it is in the list, or else it returns -1.
    if puzzle in list:
        return list.index(puzzle)
    else:
        return -1


def createSolution(n):
    # Creates the solution state for a nxn puzzle.
    solution = np.arange(1, (n ** 2) + 1).reshape(n, n)
    solution[n - 1][n - 1] = 0
    return solution


class TimeException(Exception):
    """Class that allows for the algorithm to throw a time exception."""
    pass


def timeHandler(a,b):
    raise TimeException


signal.signal(signal.SIGALRM, timeHandler)

# =============== Timing Functions ==================


def runTiming(shuffle, n):
    """Shuffle: the number of times each puzzle will be shuffled. The higher the number of shuffles, the harder the
    puzzle tends to be
    n: the dimensions of a puzzle

    Function creates and solves a single puzzle using A* and BFS and records time it took"""
    duration_A_star = 0
    AstarException = 0
    duration_bfs = 0
    bfsException = 0
    puzzle = Puzzle(n)
    puzzle.randomize(shuffle)

    print("Original Puzzle:")
    printPuzzle(puzzle)
    for i in range(2):
        if i == 0:      # this means the function solves the A* algorithm first, and then the BFS one
            try:
                startTime = time.time()
                signal.alarm(60)
                print("A* algorithm:")
                path = puzzle.a_star_solve()
                printResults(path, puzzle)
                endTime = time.time()
                duration_A_star = endTime -startTime
                print(duration_A_star, "seconds")
            except TimeException:
                AstarException = 1
                print("Took longer than 60 seconds, unsolvable")
                pass
        else:
            try:
                startTime = time.time()
                signal.alarm(60)
                print("BFS algorithm:")
                path = puzzle.bfs_solve()
                printResults(path, puzzle)
                endTime = time.time()
                duration_bfs = endTime - startTime
                print(duration_bfs, "seconds")
            except TimeException:
                bfsException = 1
                print("Took longer than 60 seconds unsolvable")
                pass
        print(" ")
    return [duration_A_star, AstarException, duration_bfs, bfsException]


def printResults(path, puzzle):
    """ path: the steps in a solution to a puzzle

    Function prints the number of moves it took to get a solution
    can also print the matrix at every step in between the original and the solution
    """
    path.reverse()
    moves = 0
    for i in path:
        moves += 1
        # printPuzzle(i) #uncomment to see individual solution steps
    if moves == 0:
        # printPuzzle(puzzle)
        print(0, "moves")
    else:
        print(moves, "moves")


def printPuzzle(puzzle):
    """Function converts a puzzle matrix into a readable string object"""

    print("\n".join(' '.join(map(str, row)) for row in puzzle.matrix))
    print()


def runAverage(shuffle, n):
    """Shuffle: the number of times each puzzle will be shuffled. The higher the number of shuffles, the harder the
    puzzle tends to be
    n: the dimensions of the puzzle

    Function runs the program 10 times and calculates the average time each algorithm takes
    keeps track of number of exceptions thrown """

    totalAStarTime = 0
    totalExceptionsAStar = 0
    totalBFSTime = 0
    totalExceptionsBFS = 0

    for i in range(10):
        print("#" + str(i + 1))
        result = runTiming(shuffle, n)
        totalAStarTime = totalAStarTime + result[0]
        totalExceptionsAStar = totalExceptionsAStar + result[1]
        totalBFSTime = totalBFSTime + result[2]
        totalExceptionsBFS = totalExceptionsBFS + result[3]
        print("----")
        print(" ")

    averageAStarTime = totalAStarTime/10
    averageBFSTime = totalBFSTime/10

    print("Average solution times for a", n, "x", n, "puzzle that has been shuffled", shuffle, "times:")
    print("A* algorithm:", averageAStarTime, "seconds with", totalExceptionsAStar, "exceptions")
    print("A* algorithm:", averageBFSTime, "seconds with", totalExceptionsBFS, "exceptions")



runAverage(10, 3)
#runAverage(20,3)
#runAverage(10,4)
#runAverage(10,4)

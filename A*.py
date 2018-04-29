import random
import numpy as np


def index(puzzle, list):
    if puzzle in list:
        return list.index(puzzle)
    else:
        return -1

class Puzzle:
    #The class that represents the puzzles, their characteristics, and their functions. n is the dimension of
    #the puzzle. Zero represents the empty spot in the puzzle where neighbor tiles can be moved to.
    def __init__(self,n):
        self.distance= 0 #distance away from solution
        self.size = n #size of puzzle
        self.height = 0 #depth of the puzzle in the search algorithm
        self.previous = None #parent puzzle
        self.matrix = createSolution(n) #3x3 array of the matrix
        self.solution = createSolution(n) #solution matrix

    def randomize(self,numMoves):
        #Function that shuffles a solution to get a random puzzle. As numMoves increases,
        #the difficulty of the puzzles tend to increase. If numMoves is 100, most of the puzzles take too long to solve.
        for i in range(0,numMoves):
            moves = self.getPossibleMoves()
            tile = random.choice(moves)
            self.swapTiles(tile)

    def getPossibleMoves(self): #Function that returns the row and column of the possible moves for a puzzle.
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

    def findValue(self, value): #Finds the row and column of the value in the puzzle
        for row in range(self.size):
            for col in range(self.size):
                if self.matrix[row][col] == value:
                    return row, col

    def getValue(self, row, col): #Function that returns the value with the specified row and column.
        return self.matrix[row][col]

    def setValue(self, row, col, value):
        self.matrix[row][col] = value

    def swapTiles(self, tile):
        zero = self.findValue(0)
        self.setValue(zero[0], zero[1], self.getValue(*tile))
        self.setValue(tile[0], tile[1], 0)

    def findDistance(self,solution):
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
        if self.previous is None:
            return path
        else:
            path.append(self)
            return self.previous.findPath(path)

    def copy(self):
        puzzle = Puzzle(self.size)
        for i in range(0,self.size):
            puzzle.matrix[i] = self.matrix[i][:]
        return puzzle

    def createMoves(self):
        moves = self.getPossibleMoves()
        map = {}
        for move in moves:
            map[self.swapAndCopy(move)] = moves
        return map

    def swapAndCopy(self,b):
        p = self.copy()
        p.swapTiles(b)
        p.height = self.height + 1
        p.previous = self
        return p

    def a_star_solve(self):
        openList = [self]
        closedList = []
        while len(openList) > 0:
            puzzle = openList.pop(0)
            if np.all(puzzle.matrix==puzzle.solution):
                if len(closedList)>0:
                    return puzzle.findPath([])
                else:
                    return [puzzle]
            children = puzzle.createMoves()
            for child in children:
                openIndex = index(child, openList)
                closedIndex = index(child, closedList)
                distance = child.findDistance(child.solution)
                value = distance + child.height
                if closedIndex == -1 and openIndex == -1:
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
        frontier = [self]  # queue
        while frontier:  # while frontier has items in it
            puzzle = frontier.pop(0)  # the node the frontier is working on
            children = puzzle.createMoves()
            for child in children:
                if np.all(child.matrix == puzzle.solution):
                        return puzzle.findPath([])
                else:
                    frontier.append(child)
        return []


def createSolution(n):
    solution = np.arange(1, (n ** 2) + 1).reshape(n, n)
    solution[n - 1][n - 1] = 0
    return solution


def run(n,shuffle):
    puzzle = Puzzle(n)
    puzzle.randomize(shuffle)
    print("\n".join(' '.join(map(str, row)) for row in puzzle.matrix))
    print()
    #path = puzzle.a_star_solve() #the list of moves it takes to solve the puzzle
    path = puzzle.bfs_solve()
    path.reverse()
    moves = 0
    for i in path:
        moves += 1
        print("\n".join(' '.join(map(str, row)) for row in i.matrix))
        print()
    if moves == 1:
        print(0)
    else:
        print(moves, "moves")


run(3,20)
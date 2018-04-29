import random
import numpy as np
import time
import signal

class TimeException(Exception): #Class that allows for the algorithm to throw a time exception.
    pass

def timeHandler(a,b):
    raise TimeException

signal.signal(signal.SIGALRM,timeHandler)

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

    def setValue(self, row, col, value): #Function that sets a given value at a specified row and column.
        self.matrix[row][col] = value

    def swapTiles(self, tile): #Function that swaps two tiles. Tile is the row and column of the tile we
        #are trying to swap.
        zero = self.findValue(0)
        self.setValue(zero[0], zero[1], self.getValue(*tile))
        self.setValue(tile[0], tile[1], 0)

    def findDistance(self,solution): #Finds the total distance that a puzzle state is away from the solution.
        #The distance is defined as the sum of the number of rows and columns that a tile is away from its solved state.
        distance = 0
        for i in range(0,self.size):
            for j in range(0,self.size):
                value = solution[i][j]
                if value != 0:
                    row, col = self.findValue(value)
                    score = abs(row-i)+abs(col-j)
                    distance = distance + score
        return distance

    def findPath(self, path): #Function that returns the path that solved the puzzle.
        if self.previous is None:
            return path
        else:
            path.append(self)
            return self.previous.findPath(path)

    def copy(self): #Function that copies a puzzle.
        puzzle = Puzzle(self.size)
        for i in range(0,self.size):
            puzzle.matrix[i] = self.matrix[i][:]
        return puzzle

    def createMoves(self): #Function that returns a dictionary with the keys being the puzzle state and the values being
        #the possible puzzle states that can be obtained by making one move.
        moves = self.getPossibleMoves()
        map = {}
        for move in moves:
            map[self.swapAndCopy(move)] = moves
        return map

    def swapAndCopy(self,b): #Function that swaps a tile b and then updates the height and previous puzzle.
        #Then, it returns the new puzzle.
        p = self.copy()
        p.swapTiles(b)
        p.height = self.height + 1
        p.previous = self
        return p

    def a_star_solve(self): #Function that solves a puzzle using the A* algorithm.
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

def index(puzzle, list): #Returns the index of the puzzle in the list if it is in the list, or else it returns -1.
    if puzzle in list:
        return list.index(puzzle)
    else:
        return -1


def createSolution(n): #Creates the solution state for a nxn puzzle.
    solution = np.arange(1, (n ** 2) + 1).reshape(n, n)
    solution[n - 1][n - 1] = 0
    return solution


def runAStar(n,shuffle): #Function that solves a nxn puzzle using A* and prints the path and the amount of moves.
    puzzle = Puzzle(n)
    puzzle.randomize(shuffle)
    print("\n".join(' '.join(map(str, row)) for row in puzzle.matrix))
    print()
    path = puzzle.a_star_solve()
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

def runBFS(n,shuffle): #Function that solves a nxn puzzle using BFS and prints the path and the amount of moves.
    puzzle = Puzzle(n)
    puzzle.randomize(shuffle)
    print("\n".join(' '.join(map(str, row)) for row in puzzle.matrix))
    print()
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

def runAStarTime(puzzle): #Function that times how long it takes the A* algorithm to find the shortest path. Throws a
    #time exception if it takes longer than 60 seconds.
    startTimeAStart = time.time()
    signal.alarm(60)
    print("\n".join(' '.join(map(str, row)) for row in puzzle.matrix))
    print()
    path = puzzle.a_star_solve()
    path.reverse()
    moves = 0
    for i in path:
        moves += 1
        print("\n".join(' '.join(map(str, row)) for row in i.matrix))
        print()
    if moves == 1:
        print(0, "moves")
        endTimeAStar = time.time()
        print(endTimeAStar - startTimeAStart)
        return endTimeAStar-startTimeAStart
    else:
        print(moves, "moves")
        endTimeAStar = time.time()
        print(endTimeAStar-startTimeAStart)
        return endTimeAStar-startTimeAStart

def runBFSTime(puzzle): #Function that times how long it takes the BFS algorithm to find the shortest path. Throws a
    #time exception if it takes longer than 60 seconds.
    startTimeBFS = time.time()
    signal.alarm(60)
    print("\n".join(' '.join(map(str, row)) for row in puzzle.matrix))
    print()
    path = puzzle.bfs_solve()
    path.reverse()
    moves = 0
    for i in path:
        moves += 1
        print("\n".join(' '.join(map(str, row)) for row in i.matrix))
        print()
    if moves == 1:
        print(0, "moves")
        endTimeBFS = time.time()
        print(endTimeBFS - startTimeBFS)
        return endTimeBFS - startTimeBFS
    else:
        print(moves, "moves")
        endTimeBFS = time.time()
        print(endTimeBFS - startTimeBFS)
        return endTimeBFS - startTimeBFS

def findTime(shuffle,n): #Function that finds how long it takes both algorithms to find the shortest path, or if they
    #throw a time exception.
    puzzle = Puzzle(n)
    puzzle.randomize(shuffle)
    puzzle2 = puzzle.copy()
    try:
        timeAStar = runAStarTime(puzzle)
        numExceptionsAStar = 0
    except TimeException:
        timeAStar = 60
        numExceptionsAStar = 1
        pass
    try:
        timeBFS = runBFSTime(puzzle2)
        numExceptionsBFS = 0
    except TimeException:
        timeBFS = 60
        numExceptionsBFS = 1
    return [timeAStar,numExceptionsAStar,timeBFS,numExceptionsBFS]

def runTime(shuffle,n): #Function that compares how long the two algorithms take to find the shortest path on average
    #and checks how many time exceptions each algorithm throws.
    totalAStarTime = 0
    totalExceptionsAStar = 0
    totalBFSTime = 0
    totalExceptionsBFS = 0
    for i in range(50):
        totalAStarTime = totalAStarTime + findTime(shuffle,n)[0]
        totalExceptionsAStar = totalExceptionsAStar + findTime(shuffle, n)[1]
        totalBFSTime = totalBFSTime + findTime(shuffle, n)[2]
        totalExceptionsBFS = totalExceptionsBFS + findTime(shuffle, n)[3]
    averageAStarTime = totalAStarTime/50
    averageBFSTime = totalBFSTime/50
    print(shuffle, "shuffles takes about", averageAStarTime, "seconds to solve using A* with", totalExceptionsAStar, "exceptions being counted as 60 seconds each")
    print(shuffle, "shuffles takes about", averageBFSTime, "seconds to solve using BFS with", totalExceptionsBFS, "exceptions being counted as 60 seconds each")

runTime(20,3)
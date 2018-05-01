# Puzzle-Slider-Algorithm
Our program addresses various ways to solve a sliding grid puzzle end to end. We use two different algorithms, Breadth First Search, and A* search, and compare which algorithm is faster for solving it. We also used the algorithm to solve different-sized puzzles as well as easier and hard puzzles and compared how efficiently the two algorithms scaled up. We tested the algorithms on 3x3 and 4x4 grids and at each of those dimensions, we ran the program on puzzles that had been shuffled 10 times, and 20 times. The difficulty of the puzzle tends to increase with the number of times it has been shuffled. A* search was found to be significantly faster than BFS--especially as the size and difficulty of the puzzle increased. 

## NxN Sliding Puzzle
A sliding puzzle that consists of a frame of numbered square tiles in random order with one tile missing. For this program, the goal state of a 3x3 sliding puzzle looks like this:

1 2 3	</br>
4 5 6 </br>
7 8 0 </br>

where 0 indicates the missing tile.

## A-star Search Algorithm
* Uses heuristics to help determine the best path
* Searches paths based on whether or not they appear to be coming closer to the solution
* NxN puzzle: value based on depth of the search & distance from current state to solution state (Manhattan Distance)
* Distance = sum of amount of rows and columns each tile is away from its position in the solution state

## BFS Algorithm
* Checks every node for the best path using a queue to keep track of nodes that need to be checked
* Simple implementation either returns the correct node or an empty list if no solution can be found
*Does not rely on any kind of intelligence to omit less useful intermediate step

## Methods and Results
* Shuffle each puzzle to create a randomized puzzle. The higher the number of shuffles, the harder the puzzle is.
* Run each solution 50 times for both a 3x3 grid and a 4x4 grid. Then calculated the average solution time for each.
* If the best path is not found after 60 seconds, a time exception is thrown. The puzzle is either too hard to solve quickly, or may be unsolvable.

## References / Resources

* http://tristanpenman.com/demos/n-puzzle/ 
* https://www.geeksforgeeks.org/a-search-algorithm/ 
* https://en.wikipedia.org/wiki/A*_search_algorithm 
* https://en.wikipedia.org/wiki/Breadth-first_search 
* https://en.wikipedia.org/wiki/15_puzzle 

## Notes

Since the program uses 'SIGALRM' attribute in module Signal, which is only available on MacOS and UNIX, the program can only run on these two operating systems.


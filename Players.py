# Authors: Greyson Wintergerst, ... (add your name here if you worked on this file) FIXME
# Description: This file contains the various player classes, which contain all data and logic for active players.
# The following classes are defined in this file:
# - Player: Parent class for all player objects
# - Human: Player object that takes input from the user to make decisions
# - Bot: Player object that implements several AI strategies (including alpha-beta pruning, DeepQ, random, and territory-based strategies)
# - PruneComputer: Player object that uses the minimax algorithm with alpha-beta pruning to make decisions
# - GenComputer: Player object that uses a genetic algorithm to make decisions
# - aStarComputer: Player object that uses the A* algorithm to make decisions

import pygame
import random
import heapq

# Parent class that defines several useful functions and characteristics for all players in a given game instance
class Player:
    def __init__(self, gameObj, color, ID, x, y, direction):
        self.color = color # color of the player and their trail
        self.ID = ID # player's ID used for logic in the board/grid implementation
        self.gameObj = gameObj # reference to a Tron object
        self.x = x # x-coordinate of the player
        self.y = y # y-coordinate of the player
        self.direction = direction # direction the player is moving (UP, RIGHT, DOWN, LEFT)

        self.prevPos = (x, y)
        self.alive = True

        self.directionQueue = [] # queue of directions to move in; created from player input
        self.timeAlive = 0 # number of ticks that the player has been alive
		

        self.gameObj.board.grid[y][x] = ID # set the player's position on the board

    # Constants for directions of the player
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    # Function to move the player in the direction they are currently facing
    def movePlayer(self):
        # move player and re-draw GameBoard object
        self.prevPos = (self.x, self.y)

        match self.direction:
            case self.UP:
                self.gameObj.board.grid[self.y - 1][self.x] = self.ID
                self.y = self.y - 1
            case self.RIGHT:
                self.gameObj.board.grid[self.y][self.x + 1] = self.ID
                self.x = self.x + 1
            case self.DOWN:
                self.gameObj.board.grid[self.y + 1][self.x] = self.ID
                self.y = self.y + 1
            case self.LEFT:
                self.gameObj.board.grid[self.y][self.x - 1] = self.ID
                self.x = self.x - 1
    
    # Returns true if the player would collide with itself based on current direction     
    def isInvalidDirection(self, nextDirection):
        return (nextDirection == Player.UP and self.direction == Player.DOWN) or \
               (nextDirection == Player.DOWN and self.direction == Player.UP) or \
               (nextDirection == Player.LEFT and self.direction == Player.RIGHT) or \
               (nextDirection == Player.RIGHT and self.direction == Player.LEFT)

    # Returns true if the player would collide with a wall or a player trail based on a given direction
    def isCollision(self, direction):
        return (direction == Player.UP and self.gameObj.board.isObstacle(self.x, self.y - 1)) or \
            (direction == Player.RIGHT and self.gameObj.board.isObstacle(self.x + 1, self.y)) or \
            (direction == Player.DOWN and self.gameObj.board.isObstacle(self.x, self.y + 1)) or \
            (direction == Player.LEFT and self.gameObj.board.isObstacle(
                self.x - 1, self.y))

    # Sets the player's alive status to False if they have collided with a wall or player trail
    def checkCollision(self, direction):
        if self.isCollision(direction):
            self.alive = False  # kill player
    
    # Returns true if going in a given direction would result in a collision with a wall or player trail
    def isCollision(self, direction):
        return (direction == Player.UP and self.gameObj.board.isObstacle(self.x, self.y - 1)) or \
            (direction == Player.RIGHT and self.gameObj.board.isObstacle(self.x + 1, self.y)) or \
            (direction == Player.DOWN and self.gameObj.board.isObstacle(self.x, self.y + 1)) or \
            (direction == Player.LEFT and self.gameObj.board.isObstacle(
                self.x - 1, self.y))
            
    # Returns the location of the player if they were to move in a given direction
    def convertDirectionToLocation(self, x, y, direction):
        match direction:
            case self.UP:
                return (x, y - 1)
            case self.RIGHT:
                return (x + 1, y)
            case self.DOWN:
                return (x, y + 1)
            case self.LEFT:
                return (x - 1, y)
            
    # Returns an estimated number of potential tiles that a player could occupy before their opponent. Uses BFS to calculate territory from a given position by
    # treating reachable open tiles as components of "layers"
    #
    # Credit for this function goes to the original authors, as seen here: https://github.com/nlieb/PyTron/blob/44721c285f3140b80c3f66816701bfc304b15c06/Player.py#L72C66-L72C67
    def calculateDirectionTerritory(self, direction, opponentDirection):
        
        nplayers = len(self.gameObj.players)
        
        playerTerritory = [0] * nplayers
        qs = [[] for x in range(nplayers)] # 1 queue for bfs search for each player
        
        start = self.convertDirectionToLocation(self.x, self.y, direction)
        
        qs[self.ID-1].append((start, 0))
        
        seenLocations = {start: self.ID}
        
        for playerID in set(self.gameObj.players)-set([self.ID]):
            start = self.convertDirectionToLocation(self.gameObj.players[playerID].x, self.gameObj.players[playerID].y, opponentDirection)
            qs[playerID-1].append((start, 0))
            
            
        # start bfs
        depth = 0
        while sum(map(len, qs)): # while there are still locations to explore
            seenThisLayer = {}
            for playerID in range(nplayers):
                seenThisPlayerLayer = {}
                
                while qs[playerID] and qs[playerID][0][1] <= depth:
                    a = qs[playerID].pop(0)
                    curloc = a[0]
                    
                    if curloc not in seenThisLayer:
                        seenThisLayer[curloc] = playerID
                    else:
                        seenThisLayer[curloc] = -1
                        
                    for dir in range(0,4):
                        loc = self.convertDirectionToLocation(curloc[0], curloc[1], dir)
                        if loc not in seenLocations and loc not in seenThisPlayerLayer and not self.gameObj.board.isObstacle(loc[0], loc[1]):
                            seenThisPlayerLayer[loc] = 1
                            qs[playerID].append((loc, a[1]+1))
                            
            for loc in seenThisLayer:
                player = seenThisLayer[loc]
                seenLocations[loc] = player
                
                if player >= 0:
                    playerTerritory[player] += 1
            
            depth += 1
            
            
        return playerTerritory[self.ID-1]
			
    # Returns the minimum distance to a wall given a position (x, y)
    def distanceToClosestWall(self, x, y):
        return min(x, y, self.gameObj.board.xTiles - x, self.gameObj.board.yTiles - y)

    def tick(self):
        pass

    def event(self, event):
        pass

class PruneComputer(Player):
    def __init__(self, gameObj, color, ID, x, y, direction):
        super().__init__(gameObj, color, ID, x, y, direction)
        self.max_depth = 5  # Adjust this to control look-ahead depth

    def tick(self):
        best_move = self.decision()
        if best_move is not None:
            self.direction = best_move

    def decision(self):
        best_value = float('-inf')
        best_move = None
        alpha = float('-inf')
        beta = float('inf')

        # Get possible moves
        valid_moves = self.get_valid_moves()

        # If no valid moves, return current direction
        if not valid_moves:
            return self.direction

        # Try each possible move
        for move in valid_moves:
            # Create a copy of current state
            next_pos = self.convertDirectionToLocation(self.x, self.y, move)

            # Skip if move leads to immediate collision
            if self.isCollision(move):
                continue

            # Simulate move
            value = self.min_value(
                next_pos, self.get_opponent_position(), 1, alpha, beta)

            if best_value < value:
                best_move = move
                best_value = value

            alpha = max(alpha, best_value)

        return best_move

    def min_value(self, my_pos, opp_pos, depth, alpha, beta):
        if depth >= self.max_depth or self.is_terminal_state(my_pos, opp_pos):
            return self.evaluate_state(my_pos, opp_pos)

        value = float('inf')
        op_moves = self.get_valid_moves_for_position(opp_pos)

        for op_move in op_moves:
            next_op_pos = self.get_next_position(opp_pos, op_move)

            if self.is_position_blocked(next_op_pos):
                continue

            cur_value = self.max_value(
                my_pos, next_op_pos, depth + 1, alpha, beta)
            value = min(value, cur_value)

            if value <= alpha:
                return value
            beta = min(beta, value)

        return value

    def max_value(self, my_pos, op_pos, depth, alpha, beta):
        if depth >= self.max_depth or self.is_terminal_state(my_pos, op_pos):
            return self.evaluate_state(my_pos, op_pos)

        value = float('-inf')

        my_moves = self.get_valid_moves_for_position(my_pos)

        for my_move in my_moves:
            next_my_pos = self.get_next_position(my_pos, my_move)

            if self.is_position_blocked(next_my_pos):
                continue

            curr_value = self.min_value(
                next_my_pos, op_pos, depth + 1, alpha, beta)
            value = max(value, curr_value)

            if value >= beta:
                return value
            alpha = max(alpha, value)

        return value

    def get_valid_moves(self):
        valid_moves = []
        for direction in [Player.UP, Player.RIGHT, Player.DOWN, Player.LEFT]:
            if not self.isInvalidDirection(direction) and not self.isCollision(direction):
                valid_moves.append(direction)
        return valid_moves

    def get_valid_moves_for_position(self, pos):
        valid_moves = []
        for direction in [Player.UP, Player.RIGHT, Player.DOWN, Player.LEFT]:
            next_pos = self.get_next_position(pos, direction)
            if not self.is_position_blocked(next_pos):
                valid_moves.append(direction)
        return valid_moves

    def get_next_position(self, pos, direction):
        x, y = pos
        if direction == Player.UP:
            return (x, y - 1)
        elif direction == Player.DOWN:
            return (x, y + 1)
        elif direction == Player.LEFT:
            return (x - 1, y)
        else:  # RIGHT
            return (x + 1, y)

    def is_position_blocked(self, pos):
        x, y = pos
        return self.gameObj.board.isObstacle(x, y)

    def get_opponent_position(self):
        opponent_id = 2 if self.ID == 1 else 1
        opponent = self.gameObj.players[opponent_id]
        return (opponent.x, opponent.y)

    def is_terminal_state(self, my_pos, op_pos):
        if self.is_position_blocked(my_pos) or self.is_position_blocked(op_pos):
            return True

        my_moves = self.get_valid_moves_for_position(my_pos)
        op_moves = self.get_valid_moves_for_position(op_pos)

        return len(my_moves) == 0 or len(op_moves) == 0

    def evaluate_state(self, my_pos, op_pos):
        my_moves = len(self.get_valid_moves_for_position(my_pos))
        op_moves = len(self.get_valid_moves_for_position(op_pos))

        if op_moves == 0:
            return float('inf')   # We won
        if my_moves == 0:
            return float('-inf')  # We lost

        # Calculate available space using flood fill
        op_space = self.flood_fill_count(op_pos)
        my_space = self.flood_fill_count(my_pos)

        return (my_moves * 10 + my_space) - (op_moves * 10 + op_space)

    def flood_fill_count(self, start_pos):
        visited = set()
        stack = [start_pos]

        while stack:
            pos = stack.pop()
            if pos not in visited and not self.is_position_blocked(pos):
                visited.add(pos)
                x, y = pos
                stack.extend([
                    (x - 1, y),
                    (x + 1, y),
                    (x, y - 1),
                    (x, y + 1)
                ])

        return len(visited)

class Human(Player):
    def __init__(self, gameObj, color, ID, x, y, direction, keybinds):
        super().__init__(gameObj, color, ID, x, y, direction)
        self.p_up = keybinds[0]
        self.p_left = keybinds[1]
        self.p_down = keybinds[2]
        self.p_right = keybinds[3]

    def tick(self):
        while self.directionQueue:

            if self.isInvalidDirection(self.directionQueue[0]) or self.directionQueue[0] == self.direction:
                self.directionQueue.pop(0)
            else:
                self.direction = self.directionQueue.pop(0)

            self.prevPos = (self.x, self.y)  # why do we need this again?

    def event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == self.p_up:
                self.directionQueue.append(Player.UP)
            elif event.key == self.p_left:
                self.directionQueue.append(Player.LEFT)
            elif event.key == self.p_down:
                self.directionQueue.append(Player.DOWN)
            elif event.key == self.p_right:
                self.directionQueue.append(Player.RIGHT)


class GenComputer(Player):
    def __init__(self, gameObj, color, ID, x, y, direction, genome):
        super(GenComputer, self).__init__(gameObj, color, ID, x, y, direction)
        self.genome = genome

    def tick(self):
        self.strategyGenetic()

    def distanceToSelf(self, direction):
        next_x, next_y = self.convertDirectionToLocation(self.x, self.y, direction)

        min_distance = float('inf')

        for point in self.gameObj.board.getBotTrail(self.ID):
            trail_x, trail_y = point
            distance = abs(next_x - trail_x) + abs(next_y - trail_y)
            min_distance = min(min_distance, distance)

        return min_distance

    def predictTrap(self, direction):
        next_x, next_y = self.convertDirectionToLocation(self.x, self.y, direction)

        temp_board = self.gameObj.board.copy()
        temp_board.grid[next_y][next_x] = self.ID

        reachable_area = self.calculateReachableArea(
            next_x, next_y, temp_board)

        trap_threshold = 50

        return reachable_area < trap_threshold

    def calculateReachableArea(self, x, y, board):
        visited = set()
        queue = [(x, y)]
        reachable_count = 0

        while queue:
            cx, cy = queue.pop(0)

            if (cx, cy) in visited:
                continue

            visited.add((cx, cy))
            if not board.isObstacle(cx, cy) or (cx == x and cy == y):
                reachable_count += 1

                neighbors = [
                    (cx + 1, cy), (cx - 1, cy),
                    (cx, cy + 1), (cx, cy - 1)
                ]
                for nx, ny in neighbors:
                    if (nx, ny) not in visited:
                        queue.append((nx, ny))

        return reachable_count

    def strategyGenetic(self):
        max_score = -float('inf')
        best_direction = None
        opponentId = (set(self.gameObj.players) - {self.ID}).pop()

        for direction in range(4):
            if self.isCollision(direction):
                continue

            if self.predictTrap(direction):
                continue

            max_attempts = 10
            attempts = 0
            opponent_direction = random.choice(range(4))
            while self.gameObj.players[opponentId].isCollision(opponent_direction) and attempts < max_attempts:
                opponent_direction = random.choice(range(4))
                attempts += 1

            if attempts >= max_attempts:
                opponent_direction = Player.UP

            distance_to_self = self.distanceToSelf(direction)
            survival = 1 if not self.isCollision(direction) else 0
            self_collision_penalty = -10 / (distance_to_self + 1)
            aggression = -abs(direction - opponent_direction)

            score = (
                self.genome[0] * survival +
                self.genome[1] * aggression +
                self_collision_penalty
            )

            if score > max_score:
                max_score = score
                best_direction = direction

        if best_direction is None:
            best_direction = self.direction

        self.direction = best_direction


class aStarComputer(Player):
    def __init__(self, gameObj, color, ID, x, y, direction):
        super().__init__(gameObj, color, ID, x, y, direction)
        self.max_depth = 5  # Adjust this to control look-ahead depth

    def deepQStrategy(self):
	# determine direction
        return random.choice([Player.UP, Player.RIGHT, Player.DOWN, Player.LEFT])
	
    def tick(self):
        best_move = self.decision()
        if best_move is not None:
            self.direction = best_move

    def decision(self):
        best_value = float('-inf')
        best_move = None
        alpha = float('-inf')
        beta = float('inf')

        # Get possible moves
        valid_moves = self.get_valid_moves()

        # If no valid moves, return current direction
        if not valid_moves:
            return self.direction

        # Try each possible move
        for move in valid_moves:
            # Create a copy of current state
            next_pos = self.convertDirectionToLocation(self.x, self.y, move)

            # Skip if move leads to immediate collision
            if self.isCollision(move):
                continue

            # Simulate move
            value = self.min_value(
                next_pos, self.get_opponent_position(), 1, alpha, beta)

            if best_value < value:
                best_move = move
                best_value = value

            alpha = max(alpha, best_value)

        return best_move

    def min_value(self, my_pos, opp_pos, depth, alpha, beta):
        if depth >= self.max_depth or self.is_terminal_state(my_pos, opp_pos):
            return self.evaluate_state(my_pos, opp_pos)

        value = float('inf')
        op_moves = self.get_valid_moves_for_position(opp_pos)

        for op_move in op_moves:
            next_op_pos = self.get_next_position(opp_pos, op_move)

            if self.is_position_blocked(next_op_pos):
                continue

            cur_value = self.max_value(
                my_pos, next_op_pos, depth + 1, alpha, beta)
            value = min(value, cur_value)

            if value <= alpha:
                return value
            beta = min(beta, value)

        return value

    def max_value(self, my_pos, op_pos, depth, alpha, beta):
        if depth >= self.max_depth or self.is_terminal_state(my_pos, op_pos):
            return self.evaluate_state(my_pos, op_pos)

        value = float('-inf')

        my_moves = self.get_valid_moves_for_position(my_pos)

        for my_move in my_moves:
            next_my_pos = self.get_next_position(my_pos, my_move)

            if self.is_position_blocked(next_my_pos):
                continue

            curr_value = self.min_value(
                next_my_pos, op_pos, depth + 1, alpha, beta)
            value = max(value, curr_value)

            if value >= beta:
                return value
            alpha = max(alpha, value)
        self.strategyAStar()

        return value

    def get_valid_moves(self):
        valid_moves = []
        for direction in [Player.UP, Player.RIGHT, Player.DOWN, Player.LEFT]:
            if not self.isInvalidDirection(direction) and not self.isCollision(direction):
                valid_moves.append(direction)
        return valid_moves

    def event(self, event):
        pass

    def get_valid_moves_for_position(self, pos):
        valid_moves = []
        for direction in [Player.UP, Player.RIGHT, Player.DOWN, Player.LEFT]:
            next_pos = self.get_next_position(pos, direction)
            if not self.is_position_blocked(next_pos):
                valid_moves.append(direction)
        return valid_moves

    def isValidMove(self, x, y):
        return (0 <= x < self.gameObj.board.xTiles and
                0 <= y < self.gameObj.board.yTiles and
                not self.gameObj.board.isObstacle(x, y))

    def directionToNextLocation(self, x, y, d):
        match d:
            case self.UP:
                return x, y-1
            case self.DOWN:
                return x, y+1
            case self.LEFT:
                return x-1, y
            case self.RIGHT:
                return x+1, y

    def get_next_position(self, pos, direction):
        x, y = pos
        if direction == Player.UP:
            return (x, y - 1)
        elif direction == Player.DOWN:
            return (x, y + 1)
        elif direction == Player.LEFT:
            return (x - 1, y)
        else:  # RIGHT
            return (x + 1, y)

    def is_position_blocked(self, pos):

        x, y = pos
        return self.gameObj.board.isObstacle(x, y)

    def get_opponent_position(self):
        opponent_id = 2 if self.ID == 1 else 1
        opponent = self.gameObj.players[opponent_id]
        return (opponent.x, opponent.y)

    def is_terminal_state(self, my_pos, op_pos):
        if self.is_position_blocked(my_pos) or self.is_position_blocked(op_pos):
            return True

        my_moves = self.get_valid_moves_for_position(my_pos)
        op_moves = self.get_valid_moves_for_position(op_pos)

        return len(my_moves) == 0 or len(op_moves) == 0

    def evaluate_state(self, my_pos, op_pos):
        my_moves = len(self.get_valid_moves_for_position(my_pos))
        op_moves = len(self.get_valid_moves_for_position(op_pos))

        if op_moves == 0:
            return float('inf')   # We won
        if my_moves == 0:
            return float('-inf')  # We lost

        # Calculate available space using flood fill
        op_space = self.flood_fill_count(op_pos)
        my_space = self.flood_fill_count(my_pos)
        return (my_moves * 10 + my_space) - (op_moves * 10 + op_space)

    def flood_fill_count(self, start_pos):
        visited = set()
        stack = [start_pos]

        while stack:
            pos = stack.pop()
            if pos not in visited and not self.is_position_blocked(pos):
                visited.add(pos)
                x, y = pos
                stack.extend([
                    (x - 1, y),
                    (x + 1, y),
                    (x, y - 1),
                    (x, y + 1)
                ])

        return len(visited)

      
    def neighbors(self, pos):
	# determines what the valid neighboring positions are
        for direction in range(4):
            nX, nY = self.directionToNextLocation(nX, nY, direction)
            if self.isValidMove(nX, nY):
                yield nX, nY, direction
	
    def heuristic(self, x, y):
	# Manhattan distance calculation
        target = self.findLargestSafeArea()
        return abs(target[0] - x) + abs(target[1] - y)

    def strategyAStar(self):
	# AStar Search Strategy
        print("Starting A*")
        open_set = [] 
        start = (self.x, self.y)
        # heap order: (priority, cost, position, direction)
        heapq.heappush(open_set, (0, 0, start, None))
        came_from = {} # dictionary: store path taken
        cost_so_far = {start: 0} # dictionary: store cost
        goal = None

        while open_set: # go through open set until it's over
            _, current_cost, current, current_dir = heapq.heappop(open_set) # pop position with the lowest priority (best option)
            print(f"Current: {current}, Cost: {current_cost}")
	
            if self.isSafeGoal(current): # current meet criteria?
                goal = current
                print(f"Goal found: {goal}")
                break

            for nX, nY, direction in self.neighbors(current): # generate the valid neigbor list with neighbor function
                next_pos = (nX, nY)
                new_cost = current_cost + 1

                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]: # 1. hasn't been visited or 2. cheper path found
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.heuristic(nX, nY) # calc priority = cost so far + estimated cost (heuristic)
                    heapq.heappush(
                        open_set, (priority, new_cost, next_pos, direction)) # add neighbor to set
                    came_from[next_pos] = (current, direction) # record path

        if goal: # goal found? reconstruct
            print(f"Path to goal found. Goal: {goal}")
            current = goal
            while came_from[current][0] != start: # backtrack
                current = came_from[current][0]
            self.direction = came_from[current][1]
            print(f"Next direction: {self.direction}")
        else: # no goal found
            print("No path found")

    def isSafeGoal(self, pos): # make sure next move is valid, also helps with debugging
        x, y = pos
        result = self.isValidMove(x, y)
        print(f"Checking if position {pos} is a safe goal: {result}")
        return result

    def findLargestSafeArea(self):
        largest_area = None # coordinates of largest area
        largest_size = 0 # size of largest area
	    
	# iterate through
        for x in range(self.gameObj.board.xTiles):
            for y in range(self.gameObj.board.yTiles):
                if not self.gameObj.board.isObstacle(x, y): # make sure current is not an obstacle
                    # count adjacent free spaces
                    size = sum([1 for nx, ny, _ in self.neighbors(
                        (x, y)) if not self.gameObj.board.isObstacle(nx, ny)])
                    if size > largest_size: # update largest_area & largest_size if found
                        largest_size = size
                        largest_area = (x, y)

        return largest_area if largest_area else (self.x, self.y)


class Bot(Player):
    def __init__(self, gameObj, color, ID, x, y, direction, deepQModel=None):
        super().__init__(gameObj, color, ID, x, y, direction)
        self.max_depth = 5  # Adjust this to control look-ahead depth for the minimax algorithm
        self.deepQModel = deepQModel # deep Q-learning model for the bot to use
        
    # Function that handles DeepQ logic for both training and playing phases of the game
    # Alters the Bot object's direction (self.direction) based on the output of the model. 
    def deepQStrategy(self, gameState):
        state = self.gameObj.getEnvState()
        if gameState == 'TRAINING':
            self.direction = self.deepQModel.act(state, self.ID)
        else:
            print("Predicting direction")
            valid_actions = [
            action for action in range(4) 
            if not self.gameObj.players[self.ID].isCollision(action)
        	]
            print(valid_actions)
            self.direction = self.deepQModel.predict(state, valid_actions)
            print("Predicted direction: ", self.direction)
    
    # Function that alters self.direction randomly at each tick.
    def strategyRandom(self):
        dir = list(range(0, 4))
        
        # 10% chance of changing directions
        if random.randint(1, 10) == 1:
            self.direction = dir[random.randint(0, len(dir)-1)]
            
        # if we would collide, pick a new random direction
        while dir and self.isCollision(self.direction):
            self.direction = dir.pop(random.randint(0, len(dir)-1))
            
    # Function that alters self.direction based on the direction that would result in the most territory for the Bot
    #
    # Credit for this function goes to the original authors, as seen here: https://github.com/nlieb/PyTron/blob/44721c285f3140b80c3f66816701bfc304b15c06/Player.py#L209
    def strategyMostTerritory(self):
        maxArea = 0
        bestDirection = 0
        
        opponentId = (set(self.gameObj.players)-set([self.ID])).pop()

        
        for direction in range(0,4):
            if self.isCollision(direction): continue
            
            # generate random direction for opponent (assumes only 2 players)
            
            a = list(range(0,4))
            opdir = a.pop(random.randint(0, 3))
            while a and self.gameObj.players[opponentId].isCollision(opdir):
                opdir = a.pop(random.randint(0, len(a)-1))
                
            area = self.calculateDirectionTerritory(direction, opdir)
            if area > maxArea:
                bestDirection = direction
                maxArea = area
                
        self.direction = bestDirection

    # Implements Bot action at each step of the game
    def tick(self):
        if self.deepQModel is None:
            if self.gameObj.state == 'TRAINING': # training DeepQ model
                self.strategyMostTerritory() # strategy for opponent in DeepQ Training
            else:
                best_move = self.decision()
                if best_move is not None:
                    self.direction = best_move
        else:
            self.deepQStrategy(self.gameObj.state)

    # Function that determines the best move for the Bot using the minimax algorithm
    def decision(self):
        best_value = float('-inf')
        best_move = None
        alpha = float('-inf')
        beta = float('inf')

        # Get possible moves
        valid_moves = self.get_valid_moves()

        # If no valid moves, return current direction
        if not valid_moves:
            return self.direction

        # Try each possible move
        for move in valid_moves:
            # Create a copy of current state
            next_pos = self.convertDirectionToLocation(self.x, self.y, move)

            # Skip if move leads to immediate collision
            if self.isCollision(move):
                continue

            # Simulate move
            value = self.min_value(
                next_pos, self.get_opponent_position(), 1, alpha, beta)

            if best_value < value:
                best_move = move
                best_value = value

            alpha = max(alpha, best_value)

        return best_move

    def min_value(self, my_pos, opp_pos, depth, alpha, beta):
        if depth >= self.max_depth or self.is_terminal_state(my_pos, opp_pos):
            return self.evaluate_state(my_pos, opp_pos)

        value = float('inf')
        op_moves = self.get_valid_moves_for_position(opp_pos)

        for op_move in op_moves:
            next_op_pos = self.get_next_position(opp_pos, op_move)

            if self.is_position_blocked(next_op_pos):
                continue

            cur_value = self.max_value(
                my_pos, next_op_pos, depth + 1, alpha, beta)
            value = min(value, cur_value)

            if value <= alpha:
                return value
            beta = min(beta, value)

        return value

    def max_value(self, my_pos, op_pos, depth, alpha, beta):
        if depth >= self.max_depth or self.is_terminal_state(my_pos, op_pos):
            return self.evaluate_state(my_pos, op_pos)

        value = float('-inf')

        my_moves = self.get_valid_moves_for_position(my_pos)

        for my_move in my_moves:
            next_my_pos = self.get_next_position(my_pos, my_move)

            if self.is_position_blocked(next_my_pos):
                continue

            curr_value = self.min_value(
                next_my_pos, op_pos, depth + 1, alpha, beta)
            value = max(value, curr_value)

            if value >= beta:
                return value
            alpha = max(alpha, value)

        return value

    def get_valid_moves(self):
        valid_moves = []
        for direction in [Player.UP, Player.RIGHT, Player.DOWN, Player.LEFT]:
            if not self.isInvalidDirection(direction) and not self.isCollision(direction):
                valid_moves.append(direction)
        return valid_moves

    def get_valid_moves_for_position(self, pos):
        valid_moves = []
        for direction in [Player.UP, Player.RIGHT, Player.DOWN, Player.LEFT]:
            next_pos = self.get_next_position(pos, direction)
            if not self.is_position_blocked(next_pos):
                valid_moves.append(direction)
        return valid_moves

    def get_next_position(self, pos, direction):
        x, y = pos
        if direction == Player.UP:
            return (x, y - 1)
        elif direction == Player.DOWN:
            return (x, y + 1)
        elif direction == Player.LEFT:
            return (x - 1, y)
        else:  # RIGHT
            return (x + 1, y)

    def is_position_blocked(self, pos):
        x, y = pos
        return self.gameObj.board.isObstacle(x, y)

    def get_opponent_position(self):
        opponent_id = 2 if self.ID == 1 else 1
        opponent = self.gameObj.players[opponent_id]
        return (opponent.x, opponent.y)

    def is_terminal_state(self, my_pos, op_pos):
        if self.is_position_blocked(my_pos) or self.is_position_blocked(op_pos):
            return True

        my_moves = self.get_valid_moves_for_position(my_pos)
        op_moves = self.get_valid_moves_for_position(op_pos)

        return len(my_moves) == 0 or len(op_moves) == 0

    def evaluate_state(self, my_pos, op_pos):
        my_moves = len(self.get_valid_moves_for_position(my_pos))
        op_moves = len(self.get_valid_moves_for_position(op_pos))

        if op_moves == 0:
            return float('inf')   # We won
        if my_moves == 0:
            return float('-inf')  # We lost

        # Calculate available space using flood fill
        op_space = self.flood_fill_count(op_pos)
        my_space = self.flood_fill_count(my_pos)

        return (my_moves * 10 + my_space) - (op_moves * 10 + op_space)

    def flood_fill_count(self, start_pos):
        visited = set()
        stack = [start_pos]

        while stack:
            pos = stack.pop()
            if pos not in visited and not self.is_position_blocked(pos):
                visited.add(pos)
                x, y = pos
                stack.extend([
                    (x - 1, y),
                    (x + 1, y),
                    (x, y - 1),
                    (x, y + 1)
                ])

        return len(visited)

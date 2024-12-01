import pygame
import random
import heapq


class Player:

    def __init__(self, gameObj, color, ID, x, y, direction):
        self.color = color
        self.ID = ID
        self.gameObj = gameObj
        self.x = x
        self.y = y
        self.direction = direction

        self.prevPos = (x, y)
        self.alive = True

        self.directionQueue = []

        self.gameObj.board.grid[y][x] = ID

    # Constants for directions of the player
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

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

    def isInvalidDirection(self, nextDirection):
        return (nextDirection == Player.UP and self.direction == Player.DOWN) or \
               (nextDirection == Player.DOWN and self.direction == Player.UP) or \
               (nextDirection == Player.LEFT and self.direction == Player.RIGHT) or \
               (nextDirection == Player.RIGHT and self.direction == Player.LEFT)

    def isCollision(self, direction):
        return (direction == Player.UP and self.gameObj.board.isObstacle(self.x, self.y - 1)) or \
            (direction == Player.RIGHT and self.gameObj.board.isObstacle(self.x + 1, self.y)) or \
            (direction == Player.DOWN and self.gameObj.board.isObstacle(self.x, self.y + 1)) or \
            (direction == Player.LEFT and self.gameObj.board.isObstacle(
                self.x - 1, self.y))
            
    def checkCollision(self, direction):
        if self.isCollision(direction):
            self.alive = False # kill player

    def convertDirectionToLocation(self, direction):
        match direction:
            case self.UP:
                return (self.x, self.y - 1)
            case self.RIGHT:
                return (self.x + 1, self.y)
            case self.DOWN:
                return (self.x, self.y + 1)
            case self.LEFT:
                return (self.x - 1, self.y)

    def tick(self):
        pass

    def event(self, event):
        pass


class Bot(Player):
    def __init__(self, gameObj, color, ID, x, y, direction):
        super().__init__(gameObj, color, ID, x, y, direction)
        
    def deepQStrategy(self):
        return random.choice([Player.UP, Player.RIGHT, Player.DOWN, Player.LEFT])

    def tick(self):
        self.strategyAStar()

    def event(self, event):
        pass

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

    def neighbors(self, pos): 
        x, y = pos
        for direction in range(4): 
            nX, nY = self.directionToNextLocation(x, y, direction)
            if self.isValidMove(nX, nY): 
                print(f"Neighbor: {(nX, nY)}, Direction: {direction}")
                yield nX, nY, direction
  
    def heuristic(self, x, y): 
        target = self.findLargestSafeArea()
        return abs(target[0] - x) + abs(target[1] - y)

    def strategyAStar(self):
        print("Starting A*")
        open_set = [] 
        start = (self.x, self.y) 
        heapq.heappush(open_set, (0, 0, start, None))  # (priority, cost, position, direction)
        came_from = {} 
        cost_so_far = {start: 0} 
        goal = None
        
        while open_set: 
            _, current_cost, current, current_dir = heapq.heappop(open_set)
            print(f"Current: {current}, Cost: {current_cost}")
            
            if self.isSafeGoal(current): 
                goal = current 
                print(f"Goal found: {goal}")
                break

            for nX, nY, direction in self.neighbors(current): 
                next_pos = (nX, nY) 
                new_cost = current_cost + 1
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]: 
                    cost_so_far[next_pos] = new_cost 
                    priority = new_cost + self.heuristic(nX, nY) 
                    heapq.heappush(open_set, (priority, new_cost, next_pos, direction)) 
                    came_from[next_pos] = (current, direction)
        
        if goal: 
            print(f"Path to goal found. Goal: {goal}")
            current = goal 
            while came_from[current][0] != start: 
                current = came_from[current][0] 
            self.direction = came_from[current][1]
            print(f"Next direction: {self.direction}")
        else:
            print("No path found")

    def isSafeGoal(self, pos): 
        x, y = pos 
        result = self.isValidMove(x, y)
        print(f"Checking if position {pos} is a safe goal: {result}")
        return result
    
    def findLargestSafeArea(self): 
        largest_area = None
        largest_size = 0

        for x in range(self.gameObj.board.xTiles): 
            for y in range(self.gameObj.board.yTiles): 
                if not self.gameObj.board.isObstacle(x, y):
                    # Count adjacent free spaces
                    size = sum([1 for nx, ny, _ in self.neighbors((x, y)) if not self.gameObj.board.isObstacle(nx, ny)])
                    if size > largest_size:
                        largest_size = size
                        largest_area = (x, y)
        
        return largest_area if largest_area else (self.x, self.y)

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

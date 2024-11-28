import pygame
import random


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
        pass

    def event(self, event):
        pass


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

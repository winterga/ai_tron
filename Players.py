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
            self.alive = False  # kill player

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
            next_pos = self.convertDirectionToLocation(move)

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

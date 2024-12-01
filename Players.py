import pygame
import random

class Player:
	def __init__(self, gameObj, color, playerid, x, y, initialDirection):
		self.color = color # rgb color of player
		self.playerid = playerid # player id indexed from 1
		self.game = gameObj

		self.posX = x
		self.posY = y
		
		# prev position is used for the input queue to prevent self collision
		self.prevPos = {'x':x, 'y':y}
		self.alive = True
		
		# list of directions inputs to be executed in subsequent frames
		self.directionQ = []
		self.maxDirectionQLen = 3
		
		# where to go when we draw the next frame
		self.direction = initialDirection

		gameObj.board.grid[y][x] = playerid
	
	#direction definitions
	UP = 0
	RIGHT = 1
	DOWN = 2
	LEFT = 3

	def movePlayer(self):
		# move the player (change position) and update the game board
		if self.direction == Player.UP:
			self.game.board.grid[self.posY-1][self.posX] = self.playerid
			self.posY = self.posY - 1
		elif self.direction == Player.RIGHT:
			self.game.board.grid[self.posY][self.posX+1] = self.playerid
			self.posX = self.posX + 1
		elif self.direction == Player.DOWN:
			self.game.board.grid[self.posY+1][self.posX] = self.playerid
			self.posY = self.posY + 1
		elif self.direction == Player.LEFT:
			self.game.board.grid[self.posY][self.posX-1] = self.playerid
			self.posX = self.posX - 1

	def checkForCollision(self, direction):
		if self.wouldCollide(direction) == True:
			self.alive = False

	def wouldCollideSelf(self, nextDirection):
		'''Check if the player would collide with their previous position going a particular direction'''
		return (nextDirection == Player.UP and self.posY-1 == self.prevPos['y']) or \
			(nextDirection == Player.RIGHT and self.posX+1 == self.prevPos['x']) or \
			(nextDirection == Player.DOWN and self.posY+1 == self.prevPos['y']) or \
			(nextDirection == Player.LEFT and self.posX-1 == self.prevPos['x'])

	def wouldCollide(self, direction):
		'''check if a player would collide with an obstacle if they move in a particular direction'''
		return (direction == Player.UP and self.game.board.isObstacle(self.posX, self.posY-1)) or \
			(direction == Player.RIGHT and self.game.board.isObstacle(self.posX+1, self.posY)) or \
			(direction == Player.DOWN and self.game.board.isObstacle(self.posX, self.posY+1)) or \
			(direction == Player.LEFT and self.game.board.isObstacle(self.posX-1, self.posY))

	def directionToNextLocation(self, posX, posY, direction):
		'''converts current position and direction to the next position'''
		if direction == Player.UP: return (posX, posY-1)
		if direction == Player.RIGHT: return (posX+1, posY)
		if direction == Player.DOWN: return (posX, posY+1)
		if direction == Player.LEFT: return (posX-1, posY)

	def calculateDirectionTerritory(self, direction, opponentDirection):
		''' Given a directions, calculate a player's predicted territory,
			the number of positions he can reach before all other player.
			assumption: the next location based on direction and current 
				position is open for each player'''
		nplayers = len(self.game.players)
		
		# array of the total territory that a player controls, player 1 is index 0
		playerTerritory = [0]*nplayers
		qs = [[] for x in range(nplayers)] # 1 queue for the bfs search for each player

		# insert into queue the start location for the current player
		start = self.directionToNextLocation(self.posX, self.posY, direction)
		# the queue is an array of tuples: ((posx, posy), depth)
		qs[self.playerid-1].append((start, 0)) 
		
		#global hashmap of viewed locations. Updated once per bfs layer depth
		seenLocations = {start:self.playerid}
		
		# insert into queue the start location for the opponents
		for playerid in set(self.game.players)-set([self.playerid]):
			start = self.directionToNextLocation(self.game.players[playerid].posX, 
													self.game.players[playerid].posY, 
													opponentDirection)
			qs[playerid-1].append((start, 0))
		
		# start bfs here
		depth = 0
		while sum(map(len, qs)): #while we still have an element in any queue
			seenThisLayer = {}

			for player in range(nplayers):
				seenThisPlayerLayer = {}

				#loop guard: advance only one bfs layer for each player
				while qs[player] and qs[player][0][1] <= depth:
					a = qs[player].pop(0)
					curloc = a[0]

					if curloc not in seenThisLayer:
						seenThisLayer[curloc] = player # this player owns this location
					else:
						seenThisLayer[curloc] = -1 # already seen this layer, there has be a tie (marked by -1)
					
					# add adjacent open locations to the player's bfs queue
					for dir in range(0,4):
						loc = self.directionToNextLocation(curloc[0], curloc[1], dir)
						if loc not in seenLocations and loc not in seenThisPlayerLayer \
						and not self.game.board.isObstacle(loc[0], loc[1]):
							seenThisPlayerLayer[loc] = 1
							
							# a[1]+1: increase the depth by one for the added locations
							qs[player].append((loc, a[1]+1)) 

			#count territory and update the global seenLocations
			for loc in seenThisLayer:
				player = seenThisLayer[loc]
				seenLocations[loc] = player

				if player >= 0: #not a tie
					playerTerritory[player] += 1

			depth += 1
		
		# debug visualisation for bfs. Drawing logic is in GameBoard.py
		# self.game.board.clearDebugBoard()
		# for loc in seenLocations: self.game.board.debugBoard[loc[1]][loc[0]] = seenLocations[loc]+1

		return playerTerritory[self.playerid-1]

	# place holders for child classes
	def tick(self):
		pass

	def event(self, event):
		pass



class Human(Player):
	def __init__(self, gameObj, color, playerid, x, y, initialDirection, controls):
		super(Human, self).__init__(gameObj, color, playerid, x, y, initialDirection)

		#controls is a 4-tuple, up, right, down, left
		self.ctl_up = controls[0]
		self.ctl_right = controls[1]
		self.ctl_down = controls[2]
		self.ctl_left = controls[3]

	def event(self, event):
		if event.type == pygame.KEYDOWN and len(self.directionQ) < self.maxDirectionQLen:
			if event.key == self.ctl_up:
				self.directionQ.append(Player.UP)
			elif event.key == self.ctl_right:
				self.directionQ.append(Player.RIGHT)
			elif event.key == self.ctl_down:
				self.directionQ.append(Player.DOWN)
			elif event.key == self.ctl_left:
				self.directionQ.append(Player.LEFT)

	def tick(self): #perform moving and collision calculations here

		while self.directionQ and self.wouldCollideSelf(self.directionQ[0]):
			self.directionQ.pop(0)
		if self.directionQ:
			self.direction = self.directionQ.pop(0)
		while self.directionQ and self.directionQ[0] == self.direction:
			self.directionQ.pop(0)

		self.prevPos['x'] = self.posX
		self.prevPos['y'] = self.posY


class Computer(Player):
	def __init__(self, gameObj, color, playerid, x, y, initialDirection):
		super(Computer, self).__init__(gameObj, color, playerid, x, y, initialDirection)

	def tick(self):
		'''select next direction'''
		self.strategyRandom()

	def strategyRandom(self):
		dir = list(range(0, 4))
		
		# 10% chance of changing directions
		if random.randint(1, 10) == 1:
			self.direction = dir[random.randint(0, len(dir)-1)]
		
		# if we would collide, pick a new random direction
		while dir and self.wouldCollide(self.direction):
			self.direction = dir.pop(random.randint(0, len(dir)-1))

class GenComputer(Computer):
	def __init__(self, gameObj, color, playerid, x, y, initialDirection, genome):
		super(GenComputer, self).__init__(gameObj, color, playerid, x, y, initialDirection)
		self.genome = genome

	def tick(self):
		self.strategyGenetic()

	def distanceToSelf(self, direction):
		next_x, next_y = self.directionToNextLocation(self.posX, self.posY, direction)

		min_distance = float('inf')

		for point in self.game.board.getBotTrail(self.playerid):
			trail_x, trail_y = point
			distance = abs(next_x - trail_x) + abs(next_y - trail_y)
			min_distance = min(min_distance, distance)

		return min_distance
	
	def predictTrap(self, direction):
		next_x, next_y = self.directionToNextLocation(self.posX, self.posY, direction)

		temp_board = self.game.board.copy()
		temp_board.grid[next_y][next_x] = self.playerid

		reachable_area = self.calculateReachableArea(next_x, next_y, temp_board)

		trap_threshold = 30

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
		opponentId = (set(self.game.players) - {self.playerid}).pop()

		for direction in range(4):
			if self.wouldCollide(direction):
				continue
				
			if self.predictTrap(direction):
				continue

			max_attempts = 10
			attempts = 0
			opponent_direction = random.choice(range(4))
			while self.game.players[opponentId].wouldCollide(opponent_direction) and attempts < max_attempts:
				opponent_direction = random.choice(range(4))
				attempts += 1

			if attempts >= max_attempts:
				opponent_direction = Player.UP

			distance_to_self = self.distanceToSelf(direction)
			survival = 1 if not self.wouldCollide(direction) else 0
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

# Authors: Greyson Wintergerst and Eileen Hsu
# Description: This file contains the Tron class, which implements the main game loop and game state logic.

import pygame
from GameBoard import GameBoard
from MainMenu import MainMenu
from GameOverMenu import GameOverMenu
from Match import Match
from Players import Player, Human, GenComputer, PruneComputer, Bot, aStarComputer
from DeepQNet import DeepQAgent
import numpy as np
import random
from GeneticTraining import train_genetic
from Stats import StatsScreen


# Game Object that contains the game loop and game state
class Tron:
    def __init__(self, xTiles, yTiles, tileSize):

        # initialize the pygame module
        pygame.init()
        logo = pygame.image.load("./tron_logo.png")
        pygame.display.set_icon(logo)
        pygame.display.set_caption("IntelliTron")

        self.xTiles = xTiles
        self.yTiles = yTiles
        self.tileSize = tileSize

        # Set window size
        self.scr_x = tileSize*xTiles
        self.scr_y = tileSize*yTiles
        self.screen = pygame.display.set_mode((self.scr_x, self.scr_y))

        self.state = 'MAIN_MENU'  # state machine for different game states
        self.board = GameBoard(self, xTiles, yTiles,
                               tileSize)  # Build game board
        self.mainMenu = MainMenu(self)
        self.gameOverMenu = None
        self.match = None
        self.players = {}  # playerId is the key

        # Initialize stats values
        self.PVE_PlayerWins = 0
        self.PVE_BotWins = 0
        self.PVE_Tie = 0

        self.PVP_Player1Wins = 0
        self.PVP_Player2Wins = 0
        self.PVP_Tie = 0

        self.GVG_Bot1Wins = 0
        self.GVG_Bot2Wins = 0
        self.GVG_Tie = 0

        self.AStarVGenetic_AStarWins = 0
        self.AStarVGenetic_GeneticWins = 0
        self.AStarVGenetic_Tie = 0

        self.ABVGenetic_ABWins = 0
        self.ABVGenetic_GeneticWins = 0
        self.ABVGenetic_Tie = 0

        self.NNVGenetic_NNWins = 0
        self.NNVGenetic_GeneticWins = 0
        self.NNVGenetic_Tie = 0

        self.AStarVAB_AStarWins = 0
        self.AStarVAB_ABWins = 0
        self.AStarVAB_Tie = 0

        self.AStarVNN_AStarWins = 0
        self.AStarVNN_NNWins = 0
        self.AStarVNN_Tie = 0

        self.ABVNN_ABWins = 0
        self.ABVNN_NNWins = 0
        self.ABVNN_Tie = 0
        
        self.DqVDq_Dq1Wins = 0
        self.DqVDq_Dq2Wins = 0
        self.DqVDq_Tie = 0
        

        self.AStarTimes = []
        self.ABTimes = []
        self.GeneticTimes = []
        self.NNTimes = []

        self.best_genome = [0.42430839081823557, 0.6795954452427644]

        self.num_tourney_rounds = 30

        self.statsScreen = StatsScreen(self)

        # Start initally at main menu
        self.switchToMenu(self.state)

    # Main event loop for the tron game
    def eventLoop(self):
        clock = pygame.time.Clock()
        active = True
        print("Starting event loop")
        while active:
            # Handle events according to game state
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    active = False
                match self.state:
                    case 'MAIN_MENU':
                        self.mainMenu.eventTick(event)
                    case 'PLAYING':
                        self.match.event(event)
                    case 'TRAINING':
                        self.startDeepTraining() # Deep Q-Learning training
                    case 'TRAINING_GENETIC':
                        self.match.event(event) # Genetic training
                    case 'GAME_OVER':
                        self.gameOverMenu.event(event)
                    case 'STATS_SCREEN':
                        self.statsScreen.event(event)

            # Handle game state transitions
            if self.state == 'MAIN_MENU' or self.state == 'STATS_SCREEN':
                pass
            if self.state == 'PLAYING':
                self.match.tick()

            if self.state == 'TRAINING':
                continue
                
            if self.state == 'TRAINING_GENETIC':
                continue

            pygame.display.flip()
            if self.state == 'PLAYING':
                # fps value of 15 for slower movement (decrease to move slower, increase to move quicker)
                clock.tick(15)
            else:
                clock.tick(60)  # menu fps

        # game not active
        pygame.quit()
        quit()
        
    # Used during DeepQ Training...Resets the game environment and returns the initial state
    def reset(self, model): 
        self.board = GameBoard(self, self.xTiles, self.yTiles, self.tileSize)
        self.players = {
            1: Bot(self, (220, 0, 30), 1, 3, 3, Player.RIGHT),
            2: Bot(self, (30, 220, 0), 2, self.board.xTiles-4, self.board.yTiles-4, Player.LEFT, deepQModel=model)
        }
        self.match = Match(self, 11) # DeepQ Training match (DqVDq Match GameMode)
        self.state = 'TRAINING'
        return self.getEnvState()
    
    # Used during DeepQ Training...Returns the current state of the game environment as a dictionary
    def getEnvState(self):
        # Gather player-related state information (scaled positions and directions)
        players_state = []
        for player in self.players.values():
            players_state.extend([
                player.x / self.board.xTiles, 
                player.y / self.board.yTiles, 
                player.direction / 3
            ])
        
        # Construct the obstacle map (2D array)
        obstacle_map = np.zeros((self.board.xTiles, self.board.yTiles))
        for x in range(self.board.xTiles):
            for y in range(self.board.yTiles):
                if self.board.grid[x][y] != 0:
                    obstacle_map[x][y] = 1

        # Return the state as a dictionary
        return {
            "map": obstacle_map,       # 2D array of 0s and 1s
            "player": players_state       # List of player-related state values
        }
        
    # Used during DeepQ Training...Performs a step in the game environment and returns the next state, rewards, and done flag
    def step(self):
        # perform action for both players
        
        self.match.tick()
        
        # Get the next state
        next_state = self.getEnvState()
        
        # Determine rewards
        current_time = max([player.timeAlive for player in self.players.values()])
        rewards = {player_id: (0, 0) for player_id in self.players}
       
        if not self.match.active:  # Check if the game is over
            for player_id, player in self.players.items():
                if player.alive:
                    opponent = [p for p in self.players.values() if p.ID != player_id][0]
                    rewards[player_id] = 50   # Reward the winner
                else:
                    rewards[player_id] = -50  # Penalize the loser
                    
        else:
            for player_id, player in self.players.items():
                if player.alive:
                    opponent = [p for p in self.players.values() if p.ID != player_id][0]
                    territory = player.calculateDirectionTerritory(player.direction, opponent.direction) # Calculate predicted territory
                    distance_to_wall = player.distanceToClosestWall(player.x, player.y) # Calculate distance to wall
                    rewards[player_id] = 0.5 + 0.0001 * territory + 0.0005 * (distance_to_wall - 10)

        # Check if the game is over
        done = not self.match.active
        

        return next_state, rewards, done
    
    # Used during DeepQ Training...Starts the Deep Q-Learning training process
    # Initializes an instance of DeepQAgent with the below parameters
    def startDeepTraining(self):
        print("Starting Deep Q-Learning training")
        state_size = len(self.getEnvState())
        action_size = 4
        episodes = 50
        hidden_size = 128
        gamma = 0.95
        epsilon = 1.0
        epsilon_min = 0.01
        epsilon_decay = 0.9975
        alpha = 0.0001
        batch_size = 64
        
        self.deepQAgent = DeepQAgent(self, state_size, action_size, hidden_size, gamma, epsilon, epsilon_min, epsilon_decay, alpha)
        self.deepQAgent.train(env=self, episodes=episodes, batch_size=batch_size, playerid=2) # makes player 2 the learning bot
        
    
        

    # Redraws the pygame screen to a menu based on the state
    def switchToMenu(self, state):
        self.screen.fill((0, 0, 0))
        self.state = state
        if self.state == 'MAIN_MENU':
            self.mainMenu.draw()
        elif self.state == 'STATS_SCREEN':
            self.statsScreen.draw()
        else:
            self.gameOverMenu.draw()

    # Starts a new match (or displays stats page) based on the match type
    # Establishes the player and game board
    # matchType: 0 - PVP, 1 - PVE, 2 - Genetic Training, 3 - Deep Q-Learning Training, 4 - Deep Q-Learning vs. Deep Q-Learning, 5 - Tournament, 6 - Stats
    def startMatch(self, matchType):
        self.board = GameBoard(self, self.xTiles, self.yTiles, self.tileSize)
        self.players = {}
        # Create players/Computers per match type.
        # PVP - 2 humans
        if matchType == 0:
            self.players[1] = Human(self, (220, 0, 30), 1, 3, 3, Player.RIGHT,
                                    (pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d))
            self.players[2] = Human(self, (30, 220, 0), 2, self.board.xTiles-4, self.board.yTiles-4,
                                    Player.LEFT, (pygame.K_UP, pygame.K_LEFT, pygame.K_DOWN, pygame.K_RIGHT))
        # PVE - 1 human and 1 bot 
        elif matchType == 1:
            self.players[1] = Human(self, (220, 0, 30), 1, 3, 3, Player.RIGHT, (
                pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d))
            self.players[2] = Bot(
                self, (30, 220, 0), 2, self.board.xTiles-4, self.board.yTiles-4, Player.LEFT) # currently defaults to Alpha-Beta Pruning
            
        # Genetic Training - 2 bots
        elif matchType == 2:
            self.state = 'TRAINING_GENETIC'
            self.start_training(self)
            
        # Deep Q-Learning Training - 1 Territory-Based Bot and 1 Deep Q-Learning Bot
        elif matchType == 3:
            self.state = 'TRAINING'
            
        # Player vs. DeepQ - 1 human and 1 Deep Q-Learning Bot
        elif matchType == 4:
            trainedDeepQ = DeepQAgent(3, self.getEnvState(), 4, 0.95, 1.0, 0.01, 0.995, 0.01) # Initialize Deep Q-Learning model
            trainedDeepQ.load("deepq_model.pth") # Load trained Deep Q-Learning model
            self.players[1] = Human(self, (220, 0, 30), 1, 3, 3, Player.RIGHT, (
                pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d))
            self.players[2] = Bot(self, (90, 220, 50), 2, self.board.xTiles-4, self.board.yTiles-4, Player.LEFT, deepQModel=trainedDeepQ)
            self.match = Match(self, 11)
            self.state = 'PLAYING'

        # Tournament - A series of matches between different bots
        elif matchType == 5:
            self.state = 'PLAYING'
            self.tournament()
            
        # Stats - Display the stats screen
        elif matchType == 6:
            self.switchToMenu('STATS_SCREEN')

        # Set state to 'PLAYING' for all other match types (for use with tournament logic)
        if (matchType != 2 and matchType != 3 and matchType != 4 and matchType != 5 and matchType != 6):
            # Build match and set program state
            self.match = Match(self, matchType)
            self.state = 'PLAYING'

    # Tournament logic for comparing different bots - FIXME E.S.
    def tournament(self):
        # A star vs Genetic
        for i in range(self.num_tourney_rounds):
            self.board = GameBoard(
                self, self.xTiles, self.yTiles, self.tileSize)
            self.players = {}
            self.players[1] = aStarComputer(  # added astar
                self, (220, 0, 30), 1, 3, 3, Player.RIGHT)
            self.players[2] = GenComputer(
                self, (90, 220, 50), 2, self.board.xTiles - 4, self.board.yTiles - 4, Player.LEFT, self.best_genome)

            match = Match(self, 10)
            clock = pygame.time.Clock()
            ticks = 0
            while match.active:
                match.tick()
                ticks += 1
                self.screen.fill((0, 0, 0))
                self.board.drawGrid()
                pygame.display.flip()
                clock.tick(100)


            self.AStarTimes.append(ticks)
            self.GeneticTimes.append(ticks)
        

        # Alpha-Beta vs Genetic
        for i in range(self.num_tourney_rounds):
            self.board = GameBoard(
                self, self.xTiles, self.yTiles, self.tileSize)
            self.players = {}
            self.players[1] = PruneComputer(
                self, (220, 0, 30), 1, 3, 3, Player.RIGHT)
            self.players[2] = GenComputer(
                self, (90, 220, 50), 2, self.board.xTiles - 4, self.board.yTiles - 4, Player.LEFT, self.best_genome)

            match = Match(self, 5)
            clock = pygame.time.Clock()
            ticks = 0
            while match.active:
                match.tick()
                ticks += 1
                self.screen.fill((0, 0, 0))
                self.board.drawGrid()
                pygame.display.flip()
                clock.tick(100)

            self.ABTimes.append(ticks)
            self.GeneticTimes.append(ticks)

        # Neural Network vs Genetic
        for i in range(self.num_tourney_rounds):
            self.board = GameBoard(
                self, self.xTiles, self.yTiles, self.tileSize)
            self.players = {}
            trainedDeepQ = DeepQAgent(3, self.getEnvState(), 4, 0.95, 1.0, 0.01, 0.995, 0.01)
            trainedDeepQ.load("deepq_model.pth")
            self.players[1] = Bot(self, (220, 0, 30), 1, 3, 3, Player.RIGHT, deepQModel=trainedDeepQ)
            self.players[2] = GenComputer(
                self, (90, 220, 50), 2, self.board.xTiles - 4, self.board.yTiles - 4, Player.LEFT, self.best_genome)

            match = Match(self, 6)
            clock = pygame.time.Clock()
            ticks = 0
            while match.active:
                match.tick()
                ticks += 1
                self.screen.fill((0, 0, 0))
                self.board.drawGrid()
                pygame.display.flip()
                clock.tick(100)

            self.GeneticTimes.append(ticks)
            self.NNTimes.append(ticks)

        # A star vs. Alpha-Beta
        for i in range(self.num_tourney_rounds):
            self.board = GameBoard(
                self, self.xTiles, self.yTiles, self.tileSize)
            self.players = {}
            self.players[1] = aStarComputer(
                self, (220, 0, 30), 1, 3, 3, Player.RIGHT)
            self.players[2] = PruneComputer(self, (90, 220, 50), 2, self.board.xTiles - 4,
                                            self.board.yTiles - 4, Player.LEFT)  # FIXME - add Alpha-Beta bot here

            match = Match(self, 7)
            clock = pygame.time.Clock()
            ticks = 0
            while match.active:
                match.tick()
                ticks += 1
                self.screen.fill((0, 0, 0))
                self.board.drawGrid()
                pygame.display.flip()
                clock.tick(100)


            self.AStarTimes.append(ticks)
            self.ABTimes.append(ticks)
        

        # A star vs. Neural Network
        for i in range(self.num_tourney_rounds):
            self.board = GameBoard(
                self, self.xTiles, self.yTiles, self.tileSize)
            self.players = {}
            trainedDeepQ = DeepQAgent(3, self.getEnvState(), 4, 0.95, 1.0, 0.01, 0.995, 0.01)
            trainedDeepQ.load("deepq_model.pth")
            self.players[1] = aStarComputer(
                self, (220, 0, 30), 1, 3, 3, Player.RIGHT)
            self.players[2] = Bot(self, (90, 220, 50), 2, self.board.xTiles - 4,
                                  self.board.yTiles - 4, Player.LEFT, deepQModel=trainedDeepQ)

            match = Match(self, 8)
            clock = pygame.time.Clock()
            ticks = 0
            while match.active:
                match.tick()
                ticks += 1
                self.screen.fill((0, 0, 0))
                self.board.drawGrid()
                pygame.display.flip()
                clock.tick(100)

            self.AStarTimes.append(ticks)
            self.NNTimes.append(ticks)

        # Alpha-Beta vs. Neural Network
        for i in range(self.num_tourney_rounds):
            self.board = GameBoard(
                self, self.xTiles, self.yTiles, self.tileSize)
            self.players = {}
            trainedDeepQ = DeepQAgent(3, self.getEnvState(), 4, 0.95, 1.0, 0.01, 0.995, 0.01)
            trainedDeepQ.load("deepq_model.pth")
            self.players[1] = PruneComputer(
                self, (220, 0, 30), 1, 3, 3, Player.RIGHT)
            self.players[2] = Bot(self, (90, 220, 50), 2, self.board.xTiles - 4,
                                  self.board.yTiles - 4, Player.LEFT, deepQModel=trainedDeepQ)

            match = Match(self, 9)
            clock = pygame.time.Clock()
            ticks = 0
            while match.active:
                match.tick()
                ticks += 1
                self.screen.fill((0, 0, 0))
                self.board.drawGrid()
                pygame.display.flip()
                clock.tick(100)

            self.ABTimes.append(ticks)
            self.NNTimes.append(ticks)

    # Initializes a population with 20 random genomes and starts the process for training the genetic algorithm
    def start_training(self, tron):
        initial_population = [[random.random() for _ in range(2)]
                              for _ in range(20)]
        print("Starting genetic algorithm training...")

        trained_population = train_genetic(
            initial_population,
            generations=50,
            mutation_rate=0.1,
            simulate_game=tron.simulate_genetic
        )

        print("Training complete!")
        best_genome = trained_population[0]
        self.best_genome = best_genome
        print(f"Best genome: {best_genome}")
        self.screen.fill((0, 0, 0))
        self.switchToMenu("MAIN_MENU")

    # Simulates a match between 2 agents using the genetic algorithm. One agent uses genome1 and the other uses genome2. 
    # Returns the survival time of the agent using genome1 (the genome being evaluated), whether the agent won or lost,
    # the state of the agent and the board at the end of the game.
    def simulate_genetic(self, genome1, genome2):
        self.board = GameBoard(self, self.xTiles, self.yTiles, self.tileSize)
        self.players = {}
        self.players[1] = GenComputer(
            self, (220, 0, 30), 1, 3, 3, Player.RIGHT, genome1)
        self.players[2] = GenComputer(
            self, (90, 220, 50), 2, self.board.xTiles - 4, self.board.yTiles - 4, Player.LEFT, genome2)

        match = Match(self, 2)

        match_ticks = 0
        clock = pygame.time.Clock()

        while match.active:
            match.tick()

            self.screen.fill((0, 0, 0))
            self.board.drawGrid()
            pygame.display.flip()
            clock.tick(100)

            match_ticks += 1

        bot = self.players[1]
        opponent = self.players[2]

        survival_time = match_ticks if bot.alive else match_ticks // 2
        wins = 1 if bot.alive and not opponent.alive else 0

        return {
            'survival_time': survival_time,
            'wins': wins,
            'board_state': self.board.copy(),
            'bot': bot
        }


if __name__ == '__main__':
    tron = Tron(40, 40, 20)
    print("Game initialized")
    tron.eventLoop()

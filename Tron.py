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

        self.best_genome = [0.42430839081823557, 0.6795954452427644, 0.8502043539767576]

        self.num_tourney_rounds = 10

        self.statsScreen = StatsScreen(self)

        # Start initally at main menu
        self.switchToMenu(self.state)

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
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_t: 
                            self.state = 'TRAINING'
                    case 'PLAYING':
                        self.match.event(event)
                    case 'TRAINING':
                        self.startDeepTraining()
                    case 'TRAINING_GENETIC':
                        self.match.event(event)
                    case 'GAME_OVER':
                        self.gameOverMenu.event(event)
                    case 'STATS_SCREEN':
                        self.statsScreen.event(event)

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
        
    def reset(self, model): 
        # reset game environment and return initial state
        self.board = GameBoard(self, self.xTiles, self.yTiles, self.tileSize)
        self.players = {
            1: Bot(self, (220, 0, 30), 1, 3, 3, Player.RIGHT),
            2: Bot(self, (30, 220, 0), 2, self.board.xTiles-4, self.board.yTiles-4, Player.LEFT, deepQModel=model)
        }
        self.match = Match(self, 11) # DeepQ Training match (DqVDq Match GameMode)
        self.state = 'TRAINING'
        return self.getEnvState()
    
    # def getEnvState(self):
    #     # return the numerical representation of the game environment
        
    #     env_state = []
    #     for player in self.players.values():
    #         env_state.extend([player.posX / self.board.xTiles, player.posY / self.board.yTiles, player.direction / 3])
            
    #     # calculate a board state, simplifying it to include a 1 if the tile is occupied by a player or wall, 0 otherwise
    #     obstacle_map = np.zeros((self.board.xTiles, self.board.yTiles))
    #     for x in range(self.board.xTiles):
    #         for y in range(self.board.yTiles):
    #             if self.board.grid[x][y] != 0:
    #                 obstacle_map[x][y] = 1
        
        
    #     env_state.extend(np.array(obstacle_map).flatten())
        
    #     # print(f"Env state, {len(env_state)},  {env_state}")
        
        
    #     return env_state
    
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

        # Return the state as a dictionary for better structure
        return {
            "map": obstacle_map,       # 2D array of 0s and 1s
            "player": players_state       # List of player-related state values
        }
        
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
                    rewards[player_id] = 100   # Reward the winner
                else:
                    rewards[player_id] = -150  # Penalize the loser
                    
        else:
            for player_id, player in self.players.items():
                if player.alive:
                    opponent = [p for p in self.players.values() if p.ID != player_id][0]
                    territory = player.calculateDirectionTerritory(player.direction, opponent.direction)
                    distance_to_wall = player.distanceToClosestWall(player.x, player.y)
                    rewards[player_id] = 0.5 + 0.001 * territory + 0.005 * distance_to_wall

        # Check if the game is over
        done = not self.match.active
        

        return next_state, rewards, done
    
    def startDeepTraining(self):
        print("Starting Deep Q-Learning training")
        state_size = len(self.getEnvState())
        action_size = 4
        episodes = 200
        hidden_size = 128
        gamma = 0.95
        epsilon = 1.0
        epsilon_min = 0.01
        epsilon_decay = 0.9975
        alpha = 0.005
        batch_size = 32
        
        self.deepQAgent = DeepQAgent(self, state_size, action_size, hidden_size, gamma, epsilon, epsilon_min, epsilon_decay, alpha)
        self.deepQAgent.train(env=self, episodes=episodes, batch_size=batch_size, playerid=2)
        
    
        

    def switchToMenu(self, state):
        self.screen.fill((0, 0, 0))
        self.state = state
        if self.state == 'MAIN_MENU':
            self.mainMenu.draw()
        elif self.state == 'STATS_SCREEN':
            self.statsScreen.draw()
        else:
            self.gameOverMenu.draw()

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
                self, (30, 220, 0), 2, self.board.xTiles-4, self.board.yTiles-4, Player.LEFT)
        elif matchType == 2:
            self.state = 'TRAINING_GENETIC'
            self.start_training(self)
        elif matchType == 3:
            trainedDeepQ = DeepQAgent(3, self.getEnvState(), 4, 0.95, 1.0, 0.01, 0.995, 0.01)
            trainedDeepQ.load("deepq_model.pth")
            self.players[1] = Human(self, (220, 0, 30), 1, 3, 3, Player.RIGHT, (
                pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d))
            self.players[2] = Bot(self, (90, 220, 50), 2, self.board.xTiles-4, self.board.yTiles-4, Player.LEFT, deepQModel=trainedDeepQ)

            
        elif matchType == 4:
            self.state = 'PLAYING'
            self.tournament()
        elif matchType == 5:
            self.switchToMenu('STATS_SCREEN')


        if (matchType != 2 and matchType != 3 and matchType != 4 and matchType != 5):
            # Build match and set program state
            self.match = Match(self, matchType)
            self.state = 'PLAYING'

    def tournament(self):
        # A star vs Genetic
        for i in range(self.num_tourney_rounds):
            self.board = GameBoard(
                self, self.xTiles, self.yTiles, self.tileSize)
            self.players = {}
            # FIXME - add A star bot here
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
            # FIXME - add Alpha-Beta bot here
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
            # FIXME - add Neural Network bot here
            self.players[1] = Bot(self, (220, 0, 30), 1, 3, 3, Player.RIGHT)
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
            # FIXME - add A star bot here
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
            # FIXME - add A star bot here
            self.players[1] = aStarComputer(
                self, (220, 0, 30), 1, 3, 3, Player.RIGHT)
            self.players[2] = Bot(self, (90, 220, 50), 2, self.board.xTiles - 4,
                                  self.board.yTiles - 4, Player.LEFT)  # FIXME - add Neural Network bot here

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
            # FIXME - add Alpha-Beta bot here
            self.players[1] = PruneComputer(
                self, (220, 0, 30), 1, 3, 3, Player.RIGHT)
            self.players[2] = Bot(self, (90, 220, 50), 2, self.board.xTiles - 4,
                                  self.board.yTiles - 4, Player.LEFT)  # FIXME - add Neural Network bot here

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

    def start_training(self, tron):
        initial_population = [[random.random() for _ in range(3)]
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

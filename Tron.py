import pygame
from GameBoard import GameBoard
from MainMenu import MainMenu
from GameOverMenu import GameOverMenu
from Match import Match
from Players import Player, Human, GenComputer, Bot
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
                    case 'PLAYING':
                        self.match.event(event)
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
            if self.state == 'TRAINING_GENETIC':
                continue
            
            pygame.display.flip()
            if self.state == 'PLAYING':
                clock.tick(15) # fps value of 15 for slower movement (decrease to move slower, increase to move quicker)
            else: 
                clock.tick(60) # menu fps
                
        # game not active
        pygame.quit()
        quit()

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
            self.players[1] = Human(self, (220, 0, 30), 1, 3, 3, Player.RIGHT, (pygame.K_UP, pygame.K_LEFT, pygame.K_DOWN, pygame.K_RIGHT))
            self.players[2] = Bot(self, (30, 220, 0), 2, self.board.xTiles-4, self.board.yTiles-4, Player.LEFT)
        elif matchType == 2:
            self.state = 'TRAINING_GENETIC'
            self.start_training(self)
        elif matchType == 3:
            self.state = 'PLAYING'
            self.tournament()
        elif matchType == 4:
            self.switchToMenu('STATS_SCREEN')

        if (matchType != 2 and matchType != 3 and matchType != 4):
            #Build match and set program state
            self.match = Match(self, matchType)
            self.state = 'PLAYING'

    def tournament(self):
        # A star vs Genetic
        for i in range(self.num_tourney_rounds):
            self.board = GameBoard(self, self.xTiles, self.yTiles, self.tileSize)
            self.players = {}
            self.players[1] = Bot(self, (220, 0, 30), 1, 3, 3, Player.RIGHT) # FIXME - add A star bot here
            self.players[2] = GenComputer(self, (90, 220, 50), 2, self.board.xTiles - 4, self.board.yTiles - 4, Player.LEFT, self.best_genome)

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
            self.board = GameBoard(self, self.xTiles, self.yTiles, self.tileSize)
            self.players = {}
            self.players[1] = Bot(self, (220, 0, 30), 1, 3, 3, Player.RIGHT) # FIXME - add Alpha-Beta bot here
            self.players[2] = GenComputer(self, (90, 220, 50), 2, self.board.xTiles - 4, self.board.yTiles - 4, Player.LEFT, self.best_genome)

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
            self.board = GameBoard(self, self.xTiles, self.yTiles, self.tileSize)
            self.players = {}
            self.players[1] = Bot(self, (220, 0, 30), 1, 3, 3, Player.RIGHT) # FIXME - add Neural Network bot here
            self.players[2] = GenComputer(self, (90, 220, 50), 2, self.board.xTiles - 4, self.board.yTiles - 4, Player.LEFT, self.best_genome)

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
            self.board = GameBoard(self, self.xTiles, self.yTiles, self.tileSize)
            self.players = {}
            self.players[1] = Bot(self, (220, 0, 30), 1, 3, 3, Player.RIGHT) # FIXME - add A star bot here
            self.players[2] = Bot(self, (90, 220, 50), 2, self.board.xTiles - 4, self.board.yTiles - 4, Player.LEFT) # FIXME - add Alpha-Beta bot here

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
            self.board = GameBoard(self, self.xTiles, self.yTiles, self.tileSize)
            self.players = {}
            self.players[1] = Bot(self, (220, 0, 30), 1, 3, 3, Player.RIGHT) # FIXME - add A star bot here
            self.players[2] = Bot(self, (90, 220, 50), 2, self.board.xTiles - 4, self.board.yTiles - 4, Player.LEFT) # FIXME - add Neural Network bot here

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
            self.board = GameBoard(self, self.xTiles, self.yTiles, self.tileSize)
            self.players = {}
            self.players[1] = Bot(self, (220, 0, 30), 1, 3, 3, Player.RIGHT) # FIXME - add Alpha-Beta bot here
            self.players[2] = Bot(self, (90, 220, 50), 2, self.board.xTiles - 4, self.board.yTiles - 4, Player.LEFT) # FIXME - add Neural Network bot here

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
        initial_population = [[random.random() for _ in range(3)] for _ in range(20)]
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
        self.screen.fill((0,0,0))
        self.switchToMenu("MAIN_MENU")

    def simulate_genetic(self, genome1, genome2):
        self.board = GameBoard(self, self.xTiles, self.yTiles, self.tileSize)
        self.players = {}
        self.players[1] = GenComputer(self, (220, 0, 30), 1, 3, 3, Player.RIGHT, genome1)
        self.players[2] = GenComputer(self, (90, 220, 50), 2, self.board.xTiles - 4, self.board.yTiles - 4, Player.LEFT, genome2)

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

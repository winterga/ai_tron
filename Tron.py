import pygame
from GameBoard import GameBoard
from MainMenu import MainMenu
from GameOverMenu import GameOverMenu
from Match import Match
from Players import Player, Human, Bot
from DeepQNet import DeepQAgent
import numpy as np




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
        self.EVE_Bot1Wins = 0
        self.EVE_Bot2Wins = 0
        self.EVE_Tie = 0

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
                    case 'GAME_OVER':
                        self.gameOverMenu.event(event)

            if self.state == 'MAIN_MENU':
                pass
            if self.state == 'PLAYING':
                self.match.tick()
            if self.state == 'TRAINING':
                continue
                

            pygame.display.flip()
            if self.state == 'PLAYING':
                # fps value of 10 for slower movement (decrease to move slower, increase to move quicker)
                clock.tick(10)
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
        self.match = Match(self, 2) # EVE match
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
                player.posX / self.board.xTiles, 
                player.posY / self.board.yTiles, 
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
                    opponent = [p for p in self.players.values() if p.playerid != player_id][0]
                    territory = player.calculateDirectionTerritory(player.direction, opponent.direction)
                    rewards[player_id] = 100   # Reward the winner
                else:
                    rewards[player_id] = -150  # Penalize the loser
                    
        else:
            for player_id, player in self.players.items():
                if player.alive:
                    opponent = [p for p in self.players.values() if p.playerid != player_id][0]
                    territory = player.calculateDirectionTerritory(player.direction, opponent.direction)
                    distance_to_wall = player.distanceToWall(player.posX, player.posY, player.direction)
                    rewards[player_id] = 1 + 0.001 * territory + 0.005 * distance_to_wall

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
        self.state = state
        if self.state == 'MAIN_MENU':
            self.mainMenu.draw()
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
            # EVE - 2 bots
        elif matchType == 2:
            self.players[1] = Bot(
                self, (220, 0, 30), 1, 3, 3, Player.RIGHT)
            self.players[2] = Bot(
                self, (90, 220, 50), 2, self.board.xTiles-4, self.board.yTiles-4, Player.LEFT)
        elif matchType == 3:
            trainedDeepQ = DeepQAgent(3, self.getEnvState(), 4, 0.95, 1.0, 0.01, 0.995, 0.01)
            trainedDeepQ.load("deepq_model.pth")
            self.players[1] = Human(self, (220, 0, 30), 1, 3, 3, Player.RIGHT, (
                pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d))
            self.players[2] = Bot(self, (90, 220, 50), 2, self.board.xTiles-4, self.board.yTiles-4, Player.LEFT, deepQModel=trainedDeepQ)

            # Build match and set program state
        self.match = Match(self, matchType)
        self.state = 'PLAYING'


if __name__ == '__main__':
    tron = Tron(40, 40, 20)
    print("Game initialized")
    tron.eventLoop()

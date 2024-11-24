import pygame
from GameBoard import GameBoard
from MainMenu import MainMenu
from GameOverMenu import GameOverMenu
from Match import Match
from Players import Player, Human, Computer




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

		#Set window size
        self.scr_x = tileSize*xTiles
        self.scr_y = tileSize*yTiles
        self.screen = pygame.display.set_mode((self.scr_x, self.scr_y))
        
        self.state = 'MAIN_MENU' #state machine for different game states
        self.board = GameBoard(self, xTiles, yTiles, tileSize) #Build game board
        self.mainMenu = MainMenu(self)
        self.gameOverMenu = None
        self.match = None
        self.players = {} # playerId is the key

		#Initialize stats values
        self.PVE_PlayerWins = 0
        self.PVE_BotWins = 0
        self.PVE_Tie = 0
        self.PVP_Player1Wins = 0
        self.PVP_Player2Wins = 0
        self.PVP_Tie = 0
        self.EVE_Bot1Wins = 0
        self.EVE_Bot2Wins = 0
        self.EVE_Tie = 0

		#Start initally at main menu
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
                    case 'GAME_OVER':
                        self.gameOverMenu.event(event)
                        
            if self.state == 'MAIN_MENU':
                pass
            if self.state == 'PLAYING':
                self.match.tick()
            
            pygame.display.flip()
            if self.state == 'PLAYING':
                clock.tick(10) # fps value of 10 for slower movement (decrease to move slower, increase to move quicker)
            else: 
                clock.tick(60) # menu fps
                
        # game not active
        pygame.quit()
        quit()
        
    def switchToMenu(self, state):
        self.state = state
        if self.state == 'MAIN_MENU':
            self.mainMenu.draw()
        else:
            self.gameOverMenu.draw()
        
    def startMatch(self, matchType):
        self.board = GameBoard(self, self.xTiles, self.yTiles, self.tileSize)
        self.players = {}
		#Create players/Computers per match type.
		#PVP - 2 humans
        if matchType == 0:
            self.players[1] = Human(self, (220, 0, 30), 1, 3, 3, Player.RIGHT, (pygame.K_w, pygame.K_d, pygame.K_s, pygame.K_a))
            self.players[2] = Human(self, (30, 220, 0), 2, self.board.xTiles-4, self.board.yTiles-4, Player.LEFT, (pygame.K_UP, pygame.K_RIGHT, pygame.K_DOWN, pygame.K_LEFT))
		#PVE - 1 human and 1 bot
        elif matchType == 1:
            self.players[1] = Human(self, (220, 0, 30), 1, 3, 3, Player.RIGHT, (pygame.K_UP, pygame.K_RIGHT, pygame.K_DOWN, pygame.K_LEFT))
            self.players[2] = Computer(self, (30, 220, 0), 2, self.board.xTiles-4, self.board.yTiles-4, Player.LEFT)
		#EVE - 2 bots
        elif matchType == 2:
            self.players[1] = Computer(self, (220, 0, 30), 1, 3, 3, Player.RIGHT)
            self.players[2] = Computer(self, (90, 220, 50), 2, self.board.xTiles-4, self.board.yTiles-4, Player.LEFT)

		#Build match and set program state
        self.match = Match(self, matchType)
        self.state = 'PLAYING'
        
    
if __name__ == '__main__':
    tron = Tron(40, 40, 20)
    print("Game initialized")
    tron.eventLoop()
            
                        
                        
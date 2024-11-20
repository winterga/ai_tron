import pygame
from GameBoard import GameBoard
from MainMenu import MainMenu
from GameOverMenu import GameOverMenu
from Match import Match





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
        

        
    
if __name__ == '__main__':
    tron = Tron(40, 40, 20)
    print("Game initialized")
    tron.eventLoop()
            
                        
                        
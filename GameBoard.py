import pygame

class GameBoard:
    def __init__(self, gameObj, xTiles, yTiles, tileSize):
        self.gameObj = gameObj
        self.xTiles = xTiles
        self.yTiles = yTiles
        self.tileSize = tileSize

        self.grid = [[0 for x in range(xTiles)] for y in range(yTiles)]


        ## Possible debugging territory

        # Set up board with walls
        for x in range(xTiles):
            self.grid[0][x] = -1
            self.grid[yTiles-1][x] = -1

        for y in range(xTiles):
            self.grid[y][0] = -1
            self.grid[y][yTiles-1] = -1

        
        def addRectObstacle(self, x1, y1, x2, y2):
            # Rectagles are defined by the top left and bottom right coordinates
            # Define with two points [top-left] (x1, y1), [bottom-right] (x2, y2)
            for x in range(x1, x2+1):
                self.grid[y1][x] = -1
                self.grid[y2][x] = -1
            for y in range(y2, y1+1):
                self.grid[y][x1] = -1
                self.grid[y][x2] = -1

        
        def drawGrid(self):
            for i in range(self.xTiles):
                for j in range(self.yTiles):
                    # load coordinates for rectangles for pygame to draw (using [x, y, w, h] format)
                    x = i*self.tileSize+1
                    y = j*self.tileSize+1
                    w = self.tileSize
                    h = self.tileSize
                    
                    color = None

                    if self.grid[i][j] == -1:
                        color = (148, 156, 168)
                    elif self.grid[i][j] == 0:
                        color = (0,0,0) # empty space yet to be filled
                    else: # player line
                        color = self.game.players[self.grid[i][j]].color

                
                    pygame.draw.rect(self.game.screen, color=color, rect=(x,y,w,h))
                    
        def isObstacle(self, x, y):
            if (x < 0 or x > self.xTiles-1 or y < 0 or y > self.yTiles-1):
                return True
            else:
                return (self.grid[x][y] != 0)
import pygame

class Match:
    def __init__(self, gameObj, gameMode):
        self.gameObj = gameObj
        self.numPlayers = len(self.gameObj.players)
        self.active = True
        self.gameMode = gameMode

    
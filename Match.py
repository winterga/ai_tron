# Author: Eileen Hsu
# Description: This file contains the Match class, which implements logic for handling gameplay mechanisms.

from GameOverMenu import GameOverMenu

class Match:
    def __init__(self, gameObj, gameMode):
        self.gameObj = gameObj
        self.active = True
        self.gameMode = gameMode

    def tick(self):
        # get next move for each player
        for player in self.gameObj.players:
            self.gameObj.players[player].tick()
            self.gameObj.players[player].timeAlive += 1

        # check if they collide with each other at the same time
        if self.checkTie():
            self.checkStatus(True)
            return

        # check for any other collision situations and update positions
        for player in self.gameObj.players:
            self.gameObj.players[player].checkCollision(self.gameObj.players[player].direction)
            self.gameObj.players[player].movePlayer()
            
        # check if anyone is dead
        self.checkStatus(False)

        # render the screen
        if self.active:
            self.gameObj.screen.fill((0, 0, 0))
            self.gameObj.board.drawGrid()

    def event(self, event):
        for player in self.gameObj.players:
            self.gameObj.players[player].event(event)

    # check the next positions of both players; if they are the same, draw a tie square for visualization
    def checkTie(self):
        nextPositions = []
        for player in self.gameObj.players:

            nextPosition = self.gameObj.players[player].convertDirectionToLocation(
                self.gameObj.players[player].x,
                self.gameObj.players[player].y,
                self.gameObj.players[player].direction)

            if nextPosition in nextPositions:
                self.gameObj.board.drawTieSquare(
                    nextPosition[0], nextPosition[1])
                return True
            else:
                nextPositions.append(nextPosition)
        return False

    # check if any players are dead and if so, update stats and transition to game-over menu
    def checkStatus(self, tieStatus):
        aliveCount = 0

        for player in self.gameObj.players:
            if self.gameObj.players[player].alive:
                aliveCount += 1

        # if both are dead (i.e. tie)
        if tieStatus or aliveCount == 0:

            self.active = False
            print("Tie detected or no players alive.")
            # Update stats depending on the game mode
            if self.gameMode == 1:
                self.gameObj.PVE_Tie += 1
            elif self.gameMode == 0:
                self.gameObj.PVP_Tie += 1
            elif self.gameMode == 2:
                self.gameObj.GVG_Tie += 1
            elif self.gameMode == 10:
                self.gameObj.AStarVGenetic_Tie += 1
            elif self.gameMode == 5:
                self.gameObj.ABVGenetic_Tie += 1
            elif self.gameMode == 6:
                self.gameObj.NNVGenetic_Tie += 1
            elif self.gameMode == 7:
                self.gameObj.AStarVAB_Tie += 1
            elif self.gameMode == 8:
                self.gameObj.AStarVNN_Tie += 1
            elif self.gameMode == 9:
                self.gameObj.ABVNN_Tie += 1
            elif self.gameMode == 11:
                self.gameObj.DqVDq_Tie += 1
            elif self.gameMode == 12:
                self.gameObj.PlayerVDq_Tie += 1

            self.gameObj.gameOverMenu = GameOverMenu(
                self.gameObj, "Nobody", self.gameMode)
            self.gameObj.switchToMenu("GAME_OVER")
        # if one is dead
        elif aliveCount == 1:
            self.active = False
            print("Single player alive. Ending match.")
            winner = ""
            for player in self.gameObj.players:
                if self.gameObj.players[player].alive:
                    winner = player
                    break
            
            # update stats based on game mode
            if self.gameMode == 1:
                if winner == 1:
                    self.gameObj.PVE_PlayerWins += 1
                elif winner == 2:
                    self.gameObj.PVE_BotWins += 1
            elif self.gameMode == 0:
                if winner == 1:
                    self.gameObj.PVP_Player1Wins += 1
                elif winner == 2:
                    self.gameObj.PVP_Player2Wins += 1
            elif self.gameMode == 2:
                if winner == 1:
                    self.gameObj.GVG_Bot1Wins += 1
                elif winner == 2:
                    self.gameObj.GVG_Bot2Wins += 1
            elif self.gameMode == 10:
                if winner == 1:
                    self.gameObj.AStarVGenetic_AStarWins += 1
                elif winner == 2:
                    self.gameObj.AStarVGenetic_GeneticWins += 1
            elif self.gameMode == 5:
                if winner == 1:
                    self.gameObj.ABVGenetic_ABWins += 1
                elif winner == 2:
                    self.gameObj.ABVGenetic_GeneticWins += 1
            elif self.gameMode == 6:
                if winner == 1:
                    self.gameObj.NNVGenetic_NNWins += 1
                elif winner == 2:
                    self.gameObj.NNVGenetic_GeneticWins += 1
            elif self.gameMode == 7:
                if winner == 1:
                    self.gameObj.AStarVAB_AStarWins += 1
                elif winner == 2:
                    self.gameObj.AStarVAB_ABWins += 1
            elif self.gameMode == 8:
                if winner == 1:
                    self.gameObj.AStarVNN_AStarWins += 1
                elif winner == 2:
                    self.gameObj.AStarVNN_NNWins += 1
            elif self.gameMode == 9:
                if winner == 1:
                    self.gameObj.ABVNN_ABWins += 1
                elif winner == 2:
                    self.gameObj.ABVNN_NNWins += 1
                    
            elif self.gameMode == 11:
                if winner == 1:
                    self.gameObj.DqVDq_Dq1Wins += 1
                elif winner == 2:
                    self.gameObj.DqVDq_Dq2Wins += 1
            elif self.gameMode == 12:
                if winner == 1:
                    self.gameObj.PlayerVDq_PlayerWins += 1
                elif winner == 2:
                    self.gameObj.PlayerVDq_DqWins += 1

            self.gameObj.gameOverMenu = GameOverMenu(
                self.gameObj, "Player " + str(winner), self.gameMode)
            self.gameObj.switchToMenu("GAME_OVER")

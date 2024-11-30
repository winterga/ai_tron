from GameOverMenu import GameOverMenu


class Match:
    def __init__(self, gameObj, gameMode):
        self.gameObj = gameObj
        self.active = True
        self.gameMode = gameMode

    def tick(self):
        # get next move
        print(self.gameObj.state)
        for player in self.gameObj.players:
            self.gameObj.players[player].tick(self.gameObj.state)
            self.gameObj.players[player].timeAlive += 1

        # check if they collide with each at same time
        if self.checkTie():
            self.checkStatus(True)
            return

        # check for any other collision situations and update positions
        for player in self.gameObj.players:

            self.gameObj.players[player].checkForCollision(self.gameObj.players[player].direction)

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

    def checkTie(self):
        nextPositions = []
        for player in self.gameObj.players:

            nextPosition = self.gameObj.players[player].directionToNextLocation(
                self.gameObj.players[player].posX,
                self.gameObj.players[player].posY,
                self.gameObj.players[player].direction)

            if nextPosition in nextPositions:
                self.gameObj.board.drawTieSquare(
                    nextPosition[0], nextPosition[1])
                return True
            else:
                nextPositions.append(nextPosition)
        return False

    def checkStatus(self, tieStatus):
        aliveCount = 0

        for player in self.gameObj.players:
            if self.gameObj.players[player].alive:
                aliveCount += 1

        # if both are dead (i.e. tie)
        if tieStatus or aliveCount == 0:
            # render them hitting each other
            # if tieStatus:
            #     for player in self.gameObj.players:
            #         self.gameObj.players[player].movePlayer()
            #     self.gameObj.screen.fill((0,0,0))
            #     self.gameObj.board.drawGrid()
            #     self.gameObj.board.drawTieSquare(self.gameObj.players[1].posX,self.gameObj.players[1].posY)

            self.active = False
            if self.gameMode == 1:
                self.gameObj.PVE_Tie += 1
            elif self.gameMode == 0:
                self.gameObj.PVP_Tie += 1
            elif self.gameMode == 2:
                self.gameObj.EVE_Tie += 1

            self.gameObj.gameOverMenu = GameOverMenu(
                self.gameObj, "Nobody", self.gameMode)
            self.gameObj.switchToMenu("GAME_OVER")
        # if one is dead
        elif aliveCount == 1:
            print("checkStatus")
            self.active = False

            winner = ""
            for player in self.gameObj.players:
                if self.gameObj.players[player].alive:
                    winner = player
                    break

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
                    self.gameObj.EVE_Bot1Wins += 1
                elif winner == 2:
                    self.gameObj.EVE_Bot2Wins += 1

            self.gameObj.gameOverMenu = GameOverMenu(
                self.gameObj, "Player " + str(winner), self.gameMode)
            self.gameObj.switchToMenu("GAME_OVER")

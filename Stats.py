import pygame

class StatsScreen:
    def __init__(self, game):
        self.game = game
        self.font_title = pygame.font.SysFont('Arial', 30, True)
        self.font_stats = pygame.font.SysFont('Arial', 20)
        self.font_instructions = pygame.font.SysFont('Arial', 20, True)

    def draw(self):
        # Clear the screen
        self.game.screen.fill((0, 0, 0))  # Black background

        # Title
        title_surface = self.font_title.render("Game Statistics", True, (255, 255, 255))  # White text
        self.game.screen.blit(title_surface, (self.game.scr_x // 2 - title_surface.get_width() // 2, 20))

        AStar_avgtime = 0
        AB_avgtime = 0
        Genetic_avgtime = 0
        NN_avgtime = 0

        for i in self.game.AStarTimes:
            AStar_avgtime += i
        for i in self.game.ABTimes:
            AB_avgtime += i
        for i in self.game.GeneticTimes:
            Genetic_avgtime += i
        for i in self.game.NNTimes:
            NN_avgtime += i

        if len(self.game.AStarTimes) > 0:
            AStar_avgtime = round(AStar_avgtime / len(self.game.AStarTimes), 2)
        if len(self.game.ABTimes) > 0:
            AB_avgtime = round(AB_avgtime / len(self.game.ABTimes), 2)
        if len(self.game.GeneticTimes) > 0:
            Genetic_avgtime = round(Genetic_avgtime / len(self.game.GeneticTimes), 2)
        if len(self.game.NNTimes) > 0:
            NN_avgtime = round(NN_avgtime / len(self.game.NNTimes), 2)

        # Stats data
        stats = [
            f"PVE Player Wins: {self.game.PVE_PlayerWins}",
            f"PVE Bot Wins: {self.game.PVE_BotWins}",
            f"PVE Ties: {self.game.PVE_Tie}",
            f"PVP Player 1 Wins: {self.game.PVP_Player1Wins}",
            f"PVP Player 2 Wins: {self.game.PVP_Player2Wins}",
            f"PVP Ties: {self.game.PVP_Tie}",
            f"A* vs Genetic - A* Wins: {self.game.AStarVGenetic_AStarWins}",
            f"A* vs Genetic - Genetic Wins: {self.game.AStarVGenetic_GeneticWins}",
            f"A* vs Genetic - Ties: {self.game.AStarVGenetic_Tie}",
            f"AB vs Genetic - AB Wins: {self.game.ABVGenetic_ABWins}",
            f"AB vs Genetic - Genetic Wins: {self.game.ABVGenetic_GeneticWins}",
            f"AB vs Genetic - Ties: {self.game.ABVGenetic_Tie}",
            f"NN vs Genetic - NN Wins: {self.game.NNVGenetic_NNWins}",
            f"NN vs Genetic - Genetic Wins: {self.game.NNVGenetic_GeneticWins}",
            f"NN vs Genetic - Ties: {self.game.NNVGenetic_Tie}",
            f"A* vs AB - A* Wins: {self.game.AStarVAB_AStarWins}",
            f"A* vs AB - AB Wins: {self.game.AStarVAB_ABWins}",
            f"A* vs AB - Ties: {self.game.AStarVAB_Tie}",
            f"A* vs NN - A* Wins: {self.game.AStarVNN_AStarWins}",
            f"A* vs NN - NN Wins: {self.game.AStarVNN_NNWins}",
            f"A* vs NN - Ties: {self.game.AStarVNN_Tie}",
            f"AB vs NN - AB Wins: {self.game.ABVNN_ABWins}",
            f"AB vs NN - NN Wins: {self.game.ABVNN_NNWins}",
            f"AB vs NN - Ties: {self.game.ABVNN_Tie}",
            f"A* Avg Time Survived: {AStar_avgtime} ticks",
            f"AB Avg Time Survived: {AB_avgtime} ticks",
            f"Genetic Avg Time Survived: {Genetic_avgtime} ticks",
            f"NN Avg Time Survived: {NN_avgtime} ticks"
        ]

        # Split stats into two columns
        mid_index = len(stats) // 2
        column_1 = stats[:mid_index]
        column_2 = stats[mid_index:]

        # Starting positions
        start_y = 100
        line_spacing = 30
        column_1_x = 50
        column_2_x = self.game.scr_x // 2 + 25

        # Render column 1
        for i, stat in enumerate(column_1):
            y_position = start_y + i * line_spacing
            stat_surface = self.font_stats.render(stat, True, (200, 200, 200))  # Light gray text
            self.game.screen.blit(stat_surface, (column_1_x, y_position))

        # Render column 2
        for i, stat in enumerate(column_2):
            y_position = start_y + i * line_spacing
            stat_surface = self.font_stats.render(stat, True, (200, 200, 200))  # Light gray text
            self.game.screen.blit(stat_surface, (column_2_x, y_position))

        # Instructions to return to the main menu
        instructions_surface = self.font_instructions.render(
            "Press [ESC] to return to Main Menu", True, (255, 255, 255)
        )
        self.game.screen.blit(
            instructions_surface,
            (self.game.scr_x // 2 - instructions_surface.get_width() // 2, self.game.scr_y - 50)
        )

        pygame.display.flip()

    def event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.game.switchToMenu("MAIN_MENU")  # Return to the main menu

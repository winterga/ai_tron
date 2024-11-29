import pygame

class StatsScreen:
    def __init__(self, game):
        self.game = game
        self.font_title = pygame.font.SysFont('Arial', 40, True)
        self.font_stats = pygame.font.SysFont('Arial', 25)
        self.font_instructions = pygame.font.SysFont('Arial', 20, True)

    def draw(self):
        # Clear the screen
        self.game.screen.fill((0, 0, 0))  # Black background

        # Title
        title_surface = self.font_title.render("Game Statistics", True, (255, 255, 255))  # White text
        self.game.screen.blit(title_surface, (self.game.scr_x // 2 - title_surface.get_width() // 2, 20))

        # Stats data
        stats = [
            f"PVE Player Wins: {self.game.PVE_PlayerWins}",
            f"PVE Bot Wins: {self.game.PVE_BotWins}",
            f"PVE Ties: {self.game.PVE_Tie}",
            f"PVP Player 1 Wins: {self.game.PVP_Player1Wins}",
            f"PVP Player 2 Wins: {self.game.PVP_Player2Wins}",
            f"PVP Ties: {self.game.PVP_Tie}",
            f"EVE Bot 1 Wins: {self.game.EVE_Bot1Wins}",
            f"EVE Bot 2 Wins: {self.game.EVE_Bot2Wins}",
            f"EVE Ties: {self.game.EVE_Tie}",
            f"PVG Player Wins: {self.game.PVG_playerWins}",
            f"PVG Bot Wins: {self.game.PVG_BotWins}",
            f"PVG Ties: {self.game.PVG_Tie}",
            f"EVG Non-Genetic Wins: {self.game.EVG_nonGenWins}",
            f"EVG Genetic Wins: {self.game.EVG_GeneticWins}",
            f"EVG Ties: {self.game.EVG_Tie}",
        ]

        start_y = 100
        line_spacing = 30
        for i, stat in enumerate(stats):
            stat_surface = self.font_stats.render(stat, True, (200, 200, 200))  # Light gray text
            self.game.screen.blit(stat_surface, (50, start_y + i * line_spacing))

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
        """
        Handle user input for the stats screen.
        """
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.game.switchToMenu("MAIN_MENU")  # Return to the main menu

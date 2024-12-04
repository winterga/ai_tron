# Authors: Greyson Wintergerst, ... (add your name here if you worked on this file) FIXME
# Description: This file contains the MainMenu class, which contains all menu logic for the main menu screen.

import pygame

class MainMenu:
	def __init__(self, gameObj):
		self.game = gameObj

		self.font = pygame.font.SysFont('swmono', 34)

		self.menuList = ['Player vs Player', 'Player vs AI', 'Train AI w Genetic Algorithm', 'Train AI w DeepQ', 'Player vs DeepQ', 'Tournament', 'Stats']
		print(enumerate(self.menuList))
		self.items = [] #text objects, a list of lists: [text, bitmap, (width, height), (posx, posy)]
		self.activeItem = 0
		self.activeColor = (70, 235, 30)
		self.inactiveColor = (255, 255, 255)
		
		#calculate the positions of the text elements (populates self.items)
		for i, item in enumerate(self.menuList):
			if i == self.activeItem:
				color = self.activeColor
			else:
				color = self.inactiveColor
			text_surface = self.font.render(item, 1, color)

			width = text_surface.get_rect().width
			height = text_surface.get_rect().height + 40

			posx = (self.game.scr_x / 2) - (width / 2)

			total_height = len(self.items) * height
			posy = (self.game.scr_y / 2) - (total_height / 2) + (i * height)

			self.items.append([item, text_surface, (width, height), (posx, posy)])

	# Handles user input while in the MAIN_MENU state
	def eventTick(self, event):
		if event.type == pygame.KEYDOWN: #change the active menu entry
			prevActive = self.activeItem 
			
			# Change the current active item based on the key pressed
			if event.key == pygame.K_UP:
				self.activeItem = ((self.activeItem-1) % len(self.items))
			elif event.key == pygame.K_DOWN:
				self.activeItem = ((self.activeItem+1) % len(self.items))
			
			# If the active item has changed, update the colors of the text objects
			if prevActive != self.activeItem: 
				self.items[self.activeItem][1] = self.font.render(self.items[self.activeItem][0], 1, self.activeColor)
				self.items[prevActive][1] = self.font.render(self.items[prevActive][0], 1, self.inactiveColor)
				self.game.screen.fill((0,0,0))
				self.draw()
			
			if event.key == pygame.K_RETURN and self.activeItem == 7:
				self.game.switchToMenu('STATS_SCREEN')
			#select the entry, start the match
			elif event.key == pygame.K_RETURN:
				self.game.startMatch(self.activeItem) #match type is equal to the element's ID

	# Draws the pygame screen to display the main menu
	def draw(self):
     
		
		# Build title text
		titleFont = pygame.font.SysFont('magneto', 80)
		title = titleFont.render("IntelliTron", 1, (255, 255, 255))
		title_width = title.get_rect().width
		title_height = title.get_rect().height
  
		# Draw tron bike image
		bike_img = pygame.image.load('tron_bike.png')
		bike_img = pygame.transform.scale(bike_img, (self.game.scr_x/1.25, 5*title_height))
		bike_width = bike_img.get_rect().width
		bike_height = bike_img.get_rect().height
		self.game.screen.blit(bike_img, (0, title_height-20))
  
		# Draw title
		self.game.screen.blit(title, ((self.game.scr_x / 2) - (title_width / 2), 60))
  

		
  
		# Draw selectable menu options
		for name, label, (width, height), (posx, posy) in self.items:
                	self.game.screen.blit(label, (posx, posy))
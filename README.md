# ai_tron (IntelliTRON)
Development of AI tron bot algorithms, including a demo visualization. 

# Requirements
Python: `python3.11.8` was used to demo and develop this project. Use >= `python3.11` for the best, tested experience.

### For pip users: 
1. Clone this repository:
   ```bash
   git clone https://github.com/winterga/ai_tron.git
   cd ai_tron
2. Run `pip install -r requirements.txt` to install necessary project dependencies.

### For Anaconda users:
#### Setting up the Environment
1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).
2. Clone this repository:
   ```bash
   git clone https://github.com/winterga/ai_tron.git
   cd ai_tron
3. Run `conda env create -f environment.yml`
4. Activate the environment: `conda activate intellitron`


# How to Play
After cloning the repository and setting up the environment, navigate to the main directory of the project, `ai_tron`. 
Simply run `python Tron.py` to start the game loop. Options for playing can be chosen through the Main Menu.
Navigate with the arrow keys and press `[Enter]` on the highlighted option to access that menu choice.

Player vs. Player option: Player 1 uses WASD to control their bike. Player 2 uses arrow keys.
All other active match options: Human player starts in top left and uses WASD to control their bike. Bot/AI starts in bottom right.

Tournament option: The tournament will simulate games between all pairs of algorithms. The number of games it will simulate for each pairing can be changed in the Tron class by modifying `num_tourney_rounds` (line 93 in Tron.py). Currently, it is set to run for 30 games per pairing.

# Notes on Training
No datasets are required for training any of the models implemented in this repository.

The current trained model parameters have been saved to `deepq_model.pth`. Clicking on the "Train AI w/ DeepQ" option in the main menu will train the model
for 50 epochs/episodes (can be changed in the `startDeepTraining` function in `Tron.py`; see `episodes=50`).
The current trained model parameters for the genetic algorithm have been saved in the Tron class as `best_genome`. Optionally, clicking on the "Train AI w Genetic Algorithm" option in the main menu will train the model for 50 generations (20 games per generation) and overwrite the current best_genome with the newly trained results (parameters can be changed in the `start_training` function in `Tron.py` but may lead to inconsistent results).

# Credit
Authors: Greyson Wintergerst, Eileen Hsu, Debanjan Chakraborti, and Astha Sinha

Last Updated: December 6th, 2024

This project took inspiration from the following repository: https://github.com/nlieb/PyTron.
We used a similar file organization structure and iterated upon their pygame logic for creating the game and
interactable UI.

Overall, we used their repository to learn how to interact with pygame and utilized one of their algorithms for partially training
the Genetic and DeepQ models.

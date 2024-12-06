# Author: Eileen Hsu
# Description: This file contains functions utilized in training an AI with the genetic algorithm.

import random

# Calculates the fitness score of an AI after playing a simulated match
def fitness(game_results):
    survival_time = game_results['survival_time']
    wins = game_results['wins']
    board_state = game_results['board_state']
    bot = game_results['bot']

    reachable_area = bot.calculateReachableArea(bot.x, bot.y, board_state)

    survival_weight = 2
    win_weight = 100
    reachable_area_weight = 0.3

    fitness_score = (
        survival_weight * survival_time +
        win_weight * wins +
        reachable_area_weight * reachable_area
    )

    return fitness_score

# Selects the top half (10) of the population sorted by fitness score to be the parents of the next generation
def select_parents(population, fitness_scores):
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population), reverse=True)]
    return sorted_population[:len(population) // 2]

# Crosses over parents to create a child 
# i.e. if parent1 has genome [a, b] and parent2 has genome [c, d], their children will have genomes [a, d] and [c, b]
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# Mutates the genome of a child at an inputted rate
def mutate(genome, rate):
    return [
        gene + random.uniform(-rate, rate) if random.random() < rate else gene
        for gene in genome
    ]

# Main function for training an AI with the genetic algorithm. Trains an AI by simulating matches for a population,
# selecting the highest performing ones based on a fitness function, and creating children with those selected genomes
# to form the next generation. The initial population, number of generations to run for, mutation rate, and function
# used to simulate a game are passed into the function.
def train_genetic(population, generations, mutation_rate, simulate_game):
    for generation in range(generations):
        fitness_scores = []

        for i, genome1 in enumerate(population):
            genome2 = random.choice(population)
            game_results = simulate_game(genome1, genome2)
            fitness_scores.append(fitness(game_results))

        max_fitness = max(fitness_scores)
        print(f"Generation {generation + 1}/{generations}, Best Fitness: {max_fitness}")

        parents = select_parents(population, fitness_scores)

        next_population = []
        while len(next_population) < len(population):
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = crossover(parent1, parent2)
            next_population.append(mutate(child1, mutation_rate))
            if len(next_population) < len(population):
                next_population.append(mutate(child2, mutation_rate))

        population = next_population

    print("Training complete!")
    return population
import random
from concurrent.futures import ThreadPoolExecutor

# def fitness(bot, game_results):
#     survival_time = game_results['survival_time']
#     wins = game_results['wins']
#     territory = game_results['territory']

#     return 0.1 * survival_time + wins * 5 + territory

def fitness(bot, game_results, max_steps=400):
    survival_time = game_results['survival_time']
    wins = game_results['wins']
    territory = game_results['territory']

    # Offensive score: reward wins and fast elimination
    offensive_score = wins * (5 + 1 / (0.1 + survival_time)) + territory * 0.5

    # Defensive score: reward survival time and safe play
    defensive_score = 0.2 * survival_time

    # Stalemate penalty: discourage overly defensive play
    stalemate_penalty = 0
    if survival_time >= max_steps:
        stalemate_penalty = -5

    print(survival_time, ":", max_steps)
    # Total fitness: balance offense and defense
    return offensive_score + defensive_score + stalemate_penalty


def select_parents(population, fitness_scores):
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population), reverse=True)]
    return sorted_population[:len(population) // 2]


def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def mutate(genome, rate):
    return [
        gene + random.uniform(-rate, rate) if random.random() < rate else gene
        for gene in genome
    ]

# ignore tolerance for now
def train_genetic(population, generations, mutation_rate, simulate_game, tolerance=30):
    best_fitness = -float('inf')
    stagnation_count = 0

    for generation in range(generations):
        fitness_scores = []

        # Calculate fitness for each genome
        for i, genome1 in enumerate(population):
            # Select a random opponent genome for the second bot
            genome2 = random.choice(population)
            # Simulate the game and calculate fitness
            game_results = simulate_game(genome1, genome2)
            fitness_scores.append(fitness(None, game_results))

        # Determine the best fitness in this generation
        max_fitness = max(fitness_scores)
        print(f"Generation {generation + 1}/{generations}, Best Fitness: {max_fitness}")

        # Check for fitness improvement
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            stagnation_count = 0
        else:
            stagnation_count += 1

        # Stop training if fitness has plateaued
        if stagnation_count >= tolerance:
            print(f"No improvement for {tolerance} generations. Stopping early at generation {generation + 1}.")
            break

        # Select the best genomes as parents
        parents = select_parents(population, fitness_scores)

        # Create the next generation
        next_population = []
        while len(next_population) < len(population):
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = crossover(parent1, parent2)
            next_population.append(mutate(child1, mutation_rate))
            if len(next_population) < len(population):
                next_population.append(mutate(child2, mutation_rate))

        # Update the population
        population = next_population

    print("Training complete!")
    return population
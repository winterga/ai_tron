import random

def fitness(game_results):
    survival_time = game_results['survival_time']
    wins = game_results['wins']
    board_state = game_results['board_state']
    bot = game_results['bot']

    distance_to_self = bot.distanceToSelf(bot.direction)

    reachable_area = bot.calculateReachableArea(bot.x, bot.y, board_state)

    survival_weight = 2
    win_weight = 5
    distance_to_self_weight = 0.2
    reachable_area_weight = 0.3

    fitness_score = (
        survival_weight * survival_time +
        win_weight * wins +
        distance_to_self_weight * distance_to_self +
        reachable_area_weight * reachable_area
    )

    return fitness_score

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
import random
import numpy as np

# Genetic Algorithm settings
population_size = 100
mutation_rate = 0.01
num_generations = 100

# Objective function
def objective_function(a, b, c, d):
    return abs(a + 2*b + 3*c + 4*d - 30)

# Generate initial population
def generate_initial_population():
    population = []
    for _ in range(population_size):
        individual = [random.randint(1, 29) for _ in range(4)]
        population.append(individual)
    return population

# Crossover
def crossover(parent1, parent2):
    child = []
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    return child

# Mutation
def mutate(individual):
    mutated_individual = []
    for gene in individual:
        if random.random() < mutation_rate:
            mutated_individual.append(random.randint(1, 29))
        else:
            mutated_individual.append(gene)
    return mutated_individual

# Genetic Algorithm
def genetic_algorithm():
    population = generate_initial_population()

    for generation in range(num_generations):
        # Calculate fitness scores for each individual in the population
        fitness_scores = []
        for individual in population:
            fitness_scores.append(objective_function(*individual))

        # Add a small constant value to avoid zero weights
        fitness_scores = np.array(fitness_scores) + 0.01

        # Check for NaN values in fitness scores
        if np.isnan(fitness_scores).any():
            # Handle NaN values by setting them to a finite value
            fitness_scores = np.nan_to_num(fitness_scores, nan=1.0)

        # Select parents for the next generation
        selected_parents = random.choices(population, weights=1 / fitness_scores, k=population_size)

        # Generate new generation
        new_population = []
        for _ in range(population_size):
            parent1, parent2 = random.choice(selected_parents), random.choice(selected_parents)
            child = crossover(parent1, parent2)
            mutated_child = mutate(child)
            new_population.append(mutated_child)

        population = new_population

    # Find the best individual in the final generation
    best_individual = population[np.argmin([objective_function(*individual) for individual in population])]
    return best_individual

# Run the genetic algorithm and print the result
best_solution = genetic_algorithm()
a, b, c, d = best_solution
print(f"The best solution found: a={a}, b={b}, c={c}, d={d}")
print(f"Objective function value: {objective_function(a, b, c, d)}")

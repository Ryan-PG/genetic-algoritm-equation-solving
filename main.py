# Solve the `a+2b+3c+4d-30=0` Equation using Genetic Algorithm in python

import random
from deap import base, creator, tools, algorithms

# Define the optimization problem
creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Define the bounds for variables a, b, c, d
bounds = [(-10, 10), (-10, 10), (-10, 10), (-10, 10)]

# Define the genetic algorithm toolbox
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -10, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=4)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Define the evaluation function
def evaluate(individual):
    a, b, c, d = individual
    equation_result = a + 2 * b + 3 * c + 4 * d - 30
    return abs(equation_result),

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

if __name__ == "__main__":
    # Create the initial population
    population = toolbox.population(n=50)

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Run the genetic algorithm
    # mu: Number of individuals to select for the next generation.
    # lambda_: Number of offspring to produce at each generation.
    # cxpb: Probability of mating two individuals.
    # mutpb: Probability of mutating an individual.
    # ngen: Number of generations.
    # stats, halloffame, and verbose parameters are optional.
    algorithms.eaMuPlusLambda(population, toolbox, mu=50, lambda_=200, cxpb=0.7, mutpb=0.2, ngen=100, stats=None, halloffame=None, verbose=True)

    # Print the best individual
    best_individual = tools.selBest(population, k=1)[0]
    print("Best individual:", best_individual)
    print("Best solution:", [round(x, 2) for x in best_individual])


# Genetic Algorithms
# 3 main functions:
# 1 populate, generate population
# 2 reduction, reduce the size of the population
# 3 cross, cross-breeding

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [8,6]

x1_range = [-100, 100]
x2_range = [-100, 100]

population = []

def populate(features, size = 1000):
    initial = []
    
    for _ in range(size):
        entity = []
        for feature in features:
            val = np.random.randint(*feature)
            entity.append(val)
        initial.append(entity)
    return np.array(initial)

virus = np.array([5,5])

def fitness(population, virus, size = 100):
    
    scores = []
    
    for index, entity in enumerate(population):
        score = np.sum((entity - virus) ** 2)
        scores.append((score, index))
    scores = sorted(scores)[:size]
    
    return np.array(scores)[:, 1]

def draw(population, virus):
    plt.xlim((-100, 100))
    plt.ylim((-100, 100))
    plt.scatter(population[:, 0], population[:, 1], c = 'green', s = 12)
    plt.scatter(virus[0], virus[1], c = 'red', s = 60)
    
def reduction(population, virus, size = 100):
    fittest = fitness(population, virus, size)
    
    new_pop = []
    
    for item in fittest:
        new_pop.append(population[item])
    
    return np.array(new_pop)

def cross(population, size = 1000):
    
    new_pop = []
    for _ in range(size):
        p = population[np.random.randint(0, len(population))]
        m = population[np.random.randint(0, len(population))]
        
        entity = []
        entity.append(*p[:len(p)//2])
        entity.append(*m[len(m)//2:])
        
        new_pop.append(entity)
    return np.array(new_pop)

def mutate(population):
    
    return population + np.random.randint(-10, 10, 2000).reshape(1000, 2)

population = populate([x1_range, x2_range], 1000)

def cycle(population, virus, gens = 1):
    
    for _ in range(gens):
        population = reduction(population, virus, 100)
        population = cross(population, 1000)
        population = mutate(population)
        
    return population

population = cycle(population, virus)

draw(population, virus)
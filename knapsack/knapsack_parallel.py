import random
import multiprocessing as mp


class Item:
    def __init__(self, name, weight, value):
        self.name = name
        self.weight = weight
        self.value = value
    
    def __str__(self):
        return f"name: {self.name} weight: {self.weight} value: {self.value}"
    
    def __repr__(self):
        return str(self)


class Individual:
    def __init__(self, counts):
        self.counts = counts
    
    def __str__(self):
        return repr(self.counts)

    def __hash__(self):
        return hash(str(self.counts))
    
    def __eq__(self, other):
        return self.counts == other.counts
    
    def fitness(self) -> float:
        total_value = sum([
            count * item.value
            for item, count in zip(items, self.counts)
        ])

        total_weight = sum([
            count * item.weight
            for item, count in zip(items, self.counts)
        ])

        if total_weight <= MAX_KNAPSACK_WEIGHT:
            return total_value
        
        return 0



MAX_KNAPSACK_WEIGHT = 503
MUTATION_RATE = 0.01
REPRODUCTION_RATE = 0.1
POPULATION_COUNT = 30
DIFF_PRECISION = 0.05
STOP_EPOCH_NUM = 5
NUM_OF_POPUlATIONS = 1 # 50 
MIGRATION_PERIOD = 5
BEST_IN_MIGRATION = 4
IMMIGRANTS_NUM = 3

EPOCHS = 5 # 500 

# best result 1256
items = [
    Item("A", 1, 1), # 0
    Item("B", 3, 7), # 1
    Item("C", 4, 9), # 1
    Item("D", 6, 5), # 0
    Item("E", 8, 20), #  62
    Item("F", 10, 13), # 0
    Item("G", 21, 27), # 0
    Item("H", 7, 9), # 0
    Item("I", 5, 8), # 0
    Item("K", 9, 13), # 0
]

def generate_initial_population(population_count=POPULATION_COUNT):
    population = set()
    while len(population) != population_count:
        idxs = list(range(len(items)))
        random.shuffle(idxs)
        counts = [0 for i in range(len(items))]
        W = MAX_KNAPSACK_WEIGHT
        for idx in idxs[:-1]:
            i = items[idx]
            count = random.randint(0, W // i.weight)
            counts[idx] = count
            W -= count * i.weight
        counts[idxs[-1]] = W // items[idxs[-1]].weight
        population.add(Individual(counts))
    return list(population)


def selection(population):
    parents = []
    
    random.shuffle(population)
    
    if population[0].fitness() > population[1].fitness():
        parents.append(population[0])
    else:
        parents.append(population[1])
    
    if population[2].fitness() > population[3].fitness():
        parents.append(population[2])
    else:
        parents.append(population[3])

    return parents


def crossover(parents):
    N = len(items)

    crosscover_point = random.randint(1, N-2)
    print(f"crossover point: {crosscover_point}")
    child1 = Individual(parents[0].counts[:crosscover_point] + parents[1].counts[crosscover_point:])
    child2 = Individual(parents[1].counts[:crosscover_point] + parents[0].counts[crosscover_point:])

    return [child1, child2]



def mutate(individuals):
    for individual in individuals:
        if not individual.fitness():
            continue
        if random.random() < MUTATION_RATE:
            for _ in range(random.randint(len(individual.counts) // 2, len(individual.counts))):
                w = MAX_KNAPSACK_WEIGHT
                gen = random.randint(0, len(individual.counts)-1)
                print(f"mutate {str(individual)}: change gen {gen}")
                for i in range(len(individual.counts)):
                    if i == gen:
                        continue
                    w -= individual.counts[i] * items[i].weight
                individual.counts[gen] = random.randint(0, w // items[gen].weight)
    return individuals


def next_generation(population):
    next_gen = []
    while len(next_gen) < len(population):
        children = []

        #selection
        parents = selection(population)
        print("parents selection:")
        for p in parents:
            print(p)

        # reproduction
        if random.random() < REPRODUCTION_RATE:
            children = parents
        else:
            # crossover
            children = crossover(parents)
            print("children:")
            for c in children:
                print(c)
            
            # mutation
            if random.random() < MUTATION_RATE:
                children = mutate(children)
                print("mutation:")
                for c in children:
                    print(children)

        if children[0].fitness():
            next_gen.append(children[0])
            print(f"add fist children")
        if children[1].fitness():
            print("add second children")
            next_gen.append(children[1])
    
    next_gen.extend(population)

    n_sorted = sorted(next_gen, key=lambda x: x.fitness(), reverse=True)

    new_population = []
    population_len = len(n_sorted)
    rank = 0
    while len(new_population) != POPULATION_COUNT:
        if random.random() < (population_len - rank) / population_len:
            new_population.append(n_sorted[rank])
        rank = (rank + 1) % population_len
    return sorted(new_population, key=lambda i: i.fitness(), reverse=True)
        


def print_generation(population):
    for individual in population:
        total_weight = sum([
            count * item.weight
            for item, count in zip(items, individual.counts)
        ])
        print(individual.counts, 'fitness', individual.fitness(), 'total_weight', total_weight)
    print()
    print("Average fitness", average_fitness(population))
    print("-" * 32)


def average_fitness(population):
    return sum([i.fitness() for i in population]) / len(population)


def solve_knapsack(population):
    avg_fitness = []
    for _ in range(MIGRATION_PERIOD):
        population = next_generation(population)
        avg_fitness.append(average_fitness(population))
        # среднее значение функции приспособленности за последние 5 эпох не отличается от последнего значения
        if len(avg_fitness) > 5 and abs((sum(avg_fitness[-5:]) / 5) / avg_fitness[-1] - 1) < DIFF_PRECISION:
            break
    return sorted(population, key=lambda i: i.fitness(), reverse=True)


def migrations(populations):
    bestI = []
    for population in populations:
        bestI.extend(population[:BEST_IN_MIGRATION])
    
    for population in populations:
        print("migration: init population:")
        print_generation(population)
        for i in range(IMMIGRANTS_NUM):
            best = random.choice(bestI)
            if population[-i-i].fitness() < best.fitness():
                population[-i-1] = best
        print("migration: after migration:")
        print_generation(population)

def parallel_knapsack():
    populations = []
    for _ in range(NUM_OF_POPUlATIONS):
        populations.append(generate_initial_population())
        print_generation(populations[-1])
    
    
    pool = mp.Pool(processes=len(populations))
    
    prev_fitness = 0 
    for _ in range(EPOCHS // MIGRATION_PERIOD):
        # print(f"migration period {i}")
        populations = pool.map(solve_knapsack, populations)
        migrations(populations)

        best = populations[0][0]
        for population in populations[1:]:
            if population[0].fitness() > best.fitness():
                best = population[0]
        if best.fitness() == prev_fitness:
            break 
        prev_fitness = best.fitness()
    
    # for i in range(len(populations)):
    #     print(f"популяция {i}")
    #     print_generation(populations[i])

    return best

if __name__ == '__main__':
    solution = parallel_knapsack()
    print_generation([solution])
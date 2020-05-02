"""
code by Eric Stoltz
modification for use wisth celery: Juan P. Giraldo
"""

import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt, time, pickle, celery

app = celery.Celery('GATSP')
app.config_from_object('config')

class City:
    """class to instantiate city objects (Genes). implements a function to calculate distance between current city
    and a second one passed as parameter"""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


class Fitness:
    """Class instantiated to calculate the fitness and length of a distance"""

    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness


def createRoute(cityList):
    """Function to create one route"""
    route = random.sample(cityList, len(cityList))
    return route


def initialPopulation(popSize, cityList):
    """Function to create multiple routes"""

    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population


def rankRoutes(population):
    """rank a given set of routes by their lengths"""
    fitnessResults = {}
    results = []
    for i in range(0, len(population)):  # here start the customizable part
        print(population[i])
        result = celeryRankRoutes.apply_async(list(population[i]), serializer="pickle")
        results.append(result)
    fitnessResults = [result.get() for result in results]
    return sorted(fitnessResults.items(), key=operator.itemgetter(1),
                  reverse=False)  # set this equal to false to to calculate the longest path

@app.task
def celeryRankRoutes(population):
    return Fitness(population).routeFitness()


def selection(popRanked, eliteSize):
    """Function to select a given number of parents from one generation"""
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()



    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100 * random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults


def matingPool(population, selectionResults):
    """Function takes the result from selection to create the next generation of routes"""
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


def breed(parent1, parent2):
    """Function to form a new route based on two parent routes"""
    child = []
    childP1 = []
    childP2 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child


def breedPopulation(matingpool, eliteSize):
    """Function to form a new generation of routes"""
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children


def mutate(individual, mutationRate):
    """Introduction chance of ramdonly switching the route order"""
    for swapped in range(len(individual)):
        if (random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    return individual


def mutatePopulation(population, mutationRate):
    """Function to introduce a chance of altering the order of routes in the routes in the population"""
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop


def nextGeneration(currentGen, eliteSize, mutationRate):
    """Function to recursively create new generations"""
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration


def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    """implementation of genetic algorithm in serial"""

    # this commented code is to create a new random population
    pop = initialPopulation(popSize, population)
    pops = open("pop.pickle", "wb")
    pickle.dump(pop.copy(), pops)


    # code to import previously used population
    '''pops = open("pop.pickle", "rb")
    pop = pickle.load(pops)
    pops.close()

    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))
    '''
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)

    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute


def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])

    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])

    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()

@app.task
def echo23(string):
    return string


result = echo23.delay("hola")
print(result.get())


cityList = []

for i in range(0, 25):
    cityList.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))

start = time.monotonic()
geneticAlgorithm(population=cityList, popSize=99, eliteSize=20, mutationRate=0.01, generations=600)
# geneticAlgorithmPlot(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)
end = time.monotonic()
print("the total time the algorithm took to find the largest path was: " + str(end - start))

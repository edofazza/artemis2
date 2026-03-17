import os.path

from deap import base, creator, tools, algorithms
import random
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from .ensemble import Ensemble


class GeneticEnsemble(object):
    def __init__(self, ensemble_model: Ensemble, ngen, cxpb, mutpb, elite_size, pop_size=15, fold=0):
        self.ensemble = ensemble_model
        self.ngen = ngen
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.elite_size = elite_size
        self.pop_size = pop_size
        self.fold = fold

        if not os.path.exists('ga_artemis2'):
            os.mkdir('ga_artemis2')

    def train(self):
        # Problem definition (maximization problem)
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        # DEAP toolbox and registration
        toolbox = base.Toolbox()
        toolbox.register("individual", GeneticEnsemble.init_individual,
                         creator.Individual, size=len(self.ensemble.models))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", GeneticEnsemble.cx_blend_sum_to_one)
        toolbox.register("mutate", GeneticEnsemble.mut_gaussian_sum_to_one, mu=0, sigma=0.1, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", self.evaluate)
        # Create population
        population = toolbox.population(n=self.pop_size)
        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", np.max)
        stats.register("avg", np.mean)
        # Define Hall-of-Fame
        hof = tools.HallOfFame(self.elite_size)
        # perform GA
        population, logbook = GeneticEnsemble.ea_simple_with_elitism(
            population, toolbox, cxpb=self.cxpb, mutpb=self.mutpb,
            ngen=self.ngen, stats=stats, halloffame=hof, verbose=True
        )
        # print best solution
        best = hof.items[0]
        print("-- Best Ever Individual = ", best)
        print("-- Best Ever Fitness = ", best.fitness.values[0])
        # save hof
        with open(f'ga_artemis2/hof_{self.fold}.pkl', 'wb') as f:
            pickle.dump(hof, f)
        # extract statistics:
        maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

        # plot statistics:
        sns.set_style("whitegrid")
        plt.plot(maxFitnessValues, color='red', label='Max Fitness')
        plt.plot(meanFitnessValues, color='green', label='Mean Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Max / Average Fitness')
        plt.legend()
        plt.title('Max and Average fitness over Generations')
        plt.savefig(f'ga_artemis2/statistics_{self.fold}.pdf')

    def evaluate(self, weights):
        return self.ensemble.test(weights),

    @staticmethod
    def init_individual(icls, size):
        """
        Custom initialization to ensure weights sum to 1
        :param icls: class passed by creator.Individual
        :param size:
        :return:
        """
        individual = [random.random() for _ in range(size)]
        total = sum(individual)
        return icls([x / total for x in individual])

    @staticmethod
    def cx_blend_sum_to_one(ind1, ind2, alpha=0.5):
        """
        Crossover operator that maintains sum-to-one constraint
        :param ind2:
        :param alpha:
        :return:
        """
        for i in range(len(ind1)):
            gamma = (1. - 2. * alpha) * random.random() + alpha
            ind1[i], ind2[i] = gamma * ind1[i] + (1 - gamma) * ind2[i], gamma * ind2[i] + (1 - gamma) * ind1[i]

        # Normalize both individuals to sum to 1
        GeneticEnsemble.normalize(ind1)
        GeneticEnsemble.normalize(ind2)

        return ind1, ind2

    @staticmethod # 4.
    def mut_gaussian_sum_to_one(individual, mu, sigma, indpb):
        """
        Mutation operator with sum-to-one constraint
        :param individual:
        :param mu:
        :param sigma:
        :param indpb:
        :return:
        """
        for i in range(len(individual)):
            if random.random() < indpb:
                individual[i] += random.gauss(mu, sigma)
        GeneticEnsemble.normalize(individual)
        return individual,

    @staticmethod
    def normalize(individual):
        """
        Normalize the individual so that they sum to 1
        :param individual:
        :return:
        """
        total = sum(individual)
        for i in range(len(individual)):
            individual[i] /= total

    @staticmethod
    def ea_simple_with_elitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
                               halloffame=None, verbose=__debug__):
        """
        GA loop with elitism
        :param population:
        :param toolbox:
        :param cxpb:
        :param mutpb:
        :param ngen:
        :param stats:
        :param halloffame:
        :param verbose:
        :return:
        """
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is None:
            raise ValueError("halloffame parameter must not be empty!")

        halloffame.update(population)
        hof_size = len(halloffame.items) if halloffame.items else 0

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Begin the generational process
        for gen in range(1, ngen + 1):

            # Select the next generation individuals
            offspring = toolbox.select(population, len(population) - hof_size)

            # Vary the pool of individuals
            offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # add the best back to population:
            offspring.extend(halloffame.items)

            # Update the hall of fame with the generated individuals
            halloffame.update(offspring)

            # Replace the current population by the offspring
            population[:] = offspring

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)

        return population, logbook

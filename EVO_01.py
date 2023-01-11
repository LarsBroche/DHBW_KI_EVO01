import csv
import random
import argparse
import json

POPULATION_SIZE = 100
MAX_FITNESS = 0.95
MAX_ITERATIONS = 100
FITTEST_INDIVIDUALS_TO_SELECT = 50


def read_csv():

    with open('iris.data_subset_1.csv', 'r') as f:
        reader = csv.reader(f)
        rows = [row for row in reader]

    keys = ['sepal_length', 'sepal_width',
            'petal_length', 'petal_width', 'iris_species']
    iris_data = [{keys[i]: row[i] for i in range(len(keys))} for row in rows]

    return iris_data


def classify_iris(iris, classifier):

    for feature_index, operator, threshold, class_label in classifier:
        feature_value = [iris["sepal_length"], iris["sepal_width"],
                         iris["petal_length"], iris["petal_width"]][feature_index]
        if eval(str(feature_value) + operator + str(threshold)):
            return class_label
    return "Iris-versicolor"


def generate_random_classifier():
    classifierLength = random.randint(1, 5)

    classifier = []

    for _ in range(classifierLength):
        feature_index = random.randint(0, 3)
        operator = random.choice(["<", ">=", "=="])
        threshold = round(random.uniform(0, 10), 1)
        class_label = random.choice(
            ["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
        classifier.append((feature_index, operator, threshold, class_label))

    return classifier


def init_starting_population():
    print("Initializing Population...")

    population = []

    for _ in range(POPULATION_SIZE):
        population.append(generate_random_classifier())
    return population


def selection(population, iris_list, fittestIndividualsToSelect):

    fitness_scores = [(fitness(classifier, iris_list), classifier)
                      for classifier in population]

    fitness_scores.sort(reverse=True)

    return [classifier for _, classifier in fitness_scores[:fittestIndividualsToSelect]]


def fitness(classifier, iris_list):
    correct = 0
    total = len(iris_list)

    for iris in iris_list:

        predicted_species = classify_iris(iris, classifier)
        if predicted_species == iris["iris_species"]:
            correct += 1

    return correct / total


def max_fitness_score(population, iris_list):

    fitness_scores = [(fitness(classifier, iris_list), classifier)
                      for classifier in population]

    fitness_scores.sort(reverse=True)

    return fitness_scores[0][0]


def crossover(classifiers):

    new_classifiers = []

    for _ in range(POPULATION_SIZE):

        parent1 = random.choice(classifiers)
        parent2 = random.choice(classifiers)

        child = crossover_point(parent1, parent2)

        new_classifiers.append(child)

    new_classifiers += classifiers

    return new_classifiers


def crossover_point(classifier1, classifier2):
    crossover_point = random.randint(1, len(classifier1))

    new_classifier = classifier1[:crossover_point] + \
        classifier2[crossover_point:]

    return new_classifier


def genetic_algorithm(population, iris_data):

    parent_population = population
    iterations = 0
    iterations_since_last_improvement = 0

    while max_fitness_score(parent_population, iris_data) <= MAX_FITNESS and iterations != MAX_ITERATIONS and iterations_since_last_improvement != 7:

        fit_individuals = selection(
            parent_population, iris_data, FITTEST_INDIVIDUALS_TO_SELECT)
        child_population = crossover(fit_individuals)

        child_population = mutation(child_population)
        if max_fitness_score(child_population, iris_data) <= max_fitness_score(parent_population, iris_data):
            iterations_since_last_improvement += 1
        else:
            iterations_since_last_improvement = 0
        parent_population = child_population
        print("Iterations:", iterations)
        print(max_fitness_score(parent_population, iris_data))
        iterations += 1

    return parent_population


def mutation(classifiers):

    new_classifiers = []

    for classifier in classifiers:
        featureToMutate = random.randint(0, len(classifier)-1)

        mutation = random.uniform(0, 1) * random.uniform(0, 1)

        classifierAsList = list(classifier[featureToMutate])

        if (random.randint(0, 1) == 0):
            classifierAsList[2] += mutation

        else:
            classifierAsList[2] -= mutation

        classifierAsList[2] = round(classifierAsList[2], 1)
        classifier[featureToMutate] = tuple(classifierAsList)
        new_classifiers.append(classifier)

    return new_classifiers


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("sepal_length", type=float, help="Sepal Length")
    parser.add_argument("sepal_width", type=float, help="Sepal Width")
    parser.add_argument("petal_length", type=float, help="Petal Length")
    parser.add_argument("petal_width", type=float, help="Petal Width")
    parser.add_argument('-train', action='store_true')

    args = parser.parse_args()

    cliSepalLength = args.sepal_length
    cliSepalWidth = args.sepal_width
    cliPetalLength = args.petal_length
    cliPetalWidth = args.petal_width
    trainBool = args.train

    iris = {
        "sepal_length": cliSepalLength,
        "sepal_width": cliSepalWidth,
        "petal_length": cliPetalLength,
        "petal_width": cliPetalWidth,
    }
    iris_data = read_csv()
    if (trainBool):
        classifier_population = init_starting_population()

        selected_generation = genetic_algorithm(
            classifier_population, iris_data)

        with open("selected_generation.txt", "w") as fp:
            json.dump(selected_generation, fp)

        
    else:
        with open("selected_generation.txt", "r") as fp:
            selected_generation = json.load(fp)

    print(classify_iris(iris, selection(selected_generation,
          iris_data, FITTEST_INDIVIDUALS_TO_SELECT)[0]))

#Evolutinary algoithm in context of the iris dataset
#creates a classifier that uses the petal length 
import csv
import random
import argparse
import json 

POPULATION_SIZE = 225
MAX_FITNESS = 0.99
MAX_ITERATIONS = 300
FITTEST_INDIVIDUALS_TO_SELECT = 16

def read_csv():

    with open('iris.data_subset_1.csv', 'r') as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
        
    keys = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'iris_species']
    iris_data = [{keys[i]: row[i] for i in range(len(keys))} for row in rows]

    return iris_data

def classify_iris(iris, classifier):
  
  for feature_index, operator, threshold, class_label in classifier:
    feature_value = [iris["sepal_length"], iris["sepal_width"], iris["petal_length"], iris["petal_width"]][feature_index]
    if eval(str(feature_value) + operator + str(threshold)):
      return class_label
  return "Iris-versicolor"
    

def generate_random_classifier():
  classifierLength = random.randint(1, 5)
  
  classifier = []
  
  for _ in range(classifierLength):
    feature_index = random.randint(0, 3)  # 0-3 inclusive
    operator = random.choice(["<", ">=", "=="])
    threshold = random.uniform(0, 10)
    class_label = random.choice(["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
    classifier.append((feature_index, operator, threshold, class_label))
    
  return classifier


def init_starting_population():
  print("Initializing Population...")
  
  population = []

  for _ in range(POPULATION_SIZE):
    population.append(generate_random_classifier())
  return population
        





def selection(population, iris_list, fittestIndividualsToSelect):

  fitness_scores = [(fitness(classifier, iris_list), classifier) for classifier in population]
  
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

  fitness_scores = [(fitness(classifier, iris_list), classifier) for classifier in population]
  
  fitness_scores.sort(reverse=True)

  return fitness_scores[0][0]
  
  

def crossover(classifiers):

    m = FITTEST_INDIVIDUALS_TO_SELECT-1
  
 
    new_classifiers = []
  
  # Perform crossover on the input classifiers to generate the new classifiers
    for _ in range(POPULATION_SIZE - m):
    
        parent1 = random.choice(classifiers)
        parent2 = random.choice(classifiers)
    
   
        child = crossover_point(parent1, parent2)
    
    
        new_classifiers.append(child)
    
  
    new_classifiers += classifiers

  
    return new_classifiers


def crossover_point(classifier1, classifier2):
  # Choose a random crossover point
  crossover_point = random.randint(1, len(classifier1))
  
  # Create a new classifier by combining the decision points of the parent classifiers
  new_classifier = classifier1[:crossover_point] + classifier2[crossover_point:]
  
  return new_classifier

def genetic_algorithm(population, iris_data):
    
    parent_population = population
    iterations = 0
    child_population_is_fitter = True

   
    
    while max_fitness_score(parent_population, iris_data) <= MAX_FITNESS and iterations != MAX_ITERATIONS:
        fit_individuals = selection(parent_population,iris_data, FITTEST_INDIVIDUALS_TO_SELECT)
        child_population = crossover(fit_individuals)
        child_population = mutation(child_population)
        #if max_fitness_score(child_population, iris_data) == max_fitness_score(parent_population, iris_data):
            #return child_population
        parent_population = child_population
        iterations+=1
        print(iterations)
        
    return parent_population

def clasify_iris_with_population(iris, population):

  classification = []

  for classifier in population:
    for feature_index, operator, threshold, class_label in classifier:
      feature_value = [iris["sepal_length"], iris["sepal_width"], iris["petal_length"], iris["petal_width"]][feature_index]
      if eval(str(feature_value) + operator + str(threshold)):
        classification.append(class_label)
        
      #classification.append("Iris-versicolor") 

  print("Setosa", classification.count("Iris-setosa"))
  print("Versicolor", classification.count("Iris-versicolor"))
  print("Virginica", classification.count("Iris-virginica"))

def mutation(classifiers):

  new_classifiers = []
  
  for classifier in classifiers:
    featureToMutate = random.randint(0, len(classifier)-1)

    mutation = random.uniform(0, 1) * random.uniform(0, 1) #Makes smaller changes more likely

    classifierAsList = list(classifier[featureToMutate])
    
    if(random.randint(0,1) == 0):
      classifierAsList[2] += mutation

    else: 
      classifierAsList[2] -= mutation

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
  cliPetalWidth= args.petal_width
  trainBool = args.train

  iris = {
    "sepal_length" : cliSepalLength,
    "sepal_width" : cliSepalWidth,
    "petal_length" : cliPetalLength,
    "petal_width" : cliPetalWidth,
  }

  if(trainBool):
    classifier_population = init_starting_population()
    iris_data = read_csv()
    selected_generation = genetic_algorithm(classifier_population, iris_data)

    with open("selected_generation.txt", "w") as fp:
      json.dump(selected_generation, fp)

    print(max_fitness_score(selected_generation, iris_data))
  else:
    with open("selected_generation.txt", "r") as fp:
      selected_generation = json.load(fp)

  clasify_iris_with_population(iris, selected_generation)

  #print(selected_generation)
  #print(classify_iris(iris, selected_generation[0]))

  
    
    
        
    
    
 
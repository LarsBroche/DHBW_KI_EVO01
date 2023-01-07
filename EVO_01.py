#Evolutinary algoithm in context of the iris dataset
#creates a classifier that uses the petal length 
import csv
import random

def read_csv():

    with open('iris.data_subset_1.csv', 'r') as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
        
    keys = ['sepal_width', 'sepal_length', 'petal_width', 'petal_length', 'iris_species']
    iris_data = [{keys[i]: row[i] for i in range(len(keys))} for row in rows]
    return iris_data

def classify_iris(iris, classifier):
  
  
  for feature_index, operator, threshold, class_label in classifier:
    feature_value = [iris["sepal_length"], iris["sepal_width"], iris["petal_length"], iris["petal_width"]][feature_index]
    if eval(str(feature_value) + operator + str(threshold)):
      return class_label
  return "Iris-versicolor"
    

def generate_random_classifier():
  n = random.randint(1, 5)
  
  classifier = []
  
  for _ in range(n):
    feature_index = random.randint(0, 3)  # 0-3 inclusive
    operator = random.choice(["<", ">=", "=="])
    threshold = random.uniform(0, 10)
    class_label = random.choice(["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
    classifier.append((feature_index, operator, threshold, class_label))
    
  return classifier


def init_starting_population():
    print("Initializing Population...")
    population_size = 225
    population = []
    for i in range(225):
        population.append(generate_random_classifier())
    return population
        

def fitness(classifier, iris_list):
  correct = 0
  total = len(iris_list)
  
  for iris in iris_list:
    
    predicted_species = classify_iris(iris, classifier)
    if predicted_species == iris["iris_species"]:
      correct += 1
      
  return correct / total


def selection(population, iris_list, n):

  fitness_scores = [(fitness(classifier, iris_list), classifier) for classifier in population]
  
  fitness_scores.sort(reverse=True)
  
  return [classifier for _, classifier in fitness_scores[:n]]

def max_fitness_score(population, iris_list):

  fitness_scores = [(fitness(classifier, iris_list), classifier) for classifier in population]
  
  fitness_scores.sort(reverse=True)
  
  
  
  return fitness_scores[0][0]
  
  

def crossover(classifiers):
    n = 225
  

    m = 15
  
 
    new_classifiers = []
  
  # Perform crossover on the input classifiers to generate the new classifiers
    for i in range(n - m):
    
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
    
    while max_fitness_score(parent_population, iris_data) <= 0.99 and iterations != 500:
        fit_individuals = selection(parent_population,iris_data, 16)
        child_population = crossover(fit_individuals)
        #if max_fitness_score(child_population, iris_data) == max_fitness_score(parent_population, iris_data):
            #return child_population
        parent_population = child_population
        iterations+=1
        print(iterations)
        
    return parent_population

if __name__ == "__main__":
    classifier_population = init_starting_population()
    iris_data = read_csv()
    selected_generation = genetic_algorithm(classifier_population, iris_data)
    print(max_fitness_score(selected_generation, iris_data))
    
    
        
    
    
    
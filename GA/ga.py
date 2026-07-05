import numpy as np
import matplotlib.pyplot as plt
import random as rd
import math

class GeneticAlgorithm:
    """Na inicializacao do algoritmo, deve-se informar os seguintes parametros:
        population_size: Quantidade maxima de individuos na populacao;
        max_generations: Maximo de iteracoes(geracoes) no algoritmo;
        crossover_frequency: Frequencia que ocorre recombinacao entre os individuos;
        mutation_frequency: Frequencia que ocorre mutacao entre os individuos;
        """
    def __init__(self, population_size,  max_generations=100, crossover_frequency=0.85, mutation_frequency=0.10):
        self.population_size=population_size
        self.max_generations=max_generations
        self.crossover_frequency=crossover_frequency
        self.mutation_frequency=mutation_frequency
        self.lows=np.array([1,1,1,10,1])
        self.highs=np.array([10,20,5,30,30])
        self.population=[]
        for i in range(population_size):#Inicializacao da populacao
            random_values=np.random.uniform(low=self.lows,high=self.highs)
            random_values[0]=int(random_values[0])#Fatorial precisa se um numero inteiro
            self.population.append(random_values)
        self.population=np.array(self.population)

    #Avaliacao das solucoes através da Fitness Funcion
    def fitness(self,population):
        e=np.e
        v=population[:,0].astype(int)
        w=population[:,1]
        y=population[:,2]
        x=population[:,3]
        z=population[:,4]
        factorial=np.vectorize(math.factorial)
        fitness_score=((((factorial(v))/100)*w)/np.power(y,1/x))-np.power(z,e-2)
        return fitness_score

    #Selecao dos individuos por meio da Selecao por Torneio
    def tournamentSelection(self,population,fitness_score,parents_size,n_candidates):
        parents=[]
        while len(parents)<parents_size:
            round_index=np.random.choice(len(fitness_score),size=n_candidates,replace=False)
            best_index=round_index[np.argmax(fitness_score[round_index])]
            parents.append(population[best_index])
        parents=np.array(parents)
        return parents

    #Mecanismos de Reproducao   
    def crossover(self,parents):
        parent_index=np.random.choice(len(parents),size=2,replace=False)
        parent=parents[parent_index]
        i=rd.randint(1,3)
        child1=np.concatenate([parent[0,0:i],parent[1,i:5]])
        child2=np.concatenate([parent[1,0:i],parent[0,i:5]])
        child1[0]=int(child1[0])
        child2[0]=int(child2[0])
        children=[child1,child2]
        return children

    def mutation(self,parents):
        child_index=np.random.choice(len(parents))
        child=np.copy(parents[child_index])
        i=rd.randint(0,4)  
        child[i]=rd.uniform(self.lows[i],self.highs[i])
        if i==0:
            child[i] = int(child[i])
        return child  

    #Processo de busca das melhores solucoes para otimizacao da funcao objetivo
    def generatePopulation(self,parents_size,n_candidates=2):
        n=0
        fitness_generation=[]
        while n<self.max_generations:
            fitness_score=self.fitness(self.population)
            parents=self.tournamentSelection(self.population,fitness_score,parents_size,n_candidates)
            new_population=list(parents)#Nos processos de reproducao, esta sendo utilizado a politica de geracao elitista, logo, tanto os pais quanto os filhos compõem a nova populacao
            while len(new_population)<self.population_size:
                if rd.random()<self.crossover_frequency:
                    children=self.crossover(parents)
                    for child in children:
                        if len(new_population)<self.population_size:
                            new_population.append(child)
                        else:
                            break
                if rd.random()<self.mutation_frequency:
                    child=self.mutation(parents)
                    new_population.append(child)
            self.population=np.array(new_population)
            best_index=np.argmax(fitness_score)
            fitness_generation.append(fitness_score[best_index])
            n+=1
        fitness_score=self.fitness(self.population)
        best_index=np.argmax(fitness_score)
        print("-----Resultados-----")
        print(f"Melhor individuo: {self.population[best_index]}")
        print(f"Melhor fitness: {fitness_score[best_index]}")
        fitness_generation=np.array(fitness_generation)
        accuracy_generation=(fitness_generation)/(fitness_score[best_index])
        generations=np.arange(len(fitness_generation))
        plt.figure(figsize=(10,6))
        plt.plot(generations,fitness_generation,marker='o',linestyle='-',color='b')
        plt.title("Evolucao do Fitness ao longo das geracoes")
        plt.xlabel("Geracao")
        plt.ylabel("Melhor Fitness")
        plt.grid(True)
        plt.show()
        plt.figure(figsize=(10,6))
        plt.plot(generations,accuracy_generation,marker='o',linestyle='-',color='r')
        plt.title("Evolucao da Acuracia ao longo das geracoes")
        plt.xlabel("Geracao")
        plt.ylabel("Acuracia")
        plt.grid(True)
        plt.show()


#Implementacao
ga=GeneticAlgorithm(population_size=100,max_generations=100)
ga.generatePopulation(parents_size=50)


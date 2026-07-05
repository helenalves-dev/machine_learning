"""Autora: Helen Braga Alves"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

"""A classe pso abaixo refere-se ao algoritmo Particle Swarm Opmization(PSO), um algoritmo de otimização que será utilizado para otimizar os seguintes parâmetro em uma MLP: Quantidadede neurônios por camada, tipo de ativação por camada e tipo de otimizador utilizado na rede. A fitness function utilizada é a acurácia do modelo para uma dada solução. """

class pso:
    """-----Inicialização da classe-----
       Parâmetros:
       w- Coeficiente inercial;
       c1- Coeficiente cognitivo;
       c2- Coeficiente social;
       X_train, X_val, y_train, y_val- Dados de treinamento e validação do algoritmo;
       hidden_layers- Quantidade de camadas ocultas na mlp;"""
    def __init__(self,w,c1,c2,X_train,X_val,y_train,y_val,hidden_layers=6):
        self.inertia=w
        self.cognitive=c1
        self.social=c2
        self.r=np.random.rand(2)
        particles_size=10
        self.X_train=X_train
        self.X_val=X_val
        self.y_train=y_train
        self.y_val=y_val
        self.hidden_layers=hidden_layers
        self.neurons=np.array([16,32,64,128,256,512])
        self.activation=np.array(['relu','tanh','sigmoid'])
        self.optimizer=np.array(['adam','sgd'])
        self.population=[]#Inicialização da população
        for _ in range(particles_size):
            particle=[]
            for j in range(0,hidden_layers):
                particle.append(np.random.randint(0,len(self.neurons)))
                particle.append(np.random.randint(0,len(self.activation)))
            particle.append(np.random.randint(0,len(self.optimizer)))
            self.population.append(particle)
        self.population=np.array(self.population)
    
    def fitness(self,particle):
        model=tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(3072,)))
        for i in range(0,len(particle)-1,2):
            neurons=self.neurons[particle[i]]
            activation=self.activation[particle[i+1]]
            model.add(tf.keras.layers.Dense(neurons,activation=activation))
            model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(10,activation='softmax'))
        optimizer=self.optimizer[particle[-1]]
        model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        model.fit(self.X_train,self.y_train,epochs=5,verbose=0)
        loss,accuracy=model.evaluate(self.X_val,self.y_val,verbose=0)
        return accuracy

    def position(self,position,speed):
        next_position=position+speed
        next_position=np.round(next_position).astype(int)
        for i in range(0,len(next_position)-2,2):
            next_position[i]=np.clip(next_position[i],0,len(self.neurons)-1)
            next_position[i+1]=np.clip(next_position[i+1],0,len(self.activation)-1)
        next_position[-1]=np.clip(next_position[-1],0,len(self.optimizer)-1)
        return next_position
        
    def speed(self,position,speed,pbest,gbest):
        w=self.inertia
        c1=self.cognitive
        c2=self.social
        r1=self.r[0]
        r2=self.r[1]
        next_speed=(w*speed)+(c1*r1*(pbest-position))+(c2*r2*(gbest-position))
        return next_speed

    def search(self,max_iterations=5):
        fitness_values=[]
        for i in range(len(self.population)):
            fitness_values.append(self.fitness(self.population[i]))
        fitness_values=np.array(fitness_values)#Cálculo do Fitness Inicial
        self.pbest=self.population.copy()#Definir Personal Best
        self.pbest_fitness=fitness_values
        self.gbest=self.population[np.argmax(fitness_values)]#Definir Group Best
        self.gbest_fitness=np.max(fitness_values)
        speeds=np.zeros_like(self.population,dtype='float')
        n=0
        accuracy_iterations=[]
        while n<max_iterations:#Iterações
            self.inertia=self.inertia*0.95
            for i in range(len(self.population)):
                speeds[i]=self.speed(self.population[i],speeds[i],self.pbest[i],self.gbest)#Atualização da Velocidade
                self.population[i]=self.position(self.population[i],speeds[i])#Atualização da posição das partículas
                fitness_values[i]=self.fitness(self.population[i])#Atualização do Fitness
                if fitness_values[i]>self.pbest_fitness[i]:
                    self.pbest[i]=self.population[i]#Atualização do pbest
                    self.pbest_fitness[i]=fitness_values[i]
            if np.max(fitness_values)>self.gbest_fitness:
                self.gbest=self.population[np.argmax(fitness_values)]#Atualização do gbest
                self.gbest_fitness=np.max(fitness_values)
            n+=1
            accuracy_iterations.append(self.gbest_fitness)
        print("-----Melhor Solução-----")
        j=1
        for i in range(0,len(self.gbest)-1,2):
            print(f"{j}ª Camada Oculta")
            print(f"Quantidade de Neurônios: {self.neurons[self.gbest[i]]}")
            print(f"Função de Ativação: {self.activation[self.gbest[i+1]]}")
            j+=1
        print(f"Otimizador: {self.optimizer[self.gbest[-1]]}")
        print(f"Acurácia: {self.gbest_fitness}")
        accuracy_iterations=np.array(accuracy_iterations)
        iterations=np.arange(len(accuracy_iterations))
        plt.figure(figsize=(10,6))
        plt.plot(iterations,accuracy_iterations,marker='o',linestyle='-',color='b')
        plt.title("Evolução da Acurácia ao longo da iterações")
        plt.xlabel("Iteração")
        plt.ylabel("Melhor Acurácia do Grupo")
        plt.grid(True)
        plt.show()

#Implementação
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

batch1=unpickle(file='data_batch_1')
batch2=unpickle(file='data_batch_2')
batch3=unpickle(file='data_batch_3')
batch4=unpickle(file='data_batch_4')
batch5=unpickle(file='data_batch_5')
images1=batch1[b'data']
images2=batch2[b'data']
images3=batch3[b'data']
images4=batch4[b'data']
images5=batch5[b'data']
labels1=batch1[b'labels']
labels2=batch2[b'labels']
labels3=batch3[b'labels']
labels4=batch4[b'labels']
labels5=batch5[b'labels']
X1=pd.DataFrame(images1)
X2=pd.DataFrame(images2)
X3=pd.DataFrame(images3)
X4=pd.DataFrame(images4)
X5=pd.DataFrame(images5)
X=pd.concat([X1,X2,X3,X4,X5],ignore_index=True)
X=X/255.0
y1=pd.Series(labels1)
y2=pd.Series(labels2)
y3=pd.Series(labels3)
y4=pd.Series(labels4)
y5=pd.Series(labels5)
y=pd.concat([y1,y2,y3,y4,y5],ignore_index=True)
X_train,X_val,y_train,y_val=train_test_split(X,y,random_state=42,test_size=0.25)
model=pso(w=2,c1=1.5,c2=2,X_train=X_train,X_val=X_val,y_train=y_train,y_val=y_val)
model.search()


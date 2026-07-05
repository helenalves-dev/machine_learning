import numpy as np
import pandas as pd

class mlp:
    #Inicialização do MLP
    def __init__(self,input_size,hidden_layer_size,hidden_neurons_size,output_size,activation='relu',loss='multi_class_cross_entropy'):
        self.input_size=input_size
        self.hidden_layer_size=hidden_layer_size
        self.hidden_neurons_size=hidden_neurons_size
        self.output_size=output_size
        self.activation=activation
        self.loss=loss
        self.layer_size=[input_size]+[hidden_neurons_size]*hidden_layer_size+[output_size]
        self.weight=[]
        self.bias=[]
        for i in range(len(self.layer_size)-1):
            w=np.random.rand(self.layer_size[i],self.layer_size[i+1])
            b=np.random.rand(self.layer_size[i+1])
            self.weight.append(w)
            self.bias.append(b)
    
    #Funções de ativação
    def relu(self,z):
        return np.maximum(0,z)
    def sigmoid(self,z):
        return 1/(1+(np.exp(-z)))
    def tanh(self,z):
        return np.tanh(z)
    def leaky_relu(self,z,alpha=0.01):
        return np.where(z>0,z,alpha*z)
    def softmax(self,z):
        e_z=np.exp(z-np.max(z))
        return e_z/np.sum(e_z)
    
    #Aplicar ativação
    def apply_activation(self,z,is_output=False):
        if is_output:
            return self.softmax(z)
        elif self.activation=='sigmoid':
            return self.sigmoid(z)
        elif self.activation=='tanh':
            return self.tanh(z)
        elif self.activation=='leaky_relu':
            return self.leaky_relu(z)
        else:
            return self.relu(z)
    
    #Forward pass
    def forward_pass(self,x):
        self.z_values=[]
        self.a_values=[x]
        a=x
        for i in range(len(self.layer_size)-1):
            z=np.dot(self.weight[i],a)+self.bias[i]
            a=self.apply_activation(z,is_output=(i==len(self.layer_size)-2))
            self.z_values.append(z)
            self.a_values.append(a)
        return a
    
    #Função de custo
    def binary_cross_entropy(self,y_real,y_pred,epsilon=1e-15):
        y_pred=np.clip(y_pred,epsilon,1-epsilon)
        loss=-(y_real*np.log(y_pred)+(1-y_real)*np.log(1-y_pred))
        return np.mean(loss)
    def multi_class_cross_entropy(self,y_real,y_pred,epsilon=1e-15):
        y_pred=np.clip(y_pred,epsilon,1-epsilon)
        return -np.sum(y_real*np.log(y_pred))
    
    #Aplicar a função de custo
    def apply_loss(self,output):
        if self.loss=='binary_cross_entropy':
            return self.binary_cross_entropy(output)
        else:
            return self.multi_class_cross_entropy(output)
        
    #Derivada: f(a+h)-f(a)/h, com h tendendo a 0
    def derivative(self,a,h=1e-15):
        return (self.apply_activation(a+h)-self.apply_activation(a))/h
    
    #Backpropagation
    def backward_pass(self,y_real,y_pred,learning_rate=0.01):
        dw=[None]*len(self.weight)
        db=[None]*len(self.bias)
        for i in range(len(self.weight),0,-1):
            z=self.z_values[i]
            

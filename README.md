# Machine Learning - Projetos de Algoritmos

Repositório contendo implementações de diversos algoritmos de aprendizado de máquina e otimização, desenvolvidos como estudos e práticas em técnicas de IA.

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Projetos](#projetos)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Requisitos](#requisitos)
- [Como Usar](#como-usar)

---

## 🎯 Visão Geral

Este repositório reúne implementações de algoritmos fundamentais em machine learning e inteligência artificial, incluindo:
- **Algoritmos de Classificação**: KNN, MLP, SVM, Árvores de Decisão, Random Forest
- **Algoritmos de Otimização**: Algoritmos Genéticos (GA), Otimização por Enxame de Partículas (PSO)

Cada projeto está organizado em sua própria pasta com implementação do algoritmo e, quando disponível, testes e dados.

---

## 📁 Projetos

### 1. **KNN (K-Nearest Neighbors)**
**Localização**: `KNN/`

Implementação do algoritmo KNN, um método de classificação baseado em instâncias que classifica um ponto de dados com base nos K vizinhos mais próximos.

**Características**:
- Classificação não-paramétrica
- Baseado em distância (geralmente Euclidiana)
- Simples e eficaz para datasets pequenos e médios
- Sensível à escolha do valor de K

**Arquivos**: `knn.py`

---

### 2. **Árvores de Decisão (Decision Trees)**
**Localização**: `Árvores de Decisão/`

Implementação de árvores de decisão, um método de aprendizado supervisionado usado para classificação e regressão.

**Características**:
- Fácil interpretação e visualização
- Não requer normalização de dados
- Pode sofrer com overfitting
- Usa critério de divisão (Gini ou Entropia)

**Arquivos**: `decision_tree.py`

---

### 3. **MLP (Multi-Layer Perceptron)**
**Localização**: `MLP/`

Implementação de redes neurais artificiais com múltiplas camadas, incluindo exemplos com o dataset MNIST.

**Características**:
- Redes neurais feedforward
- Múltiplas camadas (entrada, ocultas, saída)
- Aprendizado por retropropagação
- Capaz de aprender padrões complexos não-lineares

**Arquivos**: 
- `MLP.py` - Implementação da rede neural
- `MNIST.py` - Aplicação com dataset MNIST (dígitos manuscritos)

---

### 4. **Random Forest**
**Localização**: `Random Forest/`

Implementação do algoritmo Random Forest, um método de ensemble que combina múltiplas árvores de decisão.

**Características**:
- Reduz overfitting através de múltiplas árvores
- Melhor performance que árvores de decisão individuais
- Seleção aleatória de features e amostras
- Bom balanceamento entre viés e variância

**Arquivos**: `random_forest.py`

---

### 5. **SVM (Support Vector Machine)**
**Localização**: `SVM/`

Implementação de máquinas de vetores de suporte, um algoritmo poderoso para classificação e regressão.

**Características**:
- Encontra o hiperplano ótimo que separa classes
- Eficaz em espaços de alta dimensionalidade
- Funciona bem com dados não-lineares (usando kernels)
- Robusto a outliers

**Arquivos**: `svm.py`

---

### 6. **GA (Algoritmos Genéticos)**
**Localização**: `GA/`

Implementação de algoritmos genéticos, uma técnica metaheurística inspirada na evolução biológica.

**Características**:
- Simula processo de seleção natural
- Usa operadores: seleção, cruzamento, mutação
- Útil para problemas de otimização complexos
- Exploração e exploração do espaço de soluções

**Arquivos**: 
- `ga.py` - Implementação do algoritmo
- `testes/` - Pasta para testes e validações

---

### 7. **PSO (Particle Swarm Optimization)**
**Localização**: `PSO/`

Implementação de otimização por enxame de partículas, inspirada no comportamento de enxames de pássaros e cardumes.

**Características**:
- Algoritmo metaheurístico de otimização
- Partículas se movem no espaço de busca
- Compartilham informação sobre melhores posições
- Excelente para otimização em espaços contínuos

**Arquivos**: 
- `pso.py` - Implementação do algoritmo
- `testes/` - Pasta contendo datasets CIFAR-10 para testes
  - `data_batch_1` a `data_batch_5` - Batches de treinamento
  - `test_batch` - Batch de teste

---

## 📂 Estrutura do Repositório

```
machine_learning/
├── README.md
├── Árvores de Decisão/
│   └── decision_tree.py
├── GA/
│   ├── ga.py
│   └── testes/
├── KNN/
│   └── knn.py
├── MLP/
│   ├── MLP.py
│   └── MNIST.py
├── PSO/
│   ├── pso.py
│   └── testes/
│       ├── data_batch_1 a 5
│       └── test_batch
├── Random Forest/
│   └── random_forest.py
└── SVM/
    └── svm.py
```

---

## 🛠️ Requisitos

Para executar os projetos, você precisará ter instalado:

- **Python 3.7+**
- Bibliotecas comuns:
  - `numpy` - Computação numérica
  - `pandas` - Manipulação de dados
  - `scikit-learn` - Algoritmos de ML (algumas implementações podem usar)
  - `matplotlib` ou `seaborn` - Visualização
  - Outras dependências específicas podem ser listadas em cada projeto

### Instalação de Dependências

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

---

## 🚀 Como Usar

1. **Clone ou acesse o repositório**
   ```bash
   cd machine_learning
   ```

2. **Acesse a pasta do projeto desejado**
   ```bash
   cd KNN
   ```

3. **Execute o script Python**
   ```bash
   python knn.py
   ```

4. **Para o projeto MNIST (MLP)**
   ```bash
   cd MLP
   python MNIST.py
   ```

5. **Para o projeto PSO com CIFAR-10**
   ```bash
   cd PSO
   python pso.py
   ```

---

## 📚 Conceitos Principais

| Algoritmo | Tipo | Aplicação | Complexidade |
|-----------|------|-----------|--------------|
| KNN | Supervisionado | Classificação | Média |
| Árvore de Decisão | Supervisionado | Classificação/Regressão | Baixa-Média |
| MLP | Supervisionado | Classificação/Regressão | Alta |
| Random Forest | Supervisionado | Classificação/Regressão | Alta |
| SVM | Supervisionado | Classificação/Regressão | Alta |
| GA | Otimização | Otimização Combinatória | Alta |
| PSO | Otimização | Otimização Contínua | Média |

---

## 📝 Notas

- Alguns projetos incluem dados de teste (como PSO com CIFAR-10)
- Cada implementação serve como estudo e prática dos conceitos
- Os algoritmos podem não estar otimizados para produção
- Consulte comentários nos códigos-fonte para detalhes técnicos

---

## 🎓 Recursos Adicionais

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Machine Learning Mastery](https://machinelearningmastery.com/)
- [Stanford CS229 - Machine Learning](http://cs229.stanford.edu/)

---

**Desenvolvido como material de estudo em Aprendizado de Máquina**

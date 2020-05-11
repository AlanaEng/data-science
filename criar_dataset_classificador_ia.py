'''
Manipulando múltiplos arquivos de dados para criar um dataset de treinamento para um classificador de IA. 
Neste exemplos o objetivo é criar um dataset com 4 classes (4 x 1200).
Cada classe possui 3 elementos do memso tipo de dado (1 x 400).
'''
import pandas as pd
import numpy as np

#efetua leitura dos dados por classe
pd1 = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Classificador/Dataset/classe01.csv')
pd2 = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Classificador/Dataset/classe02.csv')
pd3 = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Classificador/Dataset/classe03.csv')

pd4 = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Classificador/Dataset/classe04.csv')
pd5 = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Classificador/Dataset/classe05.csv')
pd6 = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Classificador/Dataset/classe06.csv')

pd7 = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Classificador/Dataset/classe07.csv')
pd8 = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Classificador/Dataset/classe08.csv')
pd9 = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Classificador/Dataset/classe09.csv')

pd10 = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Classificador/Dataset/classe10.csv')
pd11 = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Classificador/Dataset/classe11.csv')
pd12 = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Classificador/Dataset/classe12.csv')

# exclui colunas que não serão usadas
pd1.drop(columns=['Ano', 'Trimestre', 'Mês', 'Dia', 'Ordem','Tempo'], inplace=True)
pd2.drop(columns=['Ano', 'Trimestre', 'Mês', 'Dia', 'Ordem','Tempo'], inplace=True)
pd3.drop(columns=['Ano', 'Trimestre', 'Mês', 'Dia', 'Ordem','Tempo'], inplace=True)
pd4.drop(columns=['Ano', 'Trimestre', 'Mês', 'Dia', 'Ordem','Tempo'], inplace=True)
pd5.drop(columns=['Ano', 'Trimestre', 'Mês', 'Dia', 'Ordem','Tempo'], inplace=True)
pd6.drop(columns=['Ano', 'Trimestre', 'Mês', 'Dia', 'Ordem','Tempo'], inplace=True)
pd7.drop(columns=['Ano', 'Trimestre', 'Mês', 'Dia', 'Ordem','Tempo'], inplace=True)
pd8.drop(columns=['Ano', 'Trimestre', 'Mês', 'Dia', 'Ordem','Tempo'], inplace=True)
pd9.drop(columns=['Ano', 'Trimestre', 'Mês', 'Dia', 'Ordem','Tempo'], inplace=True)
pd10.drop(columns=['Ano', 'Trimestre', 'Mês', 'Dia', 'Ordem','Tempo'], inplace=True)
pd11.drop(columns=['Ano', 'Trimestre', 'Mês', 'Dia', 'Ordem','Tempo'], inplace=True)
pd12.drop(columns=['Ano', 'Trimestre', 'Mês', 'Dia', 'Ordem','Tempo'], inplace=True)

# reformata cada matriz para 400 linhas
pd1 = pd1.head(400)
pd2 = pd2.head(400)
pd3 = pd3.head(400)
pd4 = pd4.head(400)
pd5 = pd5.head(400)
pd6 = pd6.head(400)
pd7 = pd7.head(400)
pd8 = pd8.head(400)
pd9 = pd9.head(400)
pd10 = pd10.head(400)
pd11 = pd11.head(400)
pd12 = pd12.head(400)

# converte cada matriz para array
pd1 = pd1.values
pd2 = pd2.values
pd3 = pd3.values
pd4 = pd4.values
pd5 = pd5.values
pd6 = pd6.values
pd7 = pd7.values
pd8 = pd8.values
pd9 = pd9.values
pd10 = pd10.values
pd11 = pd11.values
pd12 = pd12.values

# concatena as linhas que são da mesma classe
conc1 = np.concatenate([pd1,pd2,pd3])
conc2 = np.concatenate([pd4,pd5,pd6])
conc3 = np.concatenate([pd7,pd8,pd9])
conc4 = np.concatenate([pd10,pd11,pd12])

# converte array para pandas
df1 = pd.DataFrame(conc1)
df2 = pd.DataFrame(conc2)
df3 = pd.DataFrame(conc3)
df4 = pd.DataFrame(conc4)

# inverte a matriz
df1 = df1.T
df2 = df2.T
df3 = df3.T
df4 = df4.T

# une todas as linhas 
dataset = df1.append(df2)
dataset = dataset.append(df3)
dataset = dataset.append(df4)

# converte a matriz inteira para array 
df = dataset.values

# cria um novo dataset
df = pd.DataFrame(df)
size = df.shape
table = df.head()
print(size)
print(table)

#salva o dataset na pasta 
df.to_csv('/content/drive/My Drive/Colab Notebooks/Classificador/Dataset/dados.csv')

#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importando bibliotecas
import pickle
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#Criando interface inicial
#Menu
st.sidebar.subheader('Projeto de portfólio de Ciência de Dados')
st.sidebar.markdown('''Em breve, disponibilizo o artigo descrevendo o passo a passo do desenvolvimento do modelo.''')
st.sidebar.title('Menu')
pag = st.sidebar.selectbox('Selecione a página', ['Interagir com a inteligência', 'Sobre o projeto', 'Dashboard'])

st.sidebar.markdown('Feito por : Bruno Rodrigues Carloto')

#Disponibilizando as redes sociais
st.sidebar.markdown("Redes Sociais :")
st.sidebar.markdown("- [Linkedin](https://www.linkedin.com/in/bruno-rodrigues-carloto)")
st.sidebar.markdown("- [Medium](https://br-cienciadedados.medium.com)")
st.sidebar.markdown("- [Github](https://github.com/brunnosjob)")

#Apresentação da tela de interação com o modelo
if pag == 'Interagir com a inteligência':
  st.markdown('*__Observação: para mais informações acerca do projeto, clique na seta no canto esquerdo superior da tela__* ')
  st.header('Mercado Delivery')
  st.subheader('Se cadastre em nosso app')
  st.markdown(' ')
  st.markdown(' ')
  st.markdown(' ')
  st.markdown(' ')
  
  #Chamando o modelo
  with open('cluster_fmcg.pkl', 'rb') as file:
    onehot, scaler, kmeans = pickle.load(file)

  #Criando espaços de preenchimento
  cliente = st.text_input('Insira seu nome:')
  renda = st.number_input('Insira sua renda:', 0, 1000000, 0)
  idade = st.number_input('Insira sua idade:', 18, 150, 18)
  educacao = st.selectbox('Selecione sua escolaridade:', ['Ensino básico', 'Graduado (a)', 'Especializado (a)', 'Doutor (a)'])
  ocupacao = st.selectbox('Selecione sua condição atual:',['Desempregado', 'Empregado', 'Empreendedor'])
  sexo = st.selectbox('Selecione seu sexo:', ['Feminino', 'Masculino'])
  estado_civil = st.selectbox('Selecione seu estado civil:', ['Solteiro', 'Casado'])
  tamanho_cidade = st.selectbox('Selecione o tipo de cidade em que você reside:', ['Cidade pequena', 'Cidade média', 'Cidade grande'])
  
  #Convertendo inputs educação
  if educacao == 'Ensino básico':
    educacao_num = 0
  elif educacao == 'Graduado (a)':
    educacao_num = 1
  elif educacao == 'Especializado (a)':
    educacao_num = 2
  elif educacao == 'Doutor (a)':
    educacao_num = 3
    
  #Convertendo inputs ocupação
  if ocupacao == 'Desempregado':
    ocupacao_num = 0
  elif ocupacao == 'Empregado':
    ocupacao_num = 1
  elif ocupacao == 'Empreendedor':
    ocupacao_num = 2
   
    
  #Convertendo inputs sexo
  if sexo == 'Feminino':
    sexo_num = 0
  elif sexo == 'Masculino':
    sexo_num = 1
    
  #Convertendo inputs estado civil
  if estado_civil == 'Solteiro':
    estado_civil_num = 0
  elif estado_civil == 'Casado':
    estado_civil_num = 1
    
  #Convertendo inputs tamanho da cidade
  if tamanho_cidade == 'Cidade pequena':
    tamanho_cidade_num = 0
  elif tamanho_cidade == 'Cidade média':
    tamanho_cidade_num = 1
  elif tamanho_cidade == 'Cidade grande':
    tamanho_cidade_num = 2
    
  #Pré-processamento
  #Separando variáveis que não receberão o algoritmo OneHotEncoder e gerando dataframe
  variaveis_sem_onehot = [[renda, idade, educacao_num, ocupacao_num]]
  variaveis_sem_onehot_df = pd.DataFrame(variaveis_sem_onehot)
    
  #Aplicando OneHotEncoder
  #Separando variáveis que receberão o algoritmo OneHotEncoder e gerando dataframe
  variaveis_onehot = [[sexo_num, estado_civil_num, tamanho_cidade_num]]
  variaveis_onehot_aplicada = onehot.transform(variaveis_onehot).toarray()
  variaveis_onehot_aplicada_df= pd.DataFrame(variaveis_onehot_aplicada)
    
  #Concatenando as variáveis
  df = pd.concat([variaveis_sem_onehot_df, variaveis_onehot_aplicada_df], axis=1)
    
  #Padronizando
  X_scaled = scaler.transform(df)
    
  #Segmentando
  segmento = kmeans.predict(X_scaled)
    
  #Instrução para o resultado
  if segmento == 0:
    st.markdown('''
    {}, você pertence ao público alfa.
    
    __Descrição do público alfa:__
    
    O público alfa é predominantemente feminino,
    havendo uma menor quantidade masculina, com idade de aproximadamente 30 anos, estando em algum tipo de relacionamento, como casamento ou união estável,
    com educação formal básica, residindo maiormente em pequenas cidades. A maior parte está empregada, entretanto, significativa parte está desempregada.
    Nesse público há poucos empreendedores. A renda desse público está em torno de R$ 2.300,00.
    
    '''.format(cliente))
   
  elif segmento == 1:
    st.markdown('''
    {}, você pertence ao público beta.
    
    __Descrição do público beta:__
    
    O público beta é composto pela maioria masculina, todavia, uma quantidade considerável de mulheres compõem esse grupo. A idade desse público está em cerca de 35 anos.
    É um público solteiro e graduado, residindo em pequenas cidades. A maior parte do grupo está desempregada. Mas há uma relevante quantidade de pessoas empregadas.
    Esse público não tem perfil empreendedor. Sua renda está em torno de R$ 2.000,00. 25% do público apresenta renda de até aproximadamente 1.600,00.
    '''.format(cliente))
    
  elif segmento == 2:
    st.markdown('''
    {}, você pertence ao público gama.
    
    __Descrição do público gama:__
    
    O público gama é predominantemente masculino solteiro, com idade de aproximadamente 38 anos. Há uma relevante quantidade de componentes em relacionamento,
    como casamento ou união estável. Possuem ensino superior e residem em médias e grandes cidades. É o público mais empreendedor, sendo a maior parte empregada,
    e a menor parte empreendedora.
    '''.format(cliente))

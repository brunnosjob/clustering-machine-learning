#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importando bibliotecas
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
  st.header('Bem-vindo (a) à web app de segmentação de cliente')
  st.subheader('Tecnologia machine learning aplicada ao interesse do marketing')
  st.markdown(' ')
  st.markdown(' ')

  #Criando espaços de preenchimento
  cliente = st.text_input('Insira o nome do cliente:')
  renda = st.number_input('Insira a renda do cliente:', 0, 1000000, 0)
  idade = st.number_input('Insira a idade do cliente:', 18, 150, 18)
  educacao = st.selectbox('Selecione a escolaridade do cliente:', ['Ensino básico', 'Graduado (a)', 'Especializado (a)', 'Doutor (a)'])
  ocupacao = st.selectbox('Selecione a ocupação do cliente:',['Desempregado', 'Empregado', 'Empreendedor'])
  sexo = st.selectbox('Selecione o sexo do cliente:', ['Feminino', 'Masculino'])
  estado_civil = st.selectbox('Selecione o estado civil do cliente:', ['Solteiro', 'Casado'])
  tamanho_cidade = st.selectbox('Selecione o tamanho da cidade em que o cliente reside:', ['Pequena', 'Média', 'Grande'])
  
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
    variaveis_onehot_aplicada = onehot.transform(variaveis_onehot)
    variaveis_onehot_aplicada_df= pd.DataFrame(variaveis_onehot_aplicada)
    
    #Concatenando as variáveis
    concat_df = pd.concat([variaveis_sem_onehot_df, variaveis_onehot_aplicada_df], axis=1)
    
    #Padronizando
    X_scaled = std_scaler.transform(concat_df)
    
    #Segmentando
    segmento = kmeans.predict(X_scaled)
    
    #Resultado
    st.write(segmento)
    
    
    

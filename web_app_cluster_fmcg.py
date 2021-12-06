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


# In[ ]:


#Criando interface inicial
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
estado_civil = st.selectbox('Selecione o estado civil do cliente:', ['Solteiro', 'Em uma relação'])
tamanho_cidade = st.selectbox('Selecione o tamanho da cidade em que o cliente reside:', ['Pequena', 'Média', 'Grande'])

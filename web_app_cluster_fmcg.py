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

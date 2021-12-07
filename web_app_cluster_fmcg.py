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
from PIL import Image

#Criando interface inicial
#Menu
st.sidebar.subheader('Projeto de portfólio de Ciência de Dados')
st.sidebar.markdown('''Em breve, disponibilizo o artigo descrevendo o passo a passo do desenvolvimento do modelo.''')
st.sidebar.title('Menu em construção')
pag = st.sidebar.selectbox('Selecione a página', ['Interagir com a inteligência', 'Questão de negócio', 'Dashboard acerca dos públicos'])

st.sidebar.markdown('Feito por : Bruno Rodrigues Carloto')

#Disponibilizando as redes sociais
st.sidebar.markdown("Redes Sociais :")
st.sidebar.markdown("- [Linkedin](https://www.linkedin.com/in/bruno-rodrigues-carloto)")
st.sidebar.markdown("- [Medium](https://br-cienciadedados.medium.com)")
st.sidebar.markdown("- [Github](https://github.com/brunnosjob)")

#Apresentação da tela de interação com o modelo
if pag == 'Interagir com a inteligência':
  st.markdown('*__Observação: para mais informações acerca do projeto, clique na seta no canto esquerdo superior da tela__* ')
  st.markdown('*(Os valores a seguir podem ser reais ou fictícios. Os dados não são salvos. Essa é uma web app de simulação)*')
  st.header('Vitta Delivery')
  st.subheader('Se cadastre em nosso app')
  
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
    ---
    ##### Informação para a empresa
    
    Há três públicos.
    
    __A que público o cliente pertence:__
    
    {} pertence ao público alfa.
    
    __Descrição do público alfa:__
    
    O público alfa é predominantemente feminino,
    havendo uma menor quantidade masculina. Esse é um público com idade de aproximadamente 30 anos, estando em algum tipo de relacionamento, como casamento ou união estável,
    com educação formal básica, residindo maiormente em pequenas cidades. A maior parte está empregada, entretanto, significativa parte está desempregada.
    Nesse público há poucos empreendedores. A renda desse público está em torno de R$ 2.300,00.
    
    '''.format(cliente))
   
  elif segmento == 1:
    st.markdown('''
    
    ---
    ##### Informação para a empresa
    
    Há três públicos.
    
    __A que público o cliente pertence:__
    
    {} pertence ao público beta.
    
    __Descrição do público beta:__
    
    O público beta é composto pela maioria masculina, todavia, uma quantidade considerável de mulheres compõem esse grupo. A idade desse público está em cerca de 35 anos.
    É um público solteiro e graduado, residindo em pequenas cidades. A maior parte do grupo está desempregada. Mas há uma relevante quantidade de pessoas empregadas.
    Esse público não tem perfil empreendedor. Sua renda está em torno de R$ 2.000,00. 25% do público apresenta renda de até aproximadamente R$ 1.600,00.
    '''.format(cliente))
    
  elif segmento == 2:
    st.markdown('''
    
    ---
    ##### Informação para a empresa
    
    Há três públicos.
    
    __A que público o cliente pertence:__
    
    {} pertence ao público gama.
    
    __Descrição do público gama:__
    
    O público gama é predominantemente masculino solteiro, com idade de aproximadamente 38 anos. Há uma relevante quantidade de pessoas em relacionamento,
    como casamento ou união estável. Possuem ensino superior e residem em médias e grandes cidades. É o público mais empreendedor, sendo a maior parte empregada,
    e a menor parte empreendedora. A renda desse público está em torno de R$ 2.900,00.
    '''.format(cliente))
    
elif pag == 'Questão de negócio':
  st.header('Questão de negócio')
  st.markdown('''
  O grupo gestor do supermercado (fictício) Vitta está investindo em tecnologia.
  Seu último projeto bem sucedido foi a criação do Vitta Delivery.
  A empresa de varejo desenvolveu um aplicativo a partir do qual seus clientes realizam suas compras onlines e o supermercado faz as entregas.
  
  O supermercado está com um novo projeto a desenvolver e deseja implementar um piloto.
  O projeto se constitui do interesse por segmentar seus clientes para elaborar campanhas de marketing específicas para os públicos certos.
  O envio das campanhas é feito via app diretamente ao celular do usuário.
  
  __Plano de trabalho:__
  
  1 - Segmentar os clientes em públicos;
  
  2 - Compreender o perfil social e socioeconômico dos públicos;
  
  3 - Compreender o consumo dos públicos;
  
  4 - Elaborar campanhas de marketing a partir dos pontos 1, 2 e 3;
  
  5 - Definir os métodos de entrega das campanhas;
  
  6 - Acompanhar por métricas as respostas dos clientes às entregas e às campanhas;
  
  7 - Alterar/ajustar os métodos de entrega das campanhas e as campanhas;
  
  8 - Conclusão do piloto
  
  A aplicação do piloto terá um primeiro prazo, que é de três meses. Após, terá mais 3 meses.
  
  __Etapas:__ 
  
  - A primeira  e a segunda semana de cada mês ficarão para o desenvolvimento de novas campanhas ou o aprimoramento das já aplicadas;
  
  - A terceira e a quarta semana ficarão para análise da resposta e reação dos clientes;
  
  __Métricas a serem observadas:__
  
  - Cliques para abrir as propagandas;
  
  - Aumento de compras online por parte dos públicos.
  
__Métodos de entregas a serem testados:__

- Vídeo-propaganda diretamente em tela de celular após abertura de aplicativo;

- Imagem gráfica diretamente em tela de celular após abertura de aplicativo;

- Vídeo-propaganda via link em email;

- Imagem gráfica via link em email;

- Vídeo-propaganda via link em sms;

- Imagem gráfica via link em sms.
  
Ao fim de cada três meses, há uma reunião com os resultados trimestrais do piloto.
  ''')
  
  st.markdown('#### Etapa atual')
  st.markdown('''
  A presente etapa se constitui em segmentar os clientes, 
  realizar análises e desenvolver um modelo de machine learning que segmente cada novo cliente cadastrado pelo aplicativo.
  Somente a empresa terá acesso ao pertencimento de cada cliente aos seus respectivos públicos.
  ''')
  
elif pag == 'Dashboard acerca dos públicos':
  st.header('Dashboard acerca dos públicos')
  st.subheader('Gráfico de segmentação')
  pca = Image.open('pca.png')
  st.image(pca, use_column_width=True)
  st.markdown(' ')
  st.markdown(' ') 
  st.markdown(' ')
  st.markdown(' ') 
                   
  st.subheader('Contagem de clientes em cada público/grupo')
  contagem = Image.open('frequencia_cada_publico.png')
  st.image(contagem, use_column_width=True)  
  st.markdown(' ')
  st.markdown(' ') 
  st.markdown(' ')
  st.markdown(' ') 
  
  st.subheader('Público alfa')
  
  photo0 = Image.open('0_renda.png')
  st.image(photo0, use_column_width=True)
  st.markdown(' ')
  st.markdown(' ')
  
  photo1 = Image.open('0_idade.png')
  st.image(photo1, use_column_width=True)
  st.markdown(' ')
  st.markdown(' ')
  
  photo2 = Image.open('0_educacao.png')
  st.image(photo2, use_column_width=True)
  st.markdown('''
  
  __Legenda:__
  
  0 - Ensino básico
  
  1 - Graduado (a)
  
  2 - Especializado (a)
  
  3 - Doutor (a)
  ''')
  st.markdown(' ')
  st.markdown(' ')
  
  photo3 = Image.open('0_ocupacao.png')
  st.image(photo3, use_column_width=True)
  st.markdown('''
  
  __Legenda:__
  
  0 - Desempregado
  
  1 - Empregado
  
  2 - Empreendedor
  ''')
  st.markdown(' ')
  st.markdown(' ')
  
  photo4 = Image.open('0_sexo.png')
  st.image(photo4, use_column_width=True)
  st.markdown('''
  
  __Legenda:__
  
  0 - Masculino
  
  1 - Feminino
  
  ''')
  st.markdown(' ')
  st.markdown(' ')
 
  photo5 = Image.open('0_estado_civil.png')
  st.image(photo5, use_column_width=True)
  st.markdown('''
  
  __Legenda:__
  
  0 - Solteiro
  
  1 - Em relacionamento (Casado ou união estável)
  ''')
  st.markdown(' ')
  st.markdown(' ')
  
  photo6 = Image.open('0_tamanho_cidade.png')
  st.image(photo6, use_column_width=True)
  st.markdown('''
  
  __Legenda:__
  
  0 - Cidade pequena
  
  1 - Cidade média
  
  2 - Cidade grande
  ''')
  st.markdown(' ')
  st.markdown(' ') 
  st.markdown(' ')
  st.markdown(' ') 

  st.subheader('Público beta')
  
  photo10 = Image.open('1_renda.png')
  st.image(photo10, use_column_width=True)
  st.markdown(' ')
  st.markdown(' ')
  
  photo11 = Image.open('1_idade.png')
  st.image(photo11, use_column_width=True)
  st.markdown(' ')
  st.markdown(' ')
  
  photo12 = Image.open('1_educacao.png')
  st.image(photo12, use_column_width=True)
  st.markdown('''
  
  __Legenda:__
  
  0 - Ensino básico
  
  1 - Graduado (a)
  
  2 - Especializado (a)
  
  3 - Doutor (a)
  ''')
  st.markdown(' ')
  st.markdown(' ')
  
  photo13 = Image.open('1_ocupacao.png')
  st.image(photo13, use_column_width=True)
  st.markdown('''
  
  __Legenda:__
  
  0 - Desempregado
  
  1 - Empregado
  
  2 - Empreendedor
  ''')
  st.markdown(' ')
  st.markdown(' ')
  
  photo14 = Image.open('1_sexo.png')
  st.image(photo14, use_column_width=True)
  st.markdown('''
  
  __Legenda:__
  
  0 - Masculino
  
  1 - Feminino
  
  ''')
  st.markdown(' ')
  st.markdown(' ')
 
  photo15 = Image.open('1_estado_civil.png')
  st.image(photo15, use_column_width=True)
  st.markdown('''
  
  __Legenda:__
  
  0 - Solteiro
  
  1 - Em relacionamento (Casado ou união estável)
  ''')
  st.markdown(' ')
  st.markdown(' ')
  
  photo16 = Image.open('1_tamanho_cidade.png')
  st.image(photo16, use_column_width=True)
  st.markdown('''
  
  __Legenda:__
  
  0 - Cidade pequena
  
  1 - Cidade média
  
  2 - Cidade grande
  ''')
  st.markdown(' ')
  st.markdown(' ') 
  st.markdown(' ')
  st.markdown(' ') 

  st.subheader('Público gamma')
  
  photo20 = Image.open('2_renda.png')
  st.image(photo20, use_column_width=True)
  st.markdown(' ')
  st.markdown(' ')
  
  photo21 = Image.open('2_idade.png')
  st.image(photo21, use_column_width=True)
  st.markdown(' ')
  st.markdown(' ')
  
  photo22 = Image.open('2_educacao.png')
  st.image(photo22, use_column_width=True)
  st.markdown('''
  
  __Legenda:__
  
  0 - Ensino básico
  
  1 - Graduado (a)
  
  2 - Especializado (a)
  
  3 - Doutor (a)
  ''')
  st.markdown(' ')
  st.markdown(' ')
  
  photo23 = Image.open('2_ocupacao.png')
  st.image(photo23, use_column_width=True)
  st.markdown('''
  
  __Legenda:__
  
  0 - Desempregado
  
  1 - Empregado
  
  2 - Empreendedor
  ''')
  st.markdown(' ')
  st.markdown(' ')
  
  photo24 = Image.open('2_sexo.png')
  st.image(photo24, use_column_width=True)
  st.markdown('''
  
  __Legenda:__
  
  0 - Masculino
  
  1 - Feminino
  
  ''')
  st.markdown(' ')
  st.markdown(' ')
 
  photo25 = Image.open('2_estado_civil.png')
  st.image(photo25, use_column_width=True)
  st.markdown('''
  
  __Legenda:__
  
  0 - Solteiro
  
  1 - Em relacionamento (Casado ou união estável)
  ''')
  st.markdown(' ')
  st.markdown(' ')
  
  photo26 = Image.open('2_tamanho_cidade.png')
  st.image(photo26, use_column_width=True)
  st.markdown('''
  
  __Legenda:__
  
  0 - Cidade pequena
  
  1 - Cidade média
  
  2 - Cidade grande
  ''')  


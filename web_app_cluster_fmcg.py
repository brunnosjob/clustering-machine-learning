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
  st.markdown('#### Sistema em atualização')
  st.markdown('*__Observação: para mais informações acerca do projeto, clique na seta no canto esquerdo superior da tela__* ')
  st.markdown('*(Os valores a seguir podem ser reais ou fictícios. Os dados não são salvos. Essa é uma web app de simulação)*')
  st.header('Vitta Delivery')
  st.subheader('Se cadastre em nosso app')
  
  #Chamando o modelo
  with open('cluster_vitta_delivery.pkl', 'rb') as file:
    onehot, scaler, kmeans = pickle.load(file)

  #Criando espaços de preenchimento
  cliente = st.text_input('Insira seu nome:')
  renda = st.number_input('Insira sua renda:', 0, 1000000, 0)
  idade = st.number_input('Insira sua idade:', 18, 150, 18)
  educacao = st.selectbox('Selecione sua escolaridade:', ['Ensino básico', 'Graduado', 'Especializado', 'Doutor'])
  ocupacao = st.selectbox('Selecione sua condição atual:',['Desempregado', 'Empregado', 'Empreendedor'])
  sexo = st.selectbox('Selecione seu sexo:', ['Feminino', 'Masculino'])
  estado_civil = st.selectbox('Selecione seu estado civil:', ['Solteiro', 'Casado'])
  
  #Convertendo inputs educação
  if educacao == 'Ensino básico':
    educacao_num = 0
  elif educacao == 'Graduado':
    educacao_num = 1
  elif educacao == 'Especializado':
    educacao_num = 2
  elif educacao == 'Doutor':
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
  
    
  #Pré-processamento
  #Separando variáveis que não receberão o algoritmo OneHotEncoder e gerando dataframe
  variaveis_sem_onehot = [[renda, idade, educacao_num, ocupacao_num]]
  variaveis_sem_onehot_df = pd.DataFrame(variaveis_sem_onehot)
    
  #Aplicando OneHotEncoder
  #Separando variáveis que receberão o algoritmo OneHotEncoder e gerando dataframe
  variaveis_onehot = [[sexo_num, estado_civil_num]]
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
    
    Há cinco públicos.
    
    __A que público o cliente pertence:__
    
    {} pertence ao público alfa.
    
    __Descrição do público alfa:__
    
    O público alfa é predominantemente masculino e solteiro, 
    com idade mais frequente entre 30 e 39 anos. A maior parte do cliente desse grupo é graduado, 
    havendo uma significativa quantidade com ensino de base. A maior parte está empregada, 
    havendo uma significativa quantidade de empreendedores. 
    Sua renda média está em aproximadamente R$ 2.800,00.
    É o público mais empreendedor.
    
    '''.format(cliente))
   
  elif segmento == 1:
    st.markdown('''
    
    ---
    ##### Informação para a empresa
    
    Há cinco públicos.
    
    __A que público o cliente pertence:__
    
    {} pertence ao público beta.
    
    __Descrição do público beta:__
    
    O público beta  é predominantemente feminino, 
    em um relacionamento como casamento ou união estável. 
    Frequentemente, apresenta idades entre aproximadamente 25 e 30 anos. 
    Seus salários mais frequentes estão entre aproximadamente R$ 2.000,00 e R$ 2.800,00. 
    A maior parte é graduada, havendo uma maior quantidade de empregados e uma significativa quantidade de desempregados. 
    É o público mais novo.
    
    '''.format(cliente))
    
  elif segmento == 2:
    st.markdown('''
    
    ---
    ##### Informação para a empresa
    
    Há cinco públicos.
    
    __A que público o cliente pertence:__
    
    {} pertence ao público gamma.
    
    __Descrição do público gamma:__
    
    O público gamma é mais experiente do que os anteriores, 
    apresentando uma probabilidade mais alta para idades entre aproximadamente 42 e 59 anos 
    É um grupo predominantemente feminino, em uma relação como casamento ou união estável e apresenta especialização acadêmica. 
    Há maior frequência de salários entre R$ 2.000,00 e R$ 3.000,00. É o público mais feminino dentre os demais.
    
    '''.format(cliente))
    
  if segmento == 3:
    st.markdown('''
    ---
    ##### Informação para a empresa
    
    Há cinco públicos.
    
    __A que público o cliente pertence:__
    
    {} pertence ao público delta.
    
    __Descrição do público delta:__
    
    O público delta é unissex e solteiro, apresentando graduação, 
    com uma minoria significativa de pessoas com ensino básico. 
    A tendência é de idade entre aproximadamente 25 e 39 anos. 
    Sua renda predomina entre R$ 1.400,00 e R$ 2.400,00. 
    A maior parte está desempregada. É o público mais desempregado.
    
    '''.format(cliente))
    
  if segmento == 4:
    st.markdown('''
    ---
    ##### Informação para a empresa
    
    Há cinco públicos.
    
    __A que público o cliente pertence:__
    
    {} pertence ao público epsilon.
    
    __Descrição do público epsilon:__
    
    O público epsilon é predominantemente masculino e em relacionamento, 
    como casamento ou união estável, sendo o grupo mais novo, 
    com maior frequência de idade entre aproximadamente 25 e 30 anos. 
    A maior parte é graduada. A maior parte trabalha, 
    mas há significativa quantidade desempregada. É o público masculino mais desempregado.
    

    
    '''.format(cliente))
    
elif pag == 'Questão de negócio':
  st.markdown('#### Sistema em atualização')
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
  st.markdown('#### Sistema em atualização')
  st.header('Dashboard acerca dos públicos')
  st.subheader('Gráfico de segmentação')
  pca = Image.open('cluster_grafico.png')
  st.image(pca, use_column_width=True)
  st.markdown('''
  De acordo com o gráfico, os clusters não estão significativamente agrupados; 
  há um espalhamento dos dados, gerando um relevante mistura visual. 
  O expectado é um agrupamento melhormente definido. No entanto, há um razoável padrão. 
  O cluster azul não se encontra na altura do cluster violeta. O cluster vermelho se aproxima mais do cluster violeta. 
  O cluster laranja está em posição semelhante a do cluster azul, ficando abaixo da altura do cluster violeta. 
  O cluster roxo tende a estar à esquerda do cluster azul, a uma altura maior.
  ''')
  st.markdown(' ')
  st.markdown(' ') 
  st.markdown(' ')
  st.markdown(' ') 
                   
  st.subheader('Contagem de clientes em cada público/grupo')
  contagem = Image.open('freq_gp.png')
  st.image(contagem, use_column_width=True) 
  st.markdown('''
  Os clusters (públicos) com maiores quantidades de membros são o 0 (alfa) e o 1 (beta).
  O que apresenta menor quantidade é o 2 (gamma).
  ''')
  st.markdown(' ')
  st.markdown(' ') 
  st.markdown(' ')
  st.markdown(' ') 
  
  st.subheader('Público alfa')
  
  photo0 = Image.open('10_idade.png')
  st.image(photo0, use_column_width=True)
  st.markdown(' ')
  st.markdown(' ')
  
  photo1 = Image.open('11_renda.png')
  st.image(photo1, use_column_width=True)
  st.markdown(' ')
  st.markdown(' ')
  
  photo2 = Image.open('12_sexo.png')
  st.image(photo2, use_column_width=True)
  st.markdown('''
  
  __Legenda:__
  
  0 - Masculino
  
  1 - Feminino
  
  ''')
  st.markdown(' ')
  st.markdown(' ')
  
  photo3 = Image.open('13_civil.png')
  st.image(photo3, use_column_width=True)
  st.markdown('''
  
  __Legenda:__
  
  0 - Solteiro
  
  1 - Casado/União estável
  
  ''')
  st.markdown(' ')
  st.markdown(' ')
  
  photo4 = Image.open('14_ed.png')
  st.image(photo4, use_column_width=True)
  st.markdown('''
  
  __Legenda:__
  
  0 - Ensino básico
  
  1 - Graduado
  
  2 - Especializado
  
  3 - Doutor
  
  ''')
  st.markdown(' ')
  st.markdown(' ')
 
  photo5 = Image.open('15_oc.png')
  st.image(photo5, use_column_width=True)
  st.markdown('''
  
  __Legenda:__
  
  0 - Desempregado
  
  1 - Empregado
  
  2 - Empreendedor
  ''')
  st.markdown(' ')
  st.markdown(' ')

  st.subheader('Público beta')
  
  photo10 = Image.open('21_idade.png')
  st.image(photo10, use_column_width=True)
  st.markdown(' ')
  st.markdown(' ')
  
  photo11 = Image.open('22_renda.png')
  st.image(photo11, use_column_width=True)
  st.markdown(' ')
  st.markdown(' ')
  
  photo12 = Image.open('23_sexo.png')
  st.image(photo12, use_column_width=True)
  st.markdown('''
  
  __Legenda:__
  
  0 - Masculino
  
  1 - Feminino
  ''')
  st.markdown(' ')
  st.markdown(' ')
  
  photo13 = Image.open('24_civil.png')
  st.image(photo13, use_column_width=True)
  st.markdown('''
  
  __Legenda:__
  
  0 - Solteiro
  
  1 - Casado/União estável
  
  ''')
  st.markdown(' ')
  st.markdown(' ')
  
  photo14 = Image.open('25_ed.png')
  st.image(photo14, use_column_width=True)
  st.markdown('''
  
  __Legenda:__
  
  0 - Ensino básico
  
  1 - Graduado
  
  2 - Especializado
  
  3 - Doutor
  
  ''')
  st.markdown(' ')
  st.markdown(' ')
 
  photo15 = Image.open('26_oc.png')
  st.image(photo15, use_column_width=True)
  st.markdown('''
  
  __Legenda:__
  
  0 - Desempregado
  
  1 - Empregado
  
  2 - Empreendedor
  ''')
  st.markdown(' ')
  st.markdown(' ')

  st.subheader('Público gamma')
  
  photo20 = Image.open('31_idade.png')
  st.image(photo20, use_column_width=True)
  st.markdown(' ')
  st.markdown(' ')
  
  photo21 = Image.open('32_renda.png')
  st.image(photo21, use_column_width=True)
  st.markdown(' ')
  st.markdown(' ')
  
  photo22 = Image.open('33_sexo.png')
  st.image(photo22, use_column_width=True)
  st.markdown('''
  
  __Legenda:__
  
  0 - Masculino
  
  1 - Feminino
  
  ''')
  st.markdown(' ')
  st.markdown(' ')
  
  photo23 = Image.open('34_civil.png')
  st.image(photo23, use_column_width=True)
  st.markdown('''
  
  __Legenda:__
  
  0 - Solteiro
  
  1 - Casado/União estável
  ''')
  st.markdown(' ')
  st.markdown(' ')
  
  photo24 = Image.open('35_ed.png')
  st.image(photo24, use_column_width=True)
  st.markdown('''
  
  __Legenda:__
  
  0 - Ensino básico
  
  1 - Graduado
  
  2 - Especializado
  
  3 - Doutor
  
  ''')
  st.markdown(' ')
  st.markdown(' ')
 
  photo25 = Image.open('36_oc.png')
  st.image(photo25, use_column_width=True)
  st.markdown('''
  
  __Legenda:__
  
  0 - Desempregado
  
  1 - Empregado
  
  2 - Empreendedor
  ''')
  st.markdown(' ')
  st.markdown(' ')

   st.subheader('Público delta')
  
  photo40 = Image.open('41_idade.png')
  st.image(photo40, use_column_width=True)
  st.markdown(' ')
  st.markdown(' ')
  
  photo41 = Image.open('42_renda.png')
  st.image(photo41, use_column_width=True)
  st.markdown(' ')
  st.markdown(' ')
  
  photo42 = Image.open('43_sexo.png')
  st.image(photo42, use_column_width=True)
  st.markdown('''
  
  __Legenda:__
  
  0 - Masculino
  
  1 - Feminino
  
  ''')
  st.markdown(' ')
  st.markdown(' ')
  
  photo43 = Image.open('44_civil.png')
  st.image(photo43, use_column_width=True)
  st.markdown('''
  
  __Legenda:__
  
  0 - Solteiro
  
  1 - Casado/União estável
  ''')
  st.markdown(' ')
  st.markdown(' ')
  
  photo44 = Image.open('45_ed.png')
  st.image(photo44, use_column_width=True)
  st.markdown('''
  
  __Legenda:__
  
  0 - Ensino básico
  
  1 - Graduado
  
  2 - Especializado
  
  3 - Doutor
  
  ''')
  st.markdown(' ')
  st.markdown(' ')
 
  photo45 = Image.open('46_oc.png')
  st.image(photo45, use_column_width=True)
  st.markdown('''
  
  __Legenda:__
  
  0 - Desempregado
  
  1 - Empregado
  
  2 - Empreendedor
  ''')
  st.markdown(' ')
  st.markdown(' ')

  st.subheader('Público epsilon')
  
  photo50 = Image.open('51_idade.png')
  st.image(photo50, use_column_width=True)
  st.markdown(' ')
  st.markdown(' ')
  
  photo51 = Image.open('52_renda.png')
  st.image(photo51, use_column_width=True)
  st.markdown(' ')
  st.markdown(' ')
  
  photo52 = Image.open('53_sexo.png')
  st.image(photo52, use_column_width=True)
  st.markdown('''
  
  __Legenda:__
  
  0 - Masculino
  
  1 - Feminino
  
  ''')
  st.markdown(' ')
  st.markdown(' ')
  
  photo53 = Image.open('54_civil.png')
  st.image(photo53, use_column_width=True)
  st.markdown('''
  
  __Legenda:__
  
  0 - Solteiro
  
  1 - Casado/União estável
  ''')
  st.markdown(' ')
  st.markdown(' ')
  
  photo54 = Image.open('55_ed.png')
  st.image(photo24, use_column_width=True)
  st.markdown('''
  
  __Legenda:__
  
  0 - Ensino básico
  
  1 - Graduado
  
  2 - Especializado
  
  3 - Doutor
  
  ''')
  st.markdown(' ')
  st.markdown(' ')
 
  photo55 = Image.open('56_oc.png')
  st.image(photo55, use_column_width=True)
  st.markdown('''
  
  __Legenda:__
  
  0 - Desempregado
  
  1 - Empregado
  
  2 - Empreendedor
  ''')
  st.markdown(' ')
  st.markdown(' ')


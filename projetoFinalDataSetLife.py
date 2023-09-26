# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 22:59:33 2020

@author: Edson F Alves
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import matplotlib.pyplot as plt
#import numpy as np
#from sklearn import datasets, linear_model
#from sklearn.model_selection import train_test_split
#from sklearn.neural_network import MLPRegressor
#from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import mean_squared_error, max_error, r2_score

# Importar arquivo
dsOrig = pd.read_csv(r'D:\POS\BigDataSistemasInteligentesSENAI\PROJETO_INTELIGENCIA_COMPUTACIONAL\life.csv')
ds = pd.read_csv(r'D:\POS\BigDataSistemasInteligentesSENAI\PROJETO_INTELIGENCIA_COMPUTACIONAL\life.csv')


ds = ds.drop(columns=['Status'])

###ds = ds.loc[ds['Country'] == 'Brazil']


#Colunas selecionadas = 
#Country
#Adult Mortality (Taxas de mortalidade em adultos de ambos os sexos (probabilidade de morrer entre 15 e 60 anos por 1000 habitantes))
#infant deaths
#Alcohol
#Hepatitis B
#Measles (Sarampo)
#under-five deaths (Número de mortes abaixo de cinco anos por 1000 habitantes)
#Polio
#GDP (Produto interno bruto per capita (em USD))
#Population
#Income composition of resources (Índice de Desenvolvimento Humano em termos de composição de renda dos recursos (índice de 0 a 1))
#Schooling (Número de anos de escolaridade (anos))


#separando as colunas
X = ds[['Country','Adult Mortality','infant deaths','Alcohol','Hepatitis B','Measles ','under-five deaths ','Polio','GDP','Population','Income composition of resources','Schooling']]
y = ds['Life expectancy ']


#utilizar a media do pais em especifico ao inves da media geral

X.fillna(X.mean(), inplace=True)

y.fillna(y.mean(), inplace=True)



Country_dummy=pd.get_dummies(X['Country'])

X.drop(['Country'],inplace=True,axis=1)

X=pd.concat([X,Country_dummy],axis=1)


############## SEPARANDO O DATASET 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

################################# REGRESSÃO LINEAR ################################

regr = linear_model.LinearRegression()


regr.fit(X_train, y_train)


regr.score(X_test, y_test)


y_pred = regr.predict(X_test)


#GRÁFICO RL

plt.scatter(range(len(y_test)), y_test,  color='black')
plt.plot(range(len(y_test)), regr.predict(X_test), color='blue', linewidth=3)
plt.title('Regressao Linear')
plt.xticks(())
plt.yticks(())
plt.show()



################# CROSS VALIDATE MODELO LINEAR 

cv_results = cross_validate(regr, X_train, y_train, cv=10)
print(cv_results)

cv_score = cross_val_score(regr, X_train, y_train, cv=10)
print(cv_score)

avg_score = np.mean(cv_score)
print(avg_score)



################# MÉTRICAS REGRESSÂO LINEAR

print(mean_squared_error(y_test,y_pred)**(0.5))


max_error(y_test,y_pred)

r2_score(y_test,y_pred)


y_diff = list(abs(y_test-y_pred))


item_diff = y_diff.index(max_error(y_test,y_pred))

X_test.iloc[item_diff,:]

X_test.describe()



#################### RANDOM FOREST #############################

from sklearn.ensemble import RandomForestRegressor


regrK = RandomForestRegressor(random_state=0)


regrK.fit(X_train, y_train)


regrK.score(X_test, y_test)


y_pred1 = regrK.predict(X_test)


#GRÁFICO RF

plt.scatter(range(len(y_test)), y_test,  color='black')
plt.plot(range(len(y_test)), regrK.predict(X_test), color='blue', linewidth=3)
plt.title('Random Forest')
plt.xticks(())
plt.yticks(())
plt.show()



################# CROSS VALIDATE RANDOM FOREST

cv_results2 = cross_validate(regrK, X_train, y_train, cv=10)
print(cv_results2)


cv_score2 = cross_val_score(regrK, X_train, y_train, cv=10)
print(cv_score2)


avg_score2 = np.mean(cv_score2)
print(avg_score2)



################# MÉTRICAS RANDOM FOREST

print(mean_squared_error(y_test,y_pred1)**(0.5))


max_error(y_test,y_pred1)

r2_score(y_test,y_pred1)


y_diff = list(abs(y_test-y_pred1))


item_diff = y_diff.index(max_error(y_test,y_pred1))

X_test.iloc[item_diff,:]

X_test.describe()

################################# PREVISÃO BRASIL

x2 = pd.Series(y_pred1, index=X_test['Brazil'])

x3 = x2.drop([0])

x3.mean()

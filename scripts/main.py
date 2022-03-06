# %% [markdown]
# # Regresión Logística

# %%
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

# %% [markdown]
# La regresión logística es un tipo de análisis de regresión utilizado para predecir el resultado de una variable categórica (una variable que puede adoptar un número limitado de categorías) en función de las variables independientes o predictoras.

# %% [markdown]
# Aproximadamente el 70% de los problemas en el Machine Learning son de clasificación.

# %%
manzanas = [30,70]
nombres = ['Problemas de regresión', 'Problemas de clasificación']
plt.pie(manzanas, labels=nombres, autopct="%0.1f %%")
plt.axis("equal")
plt.show()

# %% [markdown]
# La regresión logística es un método de regresión útil para resolver problemas de clasificación binaria.

# %% [markdown]
# Es un método estadístico para predecir clases binarias.

# %% [markdown]
# Nos ayuda a clasificar nuestros registros en dos o más categorías:

# %% [markdown]
# - Predecir si un cliente aleatorio que entra en la tienda va a comprar un producto en particular basándonos en sus ingresos, género, historial de compras, historial de publicidad...
# - Predecir si un equipo de futbol va a ganar o perder un partido sabiendo el rival, los detalles del equipo, el tiempo que va a hacer, la alineación, el estadio, las horas de entreno...

# %% [markdown]
# La función logística es una curva que puede tomar cualquier número de valor real y asignar cualquier valor entre 0 y 1.
# - Si la curva va a infinito POSITIVO la predicción se convertirá en 1.
# - Si la curva va a infinito NEGATIVO la predicción se convertirá en 0.

# %%
def sigmoid(x):
    a = []
    for item in x:
        a.append(1/(1+math.exp(-item)))
    return a
x = np.arange(-10., 10., 0.2)
sig = sigmoid(x)
plt.axhline(y=0.5, color="red")
plt.plot(x,sig)
plt.show()

# %% [markdown]
# - Si la salida de la función logística es mayor o igual que 0.5 podemos clasificar el resultado como 1.
# - Si la salida de la función logística es menor que 0.5 podemos clasificar el resultado como 0.

# %% [markdown]
# Por su parte si el resultado es 0.75 podemos decir en términos de probabilidad que hay un 75% de probabilidad que la salida de la función sea 1.

# %% [markdown]
# ### Regresión Lineal vs Regresión Logística

# %% [markdown]
# La primera diferencia es que la variable y ya no es continua, sino discreta.

# %% [markdown]
# $$y\in\{0, 1\}$$
# $$P\in [0, 1]$$
# $$X\in [-\infty, \infty]$$

# %% [markdown]
# | Regresión lineal (salida continua)              | Regresión logística (salida discreta)        |
# | ------------------------------------------------| -------------------------------------------- |
# | Conocer el porcentaje de probabilidad de lluvia | Conocer si va a llover o no                  |
# | Conocer el precio de una acción                 | Saber si el precio de una acción subirá o no |

# %% [markdown]
# ## Implementación de la regresión logística en python

# %% [markdown]
# ### Limpieza de datos

# %%
data = pd.read_csv('../data/bank.csv', sep = ';')

# %%
data.head()

# %%
data.shape

# %%
data.columns.values

# %%
data['y'] = (data['y'] == 'yes').astype(int)

# %%
data.tail()

# %%
data['education'].unique()

# %%
data['education'] = np.where(data['education']=='basic.4y', 'Basic', data['education'])
data['education'] = np.where(data['education']=='basic.6y', 'Basic', data['education'])
data['education'] = np.where(data['education']=='basic.9y', 'Basic', data['education'])
data['education'] = np.where(data['education']=='high.school', 'High School', data['education'])
data['education'] = np.where(data['education']=='professional.course', 'Profesional Course', data['education'])
data['education'] = np.where(data['education']=='university.degree', 'University Degree', data['education'])
data['education'] = np.where(data['education']=='illiterate', 'Illiterate', data['education'])
data['education'] = np.where(data['education']=='unknown', 'Unknown', data['education'])

# %%
data['education'].unique()

# %% [markdown]
# ### Análisis de los datos

# %%
data['y'].value_counts()

# %%
data.groupby('y').mean()

# %%
data.groupby('education').mean()

# %%
pd.crosstab(data.education, data.y).plot(kind='bar')
plt.title('Frecuencia de compra en función del nivel de educación')
plt.xlabel('Nivel de educación')
plt.ylabel('Frecuencia de compra')
plt.show()

# %%
table = pd.crosstab(data.marital, data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Diagrama apilado de estado civil contra el nivel de compras')
plt.xlabel('Estado civil')
plt.ylabel('Proporción de clientes')

# %%
table = pd.crosstab(data.day_of_week, data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Frecuencia de compra en función del día de la semana')
plt.xlabel('Día de la semana')
plt.ylabel('Frecuencia de compra del producto')
plt.show()

# %%
table = pd.crosstab(data.month, data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Frecuencia de compra en función del mes')
plt.xlabel('Mes del año')
plt.ylabel('Frecuencia de compra del producto')
plt.show()

# %%
table = pd.crosstab(data.month, data.y).plot(kind='bar')
plt.title('Frecuencia de compra en función del mes')
plt.xlabel('Mes del año')
plt.ylabel('Frecuencia de compra del producto')
plt.show()

# %%
data.age.hist()
plt.title('Histograma de la edad')
plt.xlabel('Edad')
plt.ylabel('Cliente')
plt.show()

# %% [markdown]
# ### Selección de variables para el modelo logístico

# %% [markdown]
# #### Conversión de las variables categóricas a dummies

# %%
categories = ['job', 'marital', 'education', 'housing', 'loan', 'contact',
              'month', 'day_of_week', 'poutcome']
for category in categories:
    cat_list = 'cat' + '_' + category
    cat_dummies = pd.get_dummies(data[category], prefix=cat_list)
    data = data.join(cat_dummies)

# %%
data.columns.values

# %%
data.head()

# %%
data_vars = data.columns.values.tolist()

# %%
to_keep = [v for v in data_vars if v not in categories]
to_keep = [v for v in to_keep if v not in ['default']]

# %%
bank_data = data[to_keep]
bank_data.columns.values

# %%
bank_data_vars = bank_data.columns.values.tolist()
Y = ['y']
X = [v for v in bank_data_vars if v not in Y]

# %% [markdown]
# ### Selección de rasgos para el modelo

# %%
n = 12

# %%
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# %%
lr = LogisticRegression(solver='liblinear')

# %%
rfe = RFE(lr, n)
rfe.fit(bank_data[X], bank_data[Y].values.ravel())

# %%
rfe.support_

# %%
rfe.ranking_

# %%
z = zip(bank_data_vars, rfe.support_, rfe.ranking_)

# %%
list(z)

# %%
cols = ['previous', 'euribor3m', 'cat_job_blue-collar', 'cat_job_retired', 'cat_month_aug',
       'cat_month_dec', 'cat_month_jul', 'cat_month_jun', 'cat_month_mar', 'cat_month_nov',
       'cat_day_of_week_wed', 'cat_poutcome_nonexistent']

# %%
X = bank_data[cols]
Y = bank_data['y']

# %% [markdown]
# ### Implementación del modelo de regresión logística con statsmodel.api

# %%
import statsmodels.api as sm

# %%
logit_model = sm.Logit(Y, X)

# %%
result = logit_model.fit()

# %%
result.summary2()

# %% [markdown]
# ### Implementación del modelo en Python con scikit-learn

# %%
from sklearn import linear_model

# %%
logit_model = linear_model.LogisticRegression()
logit_model.fit(X, Y)

# %%
# Factor de R^2
logit_model.score(X, Y)

# %%
Y.mean()

# %%
1-Y.mean()

# %%
pd.DataFrame(list(zip(X.columns, np.transpose(logit_model.coef_))))

# %% [markdown]
# ### Validación del modelo logístico

# %%
from sklearn.model_selection import train_test_split

# %%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# %%
lm = linear_model.LogisticRegression()
lm.fit(X_train, Y_train)

# %% [markdown]
# 
# $$Y_p=\begin{cases}0& si\ p\leq0.5\\1&si\ p >0.5\end{cases}$$

# %% [markdown]
# Sin embargo, se puede decidir cortar por otro lado, 0.5 es un poco drástico. A veces, sobre todo en un dataset tan desviado como este donde hay tan poca gente con éxito (solo un 10%), nos interesará definir un umbral.

# %%
# Hacemos la predicción con el conjunto de testing
prediction = lm.predict(X_test)

# %%
# Probabilidades menores a 0.5 es 0
prediction

# %% [markdown]
# En nuestro caso, tenemos solo un 10% de clientes que compran el producto, entonces establecer la probabilidad de epsilon en 0.10 PUEDE ser un buen umbral de decisión.

# %% [markdown]
# $$ \varepsilon\in (0, 1), Y_p=\begin{cases}0& si\ p\leq\varepsilon\\1&si\ p >\varepsilon\end{cases} $$

# %%
probs = lm.predict_proba(X_test)
prob = probs[:,1]
prob_df = pd.DataFrame(prob)
threshold = 0.1
prob_df['prediction'] = np.where(prob_df[0]>threshold, 1, 0)
prob_df.head()

# %%
pd.crosstab(prob_df.prediction, columns='count')

# %% [markdown]
# ### Eficacia del modelo

# %%
from sklearn import metrics

# %%
metrics.accuracy_score(Y_test, prediction)

# %% [markdown]
# #### La eficacia de nuestro modelo logístico es del 90.13%



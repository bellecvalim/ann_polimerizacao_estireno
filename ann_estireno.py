import numpy as np
from numpy import linalg as la
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing as pre
import matplotlib.pyplot as plt
#from mlxtend.plotting import plot_learning_curves

#Carregando e separando dados de entrada e saída pra rede
ent1, ent2, ent3 = np.loadtxt('dados_rede.txt', delimiter = '\t', usecols=(0,1,2), unpack = True)
sai1, sai2, sai3 = np.loadtxt('dados_rede.txt', delimiter = '\t', usecols=(3,4,5), unpack = True)

#Normalizando os dados de entrada e saída entre 0 e 1
ent1_norm = np.zeros(len(ent1))
ent2_norm = np.zeros(len(ent2))
ent3_norm = np.zeros(len(ent3))
sai1_norm = np.zeros(len(sai1))
sai2_norm = np.zeros(len(sai2))
sai3_norm = np.zeros(len(sai3))
for i in range (0, len(ent1)):
    ent1_norm[i] = (ent1[i]-np.min(ent1))/(np.max(ent1)-np.min(ent1))
    ent2_norm[i] = (ent2[i]-np.min(ent2))/(np.max(ent2)-np.min(ent2))
    ent3_norm[i] = (ent3[i]-np.min(ent3))/(np.max(ent3)-np.min(ent3))
    sai1_norm[i] = (sai1[i]-np.min(sai1))/(np.max(sai1)-np.min(sai1))
    sai2_norm[i] = (sai2[i]-np.min(sai2))/(np.max(sai2)-np.min(sai2))
    sai3_norm[i] = (sai3[i]-np.min(sai3))/(np.max(sai3)-np.min(sai3))

X = np.array((ent1_norm, ent2_norm, ent3_norm), dtype = np.float)
Y = np.array((sai1_norm, sai2_norm, sai3_norm), dtype = np.float)

#Função pra truncar o valor de R² 
def truncate(f, n):
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

#Fazendo a transposta da matriz de entradas
X_norm_trans = np.transpose(X)

#Treinando o modelo pra primeira saída
Saida1 = Y[0]

#Separa randomicamente o conjunto de dados para treinamento e teste da rede
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X_norm_trans, Saida1, test_size = 0.3, random_state = 0)

#Etapa de treinamento da rede
scaler = pre.StandardScaler()
X_train_scaled = scaler.fit_transform(X_treino)

rede = MLPRegressor(max_iter=500, learning_rate_init=0.1, random_state=1, solver='lbfgs', 
                    tol=0.001, hidden_layer_sizes= (5), activation= 'tanh')
Y_treino1 = np.ravel(Y_treino)
Y_calc = rede.fit(X_train_scaled, Y_treino1)

#Pegando pesos e bias
pesos = rede.coefs_[0] #Entre camada de entrada e primeira camada intermediária
#pesos2 = rede.coefs_[1] #Entre segunda camada intermediária e camada final
bias_int = rede.intercepts_[0] #Da primeira camada intermediária
#bias_int2 = rede.intercepts_[1] #Da segunda camada intermediária
bias_final = rede.intercepts_[1] #Da camada final
print ('Pesos entre entrada e camada 1 = \n', pesos)
#print ('Pesos entre camada 1 e camada 2 = \n', pesos2)
print('Bias_Int = \n', bias_int)
#print('Bias_Int2 = \n', bias_int2)
print('Bias_Final = \n', bias_final)

#Etapa de teste da rede
X_test_scaled = scaler.fit_transform(X_teste)
Y_teste1 = np.ravel(Y_teste)
Y_calc = rede.predict(X_test_scaled)

#Cálculo do MSE e R2
mse = truncate(mean_squared_error(Y_teste1, Y_calc),4)
r2 = truncate(r2_score(Y_teste1, Y_calc), 4)
print('MSE = \n', mse)
print('R2 = \n', r2)

#Plot do gráfico de regressão para a etapa de teste
titulo = 'Mn Observed vs Predicted - R² = ' + str(r2)
fig, ax = plt.subplots()
ax.scatter(Y_teste1, Y_calc, color = 'salmon', edgecolor = 'indianred' )
ax.plot([0, 1], [0, 1], 'k--', lw=4)
plt.title(titulo)
ax.set_xlabel('Mn Observed')
ax.set_ylabel('Mn Predicted')

#Plot do gráfico preditos vs observados para a etapa de teste
sample = np.zeros(len(Y_calc))
for i in range(0,len(Y_calc)):
    sample[i] = i+1

fig2, ax = plt.subplots()
ax.scatter(sample, Y_teste1, label = 'Mn Observed', color = 'violet', edgecolor = 'mediumvioletred')
ax.scatter(sample, Y_calc, label = 'Mn Predicted', color = 'royalblue', marker = '*',  edgecolor = 'navy')

plt.title('Mn vs Simulations')
plt.legend()
ax.set_xlabel('Simulations')
ax.set_ylabel('Mn')
plt.show() 
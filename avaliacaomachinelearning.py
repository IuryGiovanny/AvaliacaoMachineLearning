# Importação das bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# 1. Carregamento do Dataset (Seguro Médico)
url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
data = pd.read_csv(url)

print("Visualização das primeiras linhas:")
display(data.head())

# 2. Pré-processamento
# Convertendo variáveis categóricas (sex, smoker, region) para numéricas
le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])
data['smoker'] = le.fit_transform(data['smoker'])
data['region'] = le.fit_transform(data['region'])

# Definindo X (features) e y (target - charges/custo)
X = data.drop('charges', axis=1)
y = data['charges']

# Divisão em Treino e Teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Aplicação dos Modelos

# Modelo Paramétrico: Regressão Linear
# Assume uma relação linear (reta/plano) entre as variáveis.
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)

# Modelo Não Paramétrico: Árvore de Decisão
# Não assume distribuição específica, divide os dados em nós baseados em regras.
tree_reg = DecisionTreeRegressor(random_state=42, max_depth=5) # max_depth limitado para evitar overfitting
tree_reg.fit(X_train, y_train)
y_pred_tree = tree_reg.predict(X_test)

# 4. Avaliação e Comparação
def avaliar_modelo(nome, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"--- {nome} ---")
    print(f"RMSE (Erro Médio): {rmse:.2f}")
    print(f"R² (Precisão): {r2:.4f}\n")

avaliar_modelo("Regressão Linear (Paramétrico)", y_test, y_pred_lin)
avaliar_modelo("Árvore de Decisão (Não Paramétrico)", y_test, y_pred_tree)

# 5. Visualização Gráfica (Exigência do Trabalho)
plt.figure(figsize=(14, 6))

# Gráfico Regressão Linear
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_lin, alpha=0.5, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Valor Real')
plt.ylabel('Previsão')
plt.title('Regressão Linear (Paramétrico)')

# Gráfico Árvore de Decisão
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_tree, alpha=0.5, color='green')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Valor Real')
plt.ylabel('Previsão')
plt.title('Árvore de Decisão (Não Paramétrico)')

plt.tight_layout()
plt.show()

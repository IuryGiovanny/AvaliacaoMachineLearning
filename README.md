# Comparação de Modelos: Paramétricos vs Não Paramétricos 📊

Este repositório contém a atividade prática referente à **2ª Avaliação da disciplina de Machine Learning** do curso de Análise e Desenvolvimento de Sistemas (UNINASSAU).

**Aluno:** Iury Giovanny Gomes da Costa  
**Matrícula:** 01751602
**Data:** 13/12/2025

---

## 🎯 Objetivo
O objetivo deste trabalho é aplicar e comparar técnicas de Machine Learning para resolver um problema real de regressão, identificando as diferenças de performance entre:
1.  **Modelo Paramétrico:** Regressão Linear (`LinearRegression`).
2.  **Modelo Não Paramétrico:** Árvore de Decisão (`DecisionTreeRegressor`).

## 🏥 O Problema (Dataset)
Foi utilizado o dataset **Medical Cost Personal Datasets** (Seguro Médico). O objetivo é prever o valor da cobrança do seguro (`charges`) com base em características do paciente.

**Variáveis utilizadas:**
* `age`: Idade do beneficiário.
* `sex`: Gênero.
* `bmi`: Índice de Massa Corporal (IMC).
* `children`: Número de filhos/dependentes.
* `smoker`: Se é fumante ou não.
* `region`: Região residencial.

## 🛠️ Tecnologias Utilizadas
O projeto foi desenvolvido em **Python** utilizando o ambiente **Google Colab**.
* **Pandas & NumPy:** Manipulação de dados.
* **Matplotlib & Seaborn:** Visualização de dados e gráficos.
* **Scikit-Learn:** Criação, treinamento e avaliação dos modelos de Machine Learning.

## 🚀 Como Executar
O código principal está contido no arquivo `.ipynb` (Jupyter Notebook) neste repositório.

1.  Clique no arquivo `.ipynb` acima.
2.  Você pode visualizar o código diretamente no GitHub ou clicar no botão "Open in Colab" para executar.

## 📊 Resultados Obtidos

Os modelos foram avaliados utilizando as métricas **RMSE** (Raiz do Erro Quadrático Médio) e **R²** (Coeficiente de Determinação).

| Modelo | Tipo | R² Score (Aproximado) | Observação |
| :--- | :--- | :--- | :--- |
| **Regressão Linear** | Paramétrico | [INSERIR O VALOR DO R² AQUI, EX: 0.78] | Captura a tendência geral, mas sofre com dados não lineares. |
| **Árvore de Decisão** | Não Paramétrico | [INSERIR O VALOR DO R² AQUI, EX: 0.85] | Melhor adaptação a padrões complexos (ex: impacto de fumantes). |

> **Conclusão:** O modelo [DIGA QUAL FOI MELHOR: Árvore ou Linear] apresentou melhor desempenho para este conjunto de dados, pois conseguiu capturar as nuances não lineares das variáveis, especialmente a variável `smoker` (fumante).

---

## 📝 Estrutura do Projeto
* `notebook.ipynb`: Código fonte completo com a implementação e gráficos.
* `README.md`: Documentação do projeto.

---
*Projeto desenvolvido para fins acadêmicos - Uninassau Natal.*

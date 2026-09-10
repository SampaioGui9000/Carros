# Análise de Veículos Elétricos (EV Analytics)

Projeto de análise de dados e machine learning sobre um dataset de veículos elétricos, dividido em três partes:

1. **Exploração estatística e comparação de grupos**
2. **Tendências de mercado ao longo do tempo**
3. **Modelagem preditiva do valor de revenda (`Resale_Value_USD`)**

## Objetivo

Entender o perfil dos veículos elétricos na base, identificar diferenças estatísticas relevantes entre grupos (tipo de veículo, região, tipo de uso), observar a evolução de indicadores de negócio ao longo dos anos (2015–2024) e construir um modelo de machine learning capaz de estimar o valor de revenda de um veículo a partir de suas características técnicas e de uso.

## Dataset

O arquivo `data/electric_vehicle_analytics.csv` contém 3.000 registros de veículos elétricos, com 25 colunas, incluindo:

- **Identificação:** `Vehicle_ID`, `Make`, `Model`, `Year`, `Region`, `Vehicle_Type`, `Usage_Type`
- **Características técnicas:** `Battery_Capacity_kWh`, `Battery_Health_%`, `Range_km`, `Charging_Power_kW`, `Charging_Time_hr`, `Charge_Cycles`, `Energy_Consumption_kWh_per_100km`, `Mileage_km`, `Avg_Speed_kmh`, `Max_Speed_kmh`, `Acceleration_0_100_kmh_sec`, `Temperature_C`
- **Custos e sustentabilidade:** `CO2_Saved_tons`, `Maintenance_Cost_USD`, `Insurance_Cost_USD`, `Electricity_Cost_USD_per_kWh`, `Monthly_Charging_Cost_USD`
- **Variável-alvo:** `Resale_Value_USD` (valor de revenda em dólares)

Não há valores nulos na base.

## Como executar

### Opção 1 — Google Colab
1. Faça upload do notebook `EV_Analytics_Notebook.ipynb` no [Google Colab](https://colab.research.google.com/).
2. Faça upload do arquivo `data/electric_vehicle_analytics.csv` (ou monte o Google Drive) mantendo o caminho relativo `data/electric_vehicle_analytics.csv`, ou ajuste o caminho na primeira célula de carregamento dos dados.
3. Execute todas as células em sequência (`Ambiente de execução > Executar tudo`).

### Opção 2 — Jupyter Notebook local
```bash
# Clonar o repositório
git clone <URL_DO_REPOSITORIO>
cd <NOME_DO_REPOSITORIO>

# (Opcional) criar um ambiente virtual
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Instalar dependências
pip install -r requirements.txt

# Abrir o notebook
jupyter notebook EV_Analytics_Notebook.ipynb
```

Execute as células em ordem (`Cell > Run All` ou `Kernel > Restart & Run All`).

## Bibliotecas necessárias

- pandas
- numpy
- matplotlib
- seaborn
- scipy
- scikit-learn

Todas listadas em `requirements.txt`.

## Resumo do conteúdo do notebook

### Parte 1 — Exploração estatística e comparação de grupos
- Análise exploratória inicial (dimensão, tipos de dados, nulos, estatísticas descritivas).
- Distribuições de variáveis numéricas e categóricas.
- Testes ANOVA comparando `Resale_Value_USD` entre grupos (`Vehicle_Type`, `Region`, `Usage_Type`).
- Teste Qui-quadrado de associação entre `Vehicle_Type` e `Region`.
- Matriz de correlação entre variáveis numéricas.

### Parte 2 — Tendências de mercado ao longo do tempo
- Agregações anuais de indicadores de negócio (valor de revenda, CO₂ economizado, custos, autonomia).
- Gráficos de evolução temporal (2015–2024).
- Projeção com regressão linear simples para os anos de 2025–2027.
- Seção dedicada "Tendências de Mercado" com os principais achados.

### Parte 3 — Modelagem preditiva
- Preparação de variáveis (One-Hot Encoding para categóricas, padronização para numéricas) via `ColumnTransformer` e `Pipeline`.
- Treinamento e comparação de 4 algoritmos de regressão: Regressão Linear, Ridge, Random Forest e Gradient Boosting.
- Combinação de modelos com `VotingRegressor` (ensemble).
- Avaliação por R², MAE e RMSE.
- Análise de importância das variáveis (feature importance).
- Seção "Recomendações para o Negócio" com implicações práticas dos resultados.

## Principais resultados

- O valor de revenda é fortemente explicado pela **capacidade da bateria** e pelo **ano do veículo**, enquanto tipo de veículo e tipo de uso não têm efeito estatisticamente significativo.
- Os indicadores de negócio mostram uma tendência clara de valorização do valor médio de revenda ao longo dos anos.
- Os modelos de regressão testados atingem um R² próximo de 0,90 no conjunto de teste, com destaque para o modelo Ensemble (Voting).

## Licença dos dados

Dataset de uso educacional, incluído neste repositório em `data/electric_vehicle_analytics.csv`.
